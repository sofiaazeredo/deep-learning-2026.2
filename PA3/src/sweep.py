"""Escolha de hiperparâmetros SÓ no val (o test_public nem é carregado aqui).

Busca aleatória sobre configs/sweeps/{method}.json. Retomável (cada trial×seed vira um JSON; trial pronto é
pulado), então sobrevive à desconexão do Colab. No fim, o melhor trial (maior val F1 macro médio entre seeds,
desempate pela val loss) vai para configs/selected/{method}_b{budget}.json, que é o que part0/fixmatch/run_final usam.

  python -m src.sweep --data dados --method part0_ft --budget 5 --seeds 0 1
  python -m src.sweep --data dados --method fixmatch --budget 5 --seeds 0 --set train.steps=1500 --n_trials 8
"""
import argparse
import gc
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import common as C

BACKBONE_KEYS = ("model.arch", "data.size", "data.resize_mode")


def sample_space(space, rng):
    out = {}
    for k, spec in space.items():
        if "choices" in spec:
            out[k] = rng.choice(spec["choices"])
        elif "loguniform" in spec:
            lo, hi = spec["loguniform"]
            out[k] = float(f"{math.exp(rng.uniform(math.log(lo), math.log(hi))):.3g}")
        elif "uniform" in spec:
            lo, hi = spec["uniform"]
            out[k] = float(f"{rng.uniform(lo, hi):.3g}")
        else:
            raise ValueError(f"spec inválida para {k}: {spec}")
    return out


def make_trials(space, n_trials, sweep_seed):
    """Trial 0 = defaults (sem overrides). Os demais são amostras aleatórias, sem repetição."""
    rng = random.Random(sweep_seed)
    trials, seen = [{}], {C.cfg_hash({"o": {}})}
    guard = 0
    while len(trials) < n_trials and guard < 50 * n_trials:
        guard += 1
        t = sample_space(space, rng)
        h = C.cfg_hash({"o": t})
        if h not in seen:
            seen.add(h)
            trials.append(t)
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", required=True, choices=["part0_ft", "part0_scratch", "fixmatch"])
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--space", default=None, help="default: configs/sweeps/{method}.json")
    ap.add_argument("--n_trials", type=int, default=None)
    ap.add_argument("--sweep_seed", type=int, default=0)
    ap.add_argument("--max_minutes", type=float, default=None, help="para de iniciar trials depois disso")
    ap.add_argument("--out", default="resultados/sweeps")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--allow_defaults", action="store_true", help="backbone do fixmatch a partir de defaults")
    ap.add_argument("--no_select", action="store_true", help="só resumo, não escreve configs/selected")
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b=v", help="override FIXO em todos os trials")
    a = ap.parse_args()

    spec = C.load_json(a.space or C.ROOT / "configs" / "sweeps" / f"{a.method}.json")
    space = dict(spec["space"])
    base = C.method_cfg(a.method, a.budget, a.set, a.allow_defaults, use_selected=False)
    if a.method == "part0_ft" and a.budget != C.project()["orcamento_padrao"]:
        bb, src = C._backbone_from_part0(a.allow_defaults)         # mesmo backbone em todos os orçamentos
        base = C.deep_update(base, bb)
        space = {k: v for k, v in space.items() if k not in BACKBONE_KEYS}
        print(f"backbone/data travados em {src}")
    fixed = C.parse_overrides(a.set)
    space = {k: v for k, v in space.items() if k not in fixed}
    n_trials = a.n_trials or spec["n_trials"]
    trials = make_trials(space, n_trials, a.sweep_seed)

    out = Path(a.out) / f"{a.method}_b{a.budget}"
    (out / "trials").mkdir(parents=True, exist_ok=True)
    cache = C.DataCache(a.data, a.cache)
    need_unl = a.method == "fixmatch"
    runner = __import__("src.fixmatch", fromlist=["run"]) if need_unl else __import__("src.part0", fromlist=["run"])
    t_start = time.time()
    print(f"{a.method} b{a.budget}: {len(trials)} trials × seeds {a.seeds} | config-base: {base['_source']}")
    for ti, ov in enumerate(trials):
        cfg = C.apply_overrides(base, ov)
        h = C.cfg_hash(cfg)
        for seed in a.seeds:
            f = out / "trials" / f"{h}_s{seed}.json"
            if f.exists() and not C.load_json(f).get("oom"):     # trial em OOM é retentado ao retomar
                continue
            if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                print(f"--max_minutes atingido; pare aqui e rode de novo para continuar ({ti}/{len(trials)})")
                break
            print(f"\n--- trial {ti} [{h}] seed {seed}: {ov or 'defaults'}", flush=True)
            data = cache.get(a.budget, cfg, need_unl=need_unl, need_test=False)
            kw = dict(method=a.method) if not need_unl else {}
            oom, res = False, None
            try:
                res = runner.run(cfg, data, a.budget, seed, None, False, verbose=False, **kw)
            except torch.cuda.OutOfMemoryError:
                oom = True
            gc.collect()                                  # libera modelo/otimizador do trial anterior
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if oom:   # registra e segue: um trial sem memória não derruba o sweep
                C.save_json(dict(oom=True, overrides=ov, cfg=cfg, cfg_hash=h, seed=seed), f)
                print("    SEM MEMÓRIA na GPU: trial registrado como OOM (use tests.probe_memory e reduza train.bs)",
                      flush=True)
                continue
            C.save_json(dict(overrides=ov, cfg=cfg, **res), f)
            print(f"    val F1 {res['val_f1']:.3f} | val loss {res['val_loss']:.3f} | {res['minutes']:.1f} min",
                  flush=True)

    rows, n_oom = [], 0
    for f in sorted((out / "trials").glob("*.json")):
        r = C.load_json(f)
        if r.get("oom"):
            n_oom += 1
            continue
        rows.append(dict(cfg_hash=r["cfg_hash"], seed=r["seed"], val_f1=r["val_f1"], val_loss=r["val_loss"],
                         minutes=r["minutes"], **{f"o.{k}": (str(v) if isinstance(v, list) else v)
                                                   for k, v in r["overrides"].items()}))
    if n_oom:
        print(f"\nAVISO: {n_oom} execuções (trial×seed) terminaram em OOM e ficaram fora do resumo. "
              "Rode `python -m tests.probe_memory` e fixe um train.bs que caiba (--set train.bs=N); "
              "como isso muda a config-base, os trials são refeitos com o novo lote.")
    if not rows:
        raise SystemExit("nenhum trial concluído" + (" (todos em OOM)" if n_oom else ""))
    df = pd.DataFrame(rows)
    okeys = [c for c in df.columns if c.startswith("o.")]
    g = df.groupby("cfg_hash").agg(n_seeds=("seed", "nunique"), val_f1_mean=("val_f1", "mean"),
                                   val_f1_std=("val_f1", "std"), val_loss=("val_loss", "mean"),
                                   minutes=("minutes", "mean"), **{k: (k, "first") for k in okeys}).reset_index()
    g = g.sort_values(["val_f1_mean", "val_loss"], ascending=[False, True])
    g.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nTop trials (val F1 macro médio; diferenças menores que o desvio são ruído):")
    print(g.head(8).round(4).to_string(index=False))
    done = g[g.n_seeds == len(a.seeds)]
    if done.empty or a.no_select:
        return
    best = done.iloc[0]
    rec = C.load_json(next((out / "trials").glob(f"{best.cfg_hash}_s*.json")))
    cfg = rec["cfg"]
    cfg["_selecao"] = dict(metodo=a.method, orcamento=a.budget, escolhido_no="val", val_f1_mean=float(best.val_f1_mean),
                           val_f1_std=None if pd.isna(best.val_f1_std) else float(best.val_f1_std),
                           seeds=a.seeds, n_trials=int(len(g)), resumo=str(out / "summary.csv"))
    cfg.pop("_source", None)
    cfg.pop("_status", None)          # o escolhido foi validado no val; não herda o aviso dos defaults
    dest = C.SELECTED / f"{a.method}_b{a.budget}.json"
    C.save_json(cfg, dest)
    d0 = g[g.cfg_hash == C.cfg_hash(C.apply_overrides(base, {}))]
    if len(d0):
        print(f"\nDefaults (chute inicial): val F1 {d0.val_f1_mean.iloc[0]:.3f}  ->  selecionado: {best.val_f1_mean:.3f}")
    print(f"Selecionado salvo em {dest}")


if __name__ == "__main__":
    main()
