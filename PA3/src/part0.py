"""Parte 0 — o piso. (a) CNN do zero e (b) backbone ImageNet + fine-tuning, só com rotulos_{budget}.csv.

Hiperparâmetros vêm de configs/selected/{method}_b{budget}.json (escolhidos no val por src/sweep.py).
Sem selected, só roda com --allow_defaults (valores NÃO validados, marcados no metrics.json).

  python -m src.part0 --data dados --method both --budget 5 --seed 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import common as C
from .models import build_model, count_params


def run(cfg, data, budget, seed, out_dir=None, evaluate_test=False, device=None, verbose=True, method="part0_ft"):
    """Um treino. Com evaluate_test=False o test_public nem é carregado (modo sweep/seleção)."""
    assert evaluate_test is False or data.test is not None, "evaluate_test=True exige load_data(need_test=True)"
    assert evaluate_test or data.test is None, "modo seleção: não carregue o test_public"
    dev = C.get_device(device)
    C.seed_all(seed)
    m = cfg["model"]
    model = build_model(m["arch"], data.K, m.get("pretrained", False), m.get("widths"), m.get("dropout", 0.3))
    timer = C.Timer()
    hist, best_ep = C.fit_supervised(model, data, cfg, dev, method, verbose)
    full_ep, minvl_ep = C.memorization_epochs(hist)
    va = C.evaluate(model, C.make_loader(data.val, data.val_y, C.EvalTF(), 64, False), dev)
    res = dict(method=method, budget=budget, seed=seed, cfg_hash=C.cfg_hash(cfg), cfg_source=cfg.get("_source"),
               params=count_params(model), best_epoch=best_ep, epoch_train_acc_100=full_ep,
               epoch_min_val_loss=minvl_ep, val_f1=va["f1"], val_acc=va["acc"], val_loss=va["loss"],
               minutes=timer.minutes())
    if evaluate_test:
        te = C.evaluate(model, C.make_loader(data.test, data.test_y, C.EvalTF(), 64, False), dev)  # 1 única vez
        res.update(test_f1=te["f1"], test_acc=te["acc"])
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        hist.to_csv(out / "hist.csv", index=False)
        C.save_json(cfg, out / "config.json")
        title = f"{method} ({budget}/classe, seed {seed})"
        C.plot_curves(hist, best_ep, title, out / "curva.png")
        if evaluate_test:
            C.plot_confusion(te["ys"], te["preds"], data.names, f"{title}: test_public (F1 macro {te['f1']:.3f})",
                             out / "confusao.png")
            C.per_class_table(te["ys"], te["preds"], data.names).to_csv(out / "f1_por_classe.csv", index=False)
            np.save(out / "test_probs.npy", te["probs"])
        C.save_run_checkpoint(out / "weights.pt", model, cfg, data, meta=dict(res))
        C.save_json(res, out / "metrics.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--method", choices=["part0_ft", "part0_scratch", "both"], default="both")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_defaults", action="store_true")
    ap.add_argument("--cache", default=None, help="pasta para cache de arrays (acelera reinícios no Colab)")
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b=v", help="override de config")
    a = ap.parse_args()
    cache = C.DataCache(a.data, a.cache)
    for method in (["part0_scratch", "part0_ft"] if a.method == "both" else [a.method]):
        cfg = C.method_cfg(method, a.budget, a.set, a.allow_defaults)
        print(f"\n=== {method} | config: {cfg['_source']}")
        data = cache.get(a.budget, cfg, need_test=True)
        res = run(cfg, data, a.budget, a.seed, C.run_dir(a.out, method, a.budget, a.seed), True, method=method)
        lab = C.project()["rotulo_no_csv"].get(method)
        if lab:
            C.append_curve_row(Path(a.out) / "curva_orcamento.csv", lab, a.budget, a.seed, res["test_f1"], res["test_acc"])
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})


if __name__ == "__main__":
    main()
