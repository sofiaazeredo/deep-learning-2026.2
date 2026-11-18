"""Smoke tests do pipeline com os dados (reais ou sintéticos) + MEDIÇÃO de tempo para planejar os sweeps.

  python -m tests.check_pipeline --data dados
  python -m tests.check_pipeline --data /content/fake --fake --set model.pretrained=false data.size=64 workers=0
Usa configs/defaults (--allow_defaults implícito): serve para testar o CÓDIGO, não para escolher valores.
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import common as C  # noqa: E402
from src import fixmatch as FM  # noqa: E402
from src import part0 as P0  # noqa: E402
from src.models import ARCHS, build_model, count_params  # noqa: E402
from tests._report import Report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--overfit_epochs", type=int, default=30)
    ap.add_argument("--fm_steps", type=int, default=30, help="passos do smoke e da medição de tempo")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b=v")
    a = ap.parse_args()
    R = Report("check_pipeline")
    P = C.project()
    dev = C.get_device()
    R.info(f"torch {torch.__version__} | device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    R.check(dev == "cuda" or a.fake, "GPU disponível", "sem GPU os sweeps ficam inviáveis; no Colab: Ambiente de execução > GPU",
            warn_only=True)

    cfg = C.method_cfg("part0_ft", a.budget, a.set, allow_defaults=True)
    K = P["n_classes"]
    # 1) limite de parâmetros das arquiteturas candidatas
    for arch in ARCHS[:3]:
        n = count_params(build_model(arch, K, False))
        R.check(n <= P["limite_parametros"], f"parâmetros {arch}", f"{n:,} (limite {P['limite_parametros']:,})")
    # 2) pesos ImageNet baixam?
    if cfg["model"].get("pretrained"):
        try:
            build_model(cfg["model"]["arch"], K, True)
            R.ok(f"pesos ImageNet de {cfg['model']['arch']} (precisa de internet só aqui)")
        except Exception as e:  # noqa: BLE001
            R.fail("pesos ImageNet", f"{type(e).__name__}: {e}")
    # 3) dados
    data = C.load_data(a.data, a.budget, cfg["data"]["size"], cfg["data"]["resize_mode"], cfg["data"]["preload_factor"],
                       need_unl=True, need_test=False, cache_dir=a.cache)
    R.check(len(data.lab_y) == a.budget * data.K, "rotulados = orçamento × classes", f"{len(data.lab_y)}")
    R.check(data.lab_train.shape[1:] == (data.s, data.s, 3) and data.lab_eval.shape[1:] == (data.size, data.size, 3),
            "arrays pré-redimensionados", f"train {data.lab_train.shape}, eval {data.lab_eval.shape}")
    R.check(data.unl.shape[0] == (len(data.unl_ids)) and (a.fake or data.unl.shape[0] == P["tamanhos_esperados"]["train"]),
            "não rotulados = train inteiro", f"{data.unl.shape[0]}")
    R.check(data.test is None, "test_public NÃO carregado no modo seleção")
    R.check(set(np.unique(data.lab_y)) == set(range(data.K)), "todas as classes nos rótulos do orçamento")
    R.info(f"memória: unl {data.unl.nbytes / 1e9:.2f} GB | lab {data.lab_train.nbytes / 1e6:.0f} MB")

    # 4) overfit SEM aumentação: com 105 imagens o treino tem que chegar a 100% (se não, há bug no loop/rótulos).
    #    Com a aumentação da config isso pode levar bem mais épocas; essa é a pergunta da Parte 0, não um bug.
    c = copy.deepcopy(cfg)
    c["aug"] = dict(scale_min=1.0, hflip=0.0, jitter=None, gray_p=0.0)
    c["train"]["epochs"] = a.overfit_epochs
    C.seed_all(0)
    m = c["model"]
    model = build_model(m["arch"], data.K, m.get("pretrained", False), m.get("widths"), m.get("dropout", 0.3))
    hist, best_ep = C.fit_supervised(model, data, c, dev, "overfit", verbose=False)
    full, minvl = C.memorization_epochs(hist)
    R.check(hist.train_acc.max() >= 1.0, f"overfit sem aug: treino chega a 100% em {a.overfit_epochs} épocas",
            f"máx {hist.train_acc.max():.2f}; 1ª época com 100%: {full}; mín. val loss na época {minvl}")
    R.check(hist.val_acc.max() > 1.0 / data.K, "val acima do acaso", f"máx val acc {hist.val_acc.max():.3f} (acaso {1 / data.K:.3f})",
            warn_only=True)

    # 5) seed => reprodutível; seeds diferentes => resultados diferentes
    c2 = copy.deepcopy(cfg)
    c2["train"]["epochs"] = 2
    r0 = P0.run(c2, data, a.budget, 0, None, False, verbose=False)
    r0b = P0.run(c2, data, a.budget, 0, None, False, verbose=False)
    r1 = P0.run(c2, data, a.budget, 1, None, False, verbose=False)
    R.check(abs(r0["val_loss"] - r0b["val_loss"]) < (1e-5 if dev == "cpu" else 5e-2), "mesma seed => mesma val loss",
            f"{r0['val_loss']:.6f} vs {r0b['val_loss']:.6f}", warn_only=(dev != "cpu"))
    R.check(abs(r0["val_loss"] - r1["val_loss"]) > 1e-8, "seeds diferentes => val loss diferente")

    # 6) guarda contra uso do test_public na seleção
    try:
        d_t = C.load_data(a.data, a.budget, cfg["data"]["size"], cfg["data"]["resize_mode"], 1.15, need_test=True,
                          cache_dir=a.cache)
        P0.run(c2, d_t, a.budget, 0, None, False, verbose=False)
        R.fail("guarda do test_public", "run(evaluate_test=False) aceitou dados com test carregado")
    except AssertionError:
        R.ok("guarda do test_public: modo seleção recusa dados com test carregado")
    del d_t

    # 7) vistas do FixMatch
    ud = FM.UnlDS(data.unl, data.size, C.method_cfg("fixmatch", a.budget, a.set, allow_defaults=True)["aug"])
    wv, sv, _ = ud[0]
    R.check(wv.shape == sv.shape == (3, data.size, data.size) and not torch.equal(wv, sv), "vistas fraca/forte",
            f"{tuple(wv.shape)}, diferentes")

    # 8) FixMatch: roda, é finito, mede tempo
    sps = {}
    for corr in ("none", "da", "flex", "both"):
        fc = C.method_cfg("fixmatch", a.budget, a.set + [f"fix.corr=\"{corr}\"", f"train.steps={a.fm_steps}",
                                                          f"train.eval_every={max(1, a.fm_steps // 2)}"], allow_defaults=True)
        fc["fix"]["tau"] = min(fc["fix"]["tau"], 0.5) if corr != "none" else fc["fix"]["tau"]
        try:
            r = FM.run(fc, data, a.budget, 0, None, False, verbose=False)
            ok = np.isfinite(r["val_loss"]) and 0 <= r["mask_rate_last"] <= 1
            R.check(ok, f"FixMatch corr={corr} roda", f"val F1 {r['val_f1']:.3f} | mask {r['mask_rate_last']:.2f} | "
                                                       f"{r['sec_per_step']:.2f} s/passo")
            sps[corr] = r["sec_per_step"]
        except Exception as e:  # noqa: BLE001
            R.fail(f"FixMatch corr={corr}", f"{type(e).__name__}: {e}")
    if sps:
        s = float(np.nanmean(list(sps.values())))
        R.info(f"TEMPO: ~{s:.2f} s/passo => 1000 passos ≈ {s * 1000 / 60:.1f} min; 4000 passos ≈ {s * 4000 / 60:.0f} min por run.")
        R.info("Use isso para fixar train.steps e n_trials do sweep do FixMatch (cada trial × seed é um run completo).")
    if dev == "cuda":
        R.info(f"pico de memória GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    R.finish()


if __name__ == "__main__":
    main()
