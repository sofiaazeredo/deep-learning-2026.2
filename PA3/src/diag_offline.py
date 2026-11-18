"""Diagnósticos SEM retreinar, a partir de resultados/runs/*/test_probs.npy (só reporte; nunca selecione nada aqui).

  python -m src.diag_offline --data /content/dados --runs resultados/runs/fixmatch_b5_s0 resultados/runs/part0_ft_b5_s0 --out resultados/diag_offline

Produz por run: calibração (ECE, diagrama de confiabilidade, confiança média x acurácia), precisão/cobertura do
pseudo-rotulador por τ com a confiança BRUTA (sem FlexMatch/DA) e taxa de erro em imagens cinza x coloridas
(com a proporção base). A ordem de test_probs.npy é a de read_split(test_public).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import common as C

TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def is_gray(path, thr=3.0):
    a = np.asarray(Image.open(path).convert("RGB").resize((64, 64)), dtype=np.float32)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < thr


def ece_bins(conf, ok, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            gap = abs(conf[m].mean() - ok[m].mean())
            ece += m.mean() * gap
            rows.append(dict(bin=f"{lo:.1f}-{hi:.1f}", n=int(m.sum()), conf=conf[m].mean(), acc=ok[m].mean()))
    return ece, pd.DataFrame(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="resultados/diag_offline")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sch = C.load_schema()
    names, id2idx = C.load_classes(a.data, sch)
    ids, paths, y = C.read_split(a.data, sch, "test_public", names, id2idx)
    y = np.asarray(y)
    gray = np.array([is_gray(p) for p in paths])
    print(f"test_public: {len(y)} imagens | cinza: {gray.mean():.3f}")
    fig, axs = plt.subplots(1, len(a.runs), figsize=(5 * len(a.runs), 4.6), squeeze=False)
    summ = []
    for ax, r in zip(axs[0], a.runs):
        r = Path(r)
        P = np.load(r / "test_probs.npy")
        assert len(P) == len(y), f"{r}: {len(P)} probs x {len(y)} rótulos"
        conf, pred = P.max(1), P.argmax(1)
        ok = pred == y
        ece, tb = ece_bins(conf, ok)
        tb.to_csv(out / f"{r.name}_confiabilidade.csv", index=False)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(tb.conf, tb.acc, "o-")
        ax.set(title=f"{r.name}\nECE {ece:.3f}", xlabel="confiança", ylabel="acurácia", xlim=(0, 1), ylim=(0, 1))
        rows = []
        for t in TAUS:
            m = conf >= t
            rows.append(dict(tau=t, cobertura=m.mean(), precisao=ok[m].mean() if m.any() else np.nan, n=int(m.sum())))
        pd.DataFrame(rows).to_csv(out / f"{r.name}_tau_bruto.csv", index=False)
        print(f"\n== {r.name}: acc {ok.mean():.3f} | conf média {conf.mean():.3f} | ECE {ece:.3f}")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        eg, ec = (~ok[gray]).mean(), (~ok[~gray]).mean()
        print(f"erro em cinza {eg:.3f} (n={gray.sum()}) x colorido {ec:.3f} (n={(~gray).sum()}) | "
              f"{(~ok).sum()} erros, {gray[~ok].mean():.3f} deles cinza (base {gray.mean():.3f})")
        summ.append(dict(run=r.name, acc=ok.mean(), conf_media=conf.mean(), ece=ece, erro_cinza=eg, erro_colorido=ec,
                         frac_cinza_base=gray.mean(), frac_cinza_nos_erros=gray[~ok].mean()))
    fig.tight_layout()
    fig.savefig(out / "confiabilidade.png", dpi=150)
    pd.DataFrame(summ).to_csv(out / "resumo.csv", index=False)
    print("\nsalvo em", out)


if __name__ == "__main__":
    main()
