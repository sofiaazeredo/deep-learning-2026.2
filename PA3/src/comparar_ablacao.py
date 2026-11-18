"""Compara variantes do FixMatch (pastas --out do run_final) no orçamento 5: val e test F1 médio±desvio entre seeds
e diferença pareada por seed contra a primeira pasta. Só reporte: quem decide o modelo é o val.

  python -m src.comparar_ablacao resultados resultados_none_tau095 resultados_none_tau05
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(d, method="fixmatch", budget=5):
    rows = []
    for f in sorted(Path(d, "runs").glob(f"{method}_b{budget}_s*/metrics.json")):
        m = json.load(open(f))
        rows.append(dict(seed=m["seed"], val_f1=m["val_f1"], test_f1=m["test_f1"], test_acc=m["test_acc"],
                         top3_pl=m.get("top3_share_pl_test"), mask=m.get("mask_rate_last")))
    return pd.DataFrame(rows).set_index("seed") if rows else None


def main():
    dirs = sys.argv[1:] or ["resultados"]
    base = load(dirs[0])
    out = []
    for d in dirs:
        df = load(d)
        if df is None:
            print(f"[sem runs] {d}")
            continue
        cfg = json.load(open(next(Path(d, "runs").glob("fixmatch_b5_s*/config.json"))))["fix"]
        r = dict(pasta=d, corr=cfg["corr"], tau=cfg["tau"], n=len(df),
                 val_f1=f"{df.val_f1.mean():.3f}±{df.val_f1.std():.3f}",
                 test_f1=f"{df.test_f1.mean():.3f}±{df.test_f1.std():.3f}", test_acc=f"{df.test_acc.mean():.3f}",
                 mask=f"{df['mask'].mean():.3f}" if df["mask"].notna().any() else "")
        if base is not None and d != dirs[0]:
            j = df.join(base, rsuffix="_ref", how="inner")
            if len(j) > 1:
                dv, dt = j.val_f1 - j.val_f1_ref, j.test_f1 - j.test_f1_ref
                r["Δval vs 1ª"] = f"{dv.mean():+.3f}±{dv.std():.3f}"
                r["Δtest vs 1ª"] = f"{dt.mean():+.3f}±{dt.std():.3f}"
        out.append(r)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(pd.DataFrame(out).to_string(index=False))
    print("\nDiferenças menores que o desvio entre seeds (n=3) são ruído; a escolha do modelo é pelo val.")


if __name__ == "__main__":
    main()
