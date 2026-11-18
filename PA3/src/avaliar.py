"""Avalia um preds.csv (image_id,label) contra um CSV com rótulos. É o 'comando que avalia' do README.

  python -m src.avaliar --preds preds.csv --gt dados/test_public.csv
Colunas do gt: as de configs/data_schema.json (ou --id_col/--label_col).
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--data", default="dados", help="pasta do pacote (para ler classes.csv)")
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--label_col", default=None)
    a = ap.parse_args()
    schema = C.load_schema()
    gt = Path(a.gt)
    if not gt.exists():                   # ex.: pacote real guarda os CSVs em labels/; procura pelo nome em --data
        hits = list(Path(a.data).rglob(gt.name))
        if not hits:
            raise SystemExit(f"{a.gt} não existe e '{gt.name}' não foi achado em {a.data}")
        gt = min(hits, key=lambda q: len(q.parts))
        print(f"gt: {gt}")
    sc = schema["split_cols"]
    ic, lc = a.id_col or sc["image"], a.label_col or sc["label"]
    p, g = pd.read_csv(a.preds), pd.read_csv(gt)
    if schema["label_kind"] == "id":      # o gt traz id numérico; o predict.py escreve nomes de classes.csv
        names, id2idx = C.load_classes(a.data, schema)
        g[lc] = g[lc].map(lambda v: names[id2idx[int(v)] if id2idx else int(v)])
    p["image_id"] = p["image_id"].astype(str)
    g[ic] = g[ic].astype(str)
    m = g.merge(p, left_on=ic, right_on="image_id", how="left", suffixes=("_gt", "_pred"))
    miss = m["label_pred" if "label_pred" in m else "label"].isna().sum()
    if miss:
        print(f"AVISO: {miss} imagens do gt sem predição (contam como erro)")
    gt_col = lc + "_gt" if lc + "_gt" in m else lc
    pr_col = "label_pred" if "label_pred" in m else "label"
    y, yh = m[gt_col].astype(str), m[pr_col].fillna("<sem predição>").astype(str)
    print(f"F1 macro {f1_score(y, yh, average='macro', labels=sorted(y.unique()), zero_division=0):.4f} | "
          f"acurácia {accuracy_score(y, yh):.4f} | n={len(m)}")


if __name__ == "__main__":
    main()
