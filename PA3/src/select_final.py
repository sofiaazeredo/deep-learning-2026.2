"""Escolhe o modelo entregue entre as seeds do orçamento padrão, PELO VAL (nunca pelo test_public),
e exporta o checkpoint autocontido para pesos/modelo_final.pt (o que o predict.py carrega).

  python -m src.select_final --out resultados --method fixmatch --dest pesos/modelo_final.pt
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch

from . import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--method", default="fixmatch", help="método da trilha (part0_ft para entregar o piso)")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--data", default=None, help="se dado, grava no checkpoint o formato dos ids dos CSVs")
    ap.add_argument("--dest", default="pesos/modelo_final.pt")
    a = ap.parse_args()
    P = C.project()
    b = a.budget or P["orcamento_padrao"]
    rows = []
    for d in sorted((Path(a.out) / "runs").glob(f"{a.method}_b{b}_s*")):
        if (d / "metrics.json").exists() and (d / "weights.pt").exists():
            m = C.load_json(d / "metrics.json")
            rows.append(dict(run=d.name, val_f1=m["val_f1"], val_loss=m["val_loss"], params=m["params"]))
    if not rows:
        raise SystemExit(f"nenhum run {a.method}_b{b}_s* com weights.pt em {a.out}/runs")
    df = pd.DataFrame(rows).sort_values(["val_f1", "val_loss"], ascending=[False, True])
    print(df.to_string(index=False))
    best = df.iloc[0]
    ck = torch.load(Path(a.out) / "runs" / best.run / "weights.pt", map_location="cpu", weights_only=False)
    ck["meta"]["selecionado_por"] = "val F1 macro entre seeds (não usa test_public)"
    ck["meta"]["run"] = best.run
    if a.data:
        schema = C.load_schema()
        ck["meta"]["id_format"] = "filename" if schema.get("id_has_extension") else "stem"
    Path(a.dest).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.dest)
    C.save_json(dict(run=best.run, **{k: ck["meta"].get(k) for k in ("val_f1", "params", "cfg_hash", "cfg_source")}),
                Path(a.dest).with_suffix(".json"))
    print(f"\nExportado {best.run} -> {a.dest} ({ck['num_params']:,} parâmetros)")


if __name__ == "__main__":
    main()
