"""python predict.py --imagens PASTA --saida preds.csv

Escreve CSV com colunas image_id,label (nomes de classes.csv, guardados no checkpoint). CPU, sem internet
(o checkpoint carrega com pretrained=False, nada é baixado). Imprime o número de parâmetros do modelo.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from src.infer import predict_paths  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.preproc import IMG_EXT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagens", required=True, help="pasta com as imagens")
    ap.add_argument("--saida", required=True, help="CSV de saída")
    ap.add_argument("--pesos", default=str(Path(__file__).resolve().parent / "pesos" / "modelo_final.pt"))
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--recursivo", action="store_true")
    ap.add_argument("--com_confianca", action="store_true", help="acrescenta coluna 'conf' (prob. máxima)")
    a = ap.parse_args()

    t0 = time.time()
    model, ck = load_checkpoint(a.pesos, "cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parametros: {n_params}")
    it = Path(a.imagens).rglob("*") if a.recursivo else Path(a.imagens).glob("*")
    files = sorted(p for p in it if p.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit(f"nenhuma imagem em {a.imagens}")
    pre = ck["preprocess"]
    probs = predict_paths(model, files, pre["size"], pre["resize_mode"], "cpu", a.bs)
    idx = probs.argmax(1)
    use_name = ck.get("meta", {}).get("id_format") == "filename"
    with open(a.saida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label"] + (["conf"] if a.com_confianca else []))
        for p, i, pr in zip(files, idx, probs):
            row = [p.name if use_name else p.stem, ck["class_names"][int(i)]]
            w.writerow(row + ([f"{pr[int(i)]:.4f}"] if a.com_confianca else []))
    dt = time.time() - t0
    print(f"{len(files)} imagens em {dt:.1f}s ({len(files) / dt:.1f} img/s, {torch.get_num_threads()} threads) -> {a.saida}")


if __name__ == "__main__":
    main()
