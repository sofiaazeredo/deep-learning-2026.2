"""
Parte 1.1 — a segunda fonte de detecções.

Roda o Faster R-CNN pré-treinado do torchvision (COCO, classe person) nos
quadros das 7 sequências e grava o resultado no formato det.txt em
data/MOT17/cache/torchvision/MOT17-XX/det.txt. O rastreamento só lê esse
arquivo: o detector não volta a rodar.

Guarda tudo a partir de score 0,05 (o piso do próprio modelo) para o AP ver o
ranking inteiro; o limiar de rastreamento sai do sweep no treino
(scripts/run_baseline.py --detector torchvision --sweep-score).
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import torch

from src.dataset import SCENES, load_sequence, resolve_root, torchvision_det_path
from src.detection import TORCHVISION_CACHE_SCORE, cache_torchvision_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--root", default="data/MOT17")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true",
                        help="refaz mesmo se o cache já existir")
    args = parser.parse_args()

    root = resolve_root(args.root)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    for scene in args.scenes:
        output = torchvision_det_path(scene, root)
        if output.exists() and not args.force:
            print(f"{scene}  já em cache: {output}")
            continue
        info, _, _ = load_sequence(scene, detector="SDP", root=root)
        start = time.perf_counter()
        dets = cache_torchvision_sequence(
            info["path"], output, score_threshold=TORCHVISION_CACHE_SCORE,
            device=device)
        elapsed = time.perf_counter() - start
        print(f"{scene}  {info['n_frames']} quadros  {len(dets)} detecções  "
              f"{elapsed:.0f} s  -> {output}", flush=True)


if __name__ == "__main__":
    main()
