"""
Parte 1 — baseline por quadro.

IoU entre as detecções do quadro t e do t-1, matching guloso ou Hungarian,
limiar fixo, ID novo quando nada casa, track morta depois de k quadros sem
observação. Grava as trajetórias previstas para a avaliação.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.dataset import load_sequence, create_splits
from src.detection import public_detections, torchvision_detections
from src.tracker import Tracker

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="SDP",
                        choices=["DPM", "FRCNN", "SDP", "torchvision"])
    parser.add_argument("--matcher", default="hungarian",
                        choices=["greedy", "hungarian"])
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
