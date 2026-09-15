"""
Avaliação de trajetórias (Partes 1, 2, 3 e 5).

IDF1, ID switches, fragmentações e erro de contagem de identidades únicas, com
as métricas de src/metrics.py. MOTA é opcional e sai em coluna separada.

Grava um CSV por execução em experiments/results/.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.dataset import load_sequence
from src.metrics import evaluate_sequence

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True,
                        help="trajetórias previstas gravadas por run_baseline/run_tracker")
    parser.add_argument("--split", default="test")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
