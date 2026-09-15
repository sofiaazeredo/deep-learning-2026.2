"""
Rastreamento com o modelo temporal da Parte 2, sobre a mesma fonte de
detecções congelada do baseline. Sai no mesmo formato de run_baseline.py, para
as métricas ficarem lado a lado.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.model import load_checkpoint
from src.tracker import Tracker

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="None usa o modelo final de experiments/results/best_model.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--window", type=int, default=None,
                        help="inferência em janelas; None processa a sequência inteira")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
