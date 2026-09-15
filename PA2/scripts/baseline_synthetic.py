"""
Parte 0.4 e o ensaio da Parte 1.

Roda a associação ingênua no piso fácil (poucas elipses, lentas, sem oclusão),
onde o IDF1 tem que ficar muito perto de 1, e depois gira os botões do gerador
(mais objetos, mais rápidos, oclusão mais longa) para mostrar onde o baseline
quebra. O gráfico dessa varredura é o ensaio da Parte 1.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.synthetic import make_sequence
from src.detector_sim import degrade
from src.tracker import Tracker
from src.metrics import evaluate_sequence

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["n_objects", "speed", "occlusion"],
                        default="occlusion")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--name", default="synthetic_sweep")
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
