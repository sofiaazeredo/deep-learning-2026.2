"""
Parte 4 — galeria de falhas.

Três trechos em que o modelo final erra feio. Cada um com a tira de quadros
(ground truth e predição coloridos por identidade) mais o mapa intermediário
relevante — a caixa prevista pela recorrência ou a matriz de similaridade dos
embeddings — e o diagnóstico escrito em experiments/results/.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import matplotlib.pyplot as plt

from src.model import load_checkpoint

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-failures", type=int, default=3)
    parser.add_argument("--name", default="failures")
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
