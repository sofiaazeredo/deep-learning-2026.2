"""
Baseline de Kalman de velocidade constante.

Permitido apenas como comparação — não é o modelo temporal da Parte 2. Serve
para saber se a recorrência ganha de um baseline honesto e frequentemente
difícil de bater.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.tracker import Tracker, KalmanMotion

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--name", default="kalman")
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
