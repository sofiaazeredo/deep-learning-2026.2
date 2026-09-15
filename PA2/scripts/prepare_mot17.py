"""
Prepara o MOT17: confere o layout de data/MOT17, lista as sequências, imprime
o split por sequência e a estatística que justifica o critério (densidade,
câmera parada vs. móvel, duração de oclusão).

Roda só com o pacote de anotações (~10 MB) — não precisa dos quadros.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.dataset import load_sequence, create_splits, split_report

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/MOT17")
    parser.add_argument("--detector", default="SDP")
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
