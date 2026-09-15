"""
Parte 4 — a correção.

Pega um dos diagnósticos da galeria, implementa a mudança que ele sugere e
mostra o antes/depois nas mesmas sequências. Se não funcionar, o resultado é
reportado como está e a discussão vai para o que isso revela sobre o
diagnóstico estar errado (foi assim no PA1).
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.tracker import Tracker

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--name", default="correcao")
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
