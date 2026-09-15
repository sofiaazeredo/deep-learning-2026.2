"""
Parte 0.1 — gerador sintético.

Gera os vídeos de elipses com oclusão real (ordem de profundidade) e salva a
figura obrigatória: uma trajetória que some por N quadros e volta.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.synthetic import make_sequence, occlusion_durations

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-objects", type=int, default=8)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--occlusion-frames", type=int, default=10)
    parser.add_argument("--n-sequences", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="synthetic")
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
