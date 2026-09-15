"""
Parte 5 — teste de estresse, sem retreinar, em cima do modelo final.

--mode framerate  vídeo subamostrado a 1/2 e 1/5 da taxa original, curva de
                  degradação do IDF1. Por que um modelo de movimento aprendido
                  em dt fixo quebra quando dt muda?
--mode detector   detecções degradadas de propósito em 3 intensidades
                  (descarte, ruído nas caixas, falsos positivos). O modelo
                  temporal absorve ou amplifica a falha? mAP e IDF1 juntos.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.detector_sim import degrade, INTENSITIES

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["framerate", "detector"],
                        required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
