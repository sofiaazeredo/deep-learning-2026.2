"""
Parte 3 — sumário da ablação, 3 seeds, média ± desvio.

Um eixo só, escolhido em --axis:
  cell    RNN simples vs. LSTM vs. GRU, mesmo orçamento de parâmetros,
          variando T em {4, 8, 16, 32}
  regime  teacher forcing -> scheduled sampling -> free-running, com e sem
          gradient clipping
  input   só geometria, só aparência, ou os dois
  context causal (online) vs. bidirecional (offline), reportando ganho de IDF1
          e latência juntos

Grava o CSV do sumário e a tabela que vai para a apresentação.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import pandas as pd

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", required=True,
                        choices=["cell", "regime", "input", "context"])
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
