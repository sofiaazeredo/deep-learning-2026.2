"""
Parte 1.5 — o gráfico obrigatório do descolamento.

Dois painéis sobre as mesmas sequências:
  em cima,   mAP por quadro e IDF1;
  embaixo,   identidades previstas / verdadeiras e ID switches por identidade
             verdadeira.

Sequências ordenadas pelo eixo de dificuldade escolhido (densidade, movimento
de câmera ou duração de oclusão) — a escolha é do argumento --order-by.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import matplotlib.pyplot as plt

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/results/baseline_per_sequence.csv")
    parser.add_argument("--order-by", default="density",
                        choices=["density", "camera_motion", "occlusion"])
    parser.add_argument("--name", default="baseline_decoupling")
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
