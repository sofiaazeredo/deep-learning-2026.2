"""
Parte 4 — horizonte de memória efetivo (obrigatório), das duas formas.

1. Analítica: a norma de dL_t/dh_{t-k} em função de k, no nosso modelo e nos
   nossos dados. É a curva de gradiente que some dos slides. Se o Eixo 1 da
   Parte 3 foi feito, compara a RNN simples com o modelo com portas na mesma
   janela — os checkpoints já existem.
2. Empírica: quantos quadros o estado sobrevive a uma oclusão antes de a track
   morrer ou trocar de ID, comparado com a distribuição de duração de oclusão
   do dataset.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.training import gradient_norm_profile
from src.synthetic import occlusion_durations

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["analytic", "empirical", "both"],
                        default="both")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="mais de um compara células (Eixo 1)")
    parser.add_argument("--name", default="memory_horizon")
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
