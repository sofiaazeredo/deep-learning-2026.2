"""
Parte 2 — treino do modelo temporal.

A fonte de detecções está congelada daqui em diante; o que muda é o que
acontece entre os quadros. Treina em trajetórias do ground truth, em janelas de
T quadros (BPTT truncado).

--track a  : RNN como modelo de movimento (caixa prevista, perda smooth-L1 ou
             NLL gaussiana)
--track b  : RNN como memória de aparência (embedding agregado, perda triplet
             ou contrastiva)
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.dataset import TrackWindowDataset, create_splits
from src.model import build_model, CELLS
from src.losses import SmoothL1BoxLoss, GaussianNLLBoxLoss, TripletIdentityLoss
from src.training import REGIMES, truncated_bptt

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["a", "b"], default="a")
    parser.add_argument("--cell", choices=list(CELLS), default="gru")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--regime", choices=list(REGIMES),
                        default="scheduled_sampling")
    parser.add_argument("--clip-grad", type=float, default=1.0,
                        help="0 desliga o clipping (Eixo 2)")
    parser.add_argument("--bidirectional", action="store_true",
                        help="modo offline do Eixo 4")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
