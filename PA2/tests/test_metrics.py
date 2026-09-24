"""
Parte 0.3 — testes da métrica.

Três casos construídos à mão, com a resposta conhecida:
  (a) predição = ground truth          -> IDF1 = 1 e zero switches;
  (b) duas identidades trocadas no quadro k -> o número exato de switches;
  (c) uma track partida em duas no meio -> o efeito em IDF1, que NÃO é o
      mesmo de (b) — partir não troca identidade, fragmenta.

Roda com "python tests/test_metrics.py" ou sob pytest.
"""

import sys
from pathlib import Path

# Rodar "python tests/x.py" coloca tests/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.metrics import idf1, id_switches, fragmentations

RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def test_perfect_prediction():
    raise NotImplementedError


def test_swapped_identities():
    raise NotImplementedError


def test_broken_track():
    raise NotImplementedError


def main():
    test_perfect_prediction()
    test_swapped_identities()
    test_broken_track()
    print("ok")


if __name__ == "__main__":
    main()
