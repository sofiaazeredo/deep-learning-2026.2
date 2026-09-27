"""
Parte 2 — BPTT truncado da Trilha B.

Roda com "python tests/test_training.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.losses import ContrastiveIdentityLoss
from src.model import AppearanceRNN
from src.training import truncated_bptt


def test_truncated_bptt_reduces_loss_on_toy_window():
    torch.manual_seed(0)
    model = AppearanceRNN(cell="gru", embed_dim=8, hidden=16)
    loss_fn = ContrastiveIdentityLoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

    base = {
        1: torch.nn.functional.normalize(torch.randn(8), dim=0),
        2: torch.nn.functional.normalize(torch.randn(8), dim=0),
    }
    window = {}
    for identity, vector in base.items():
        window[identity] = [
            (frame, torch.nn.functional.normalize(vector + 0.4 * torch.randn(8),
                                                  dim=0))
            for frame in (1, 2, 3)
        ]

    first = float(truncated_bptt(model, window, loss_fn,
                                 optimizer=optimizer).detach())
    last = first
    for _ in range(12):
        last = float(truncated_bptt(model, window, loss_fn,
                                    optimizer=optimizer).detach())

    assert first > 0
    assert last < first


def main():
    test_truncated_bptt_reduces_loss_on_toy_window()
    print("ok  test_truncated_bptt_reduces_loss_on_toy_window")


if __name__ == "__main__":
    main()
