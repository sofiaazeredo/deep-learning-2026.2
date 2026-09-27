"""
Parte 2 — perda contrastiva.

Roda com "python tests/test_losses.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.losses import ContrastiveIdentityLoss, SmoothL1BoxLoss


def test_contrastive_same_id_beats_different_id():
    loss_fn = ContrastiveIdentityLoss(temperature=0.1)
    good = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.05],
                                     [0.0, 1.0], [0.0, 0.95]]), dim=-1)
    bad = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0],
                                    [1.0, 0.0], [0.0, 1.0]]), dim=-1)
    labels = torch.tensor([0, 0, 1, 1])

    assert loss_fn(good, labels).item() < loss_fn(bad, labels).item()


def test_smooth_l1_closer_box_is_smaller():
    loss_fn = SmoothL1BoxLoss()
    target = torch.tensor([[0.5, 0.5, 0.1, 0.2]])
    close = torch.tensor([[0.51, 0.49, 0.11, 0.19]])
    far = torch.tensor([[0.1, 0.9, 0.4, 0.4]])
    assert loss_fn(close, target).item() < loss_fn(far, target).item()


def main():
    test_contrastive_same_id_beats_different_id()
    test_smooth_l1_closer_box_is_smaller()
    print("ok  test_contrastive_same_id_beats_different_id")
    print("ok  test_smooth_l1_closer_box_is_smaller")


if __name__ == "__main__":
    main()
