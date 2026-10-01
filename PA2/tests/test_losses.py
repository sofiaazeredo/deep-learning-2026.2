"""
Parte 2 — perda contrastiva.

Roda com "python tests/test_losses.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.losses import (
    ContrastiveIdentityLoss,
    PredictiveContrastiveLoss,
    SmoothL1BoxLoss,
)


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


def _drifting_identities(seed=0, n_frames=8, dim=16, drift=0.35):
    """
    Duas identidades cuja aparência deriva ao longo da janela (luz, pose):
    o recorte do quadro t é parecido com o de t+1, pouco com o do começo.
    """

    torch.manual_seed(seed)
    crops = {}
    for identity in (1, 2):
        vector = F.normalize(torch.randn(dim), dim=0)
        steps = []
        for frame in range(1, n_frames + 1):
            vector = F.normalize(vector + drift * torch.randn(dim), dim=0)
            steps.append((frame, vector))
        crops[identity] = steps
    return crops


def _flatten(crops):
    labels, frames, vectors = [], [], []
    for identity, steps in crops.items():
        for frame, vector in steps:
            labels.append(identity)
            frames.append(frame)
            vectors.append(vector)
    return torch.tensor(labels), torch.tensor(frames), torch.stack(vectors)


def test_predictive_loss_rewards_updating_the_memory():
    # A memória que só guarda o primeiro recorte (o "latch" da auditoria)
    # tem que perder para a que acompanha o último recorte visto: a perda
    # compara a memória com os recortes FUTUROS da mesma identidade.
    crops = _drifting_identities()
    labels, frames, queries = _flatten(crops)
    loss_fn = PredictiveContrastiveLoss(temperature=0.1)

    latched = torch.stack([crops[int(i)][0][1] for i in labels])
    updated = queries.clone()

    loss_latched = loss_fn(latched, labels, frames, queries, labels, frames)
    loss_updated = loss_fn(updated, labels, frames, queries, labels, frames)

    assert loss_updated < loss_latched


def test_predictive_loss_ignores_present_and_past_of_the_same_id():
    # Memória igual ao recorte do próprio quadro não ganha nada: o recorte
    # do mesmo quadro (e dos anteriores) da mesma identidade não é positivo
    # nem negativo. Só o futuro conta como positivo.
    labels = torch.tensor([1, 1, 2])
    frames = torch.tensor([1, 2, 1])
    queries = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), dim=1)
    loss_fn = PredictiveContrastiveLoss(temperature=0.1)

    # Só a memória do id 1 no quadro 1 tem futuro (o quadro 2).
    points_to_now = queries[[0]]
    points_to_future = queries[[1]]
    mem_labels, mem_frames = torch.tensor([1]), torch.tensor([1])

    now = loss_fn(points_to_now, mem_labels, mem_frames, queries, labels, frames)
    future = loss_fn(points_to_future, mem_labels, mem_frames, queries, labels, frames)

    assert future < now


def test_predictive_loss_without_future_is_zero():
    labels, frames = torch.tensor([1, 2]), torch.tensor([3, 3])
    queries = F.normalize(torch.randn(2, 4), dim=1)

    loss = PredictiveContrastiveLoss()(queries, labels, frames, queries, labels, frames)

    assert float(loss) == 0.0


def main():
    test_contrastive_same_id_beats_different_id()
    test_smooth_l1_closer_box_is_smaller()
    print("ok  test_contrastive_same_id_beats_different_id")
    print("ok  test_smooth_l1_closer_box_is_smaller")


if __name__ == "__main__":
    main()
