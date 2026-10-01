"""
Parte 2 — BPTT truncado da Trilha B.

Roda com "python tests/test_training.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.losses import (
    ContrastiveIdentityLoss,
    PredictiveContrastiveLoss,
    SmoothL1BoxLoss,
)
from src.boxes import motion_features
from src.model import AppearanceRNN, MotionRNN
from src.training import gradient_norm_profile, truncated_bptt


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


def _geometry_track(start, velocity, frames):
    """
    Trajetória de velocidade constante (cx, cy) no formato da janela de
    geometria: (frame, caixa atual + velocidade codificada).
    """

    box = torch.tensor(start, dtype=torch.float32)
    step = torch.tensor([velocity[0], velocity[1], 0.0, 0.0])
    steps, prev, prev_frame = [], None, None
    for frame in frames:
        cur = box + step * (frame - 1)
        gap = 1 if prev_frame is None else frame - prev_frame
        steps.append((frame, motion_features(prev, cur, gap=gap)))
        prev, prev_frame = cur, frame
    return steps


def test_truncated_bptt_box_loss_drops():
    torch.manual_seed(0)
    model = MotionRNN(cell="gru", hidden=16)
    loss_fn = SmoothL1BoxLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    window = {
        1: _geometry_track([0.1, 0.2, 0.05, 0.1], [0.05, 0.0], range(1, 4)),
        2: _geometry_track([0.8, 0.7, 0.05, 0.1], [-0.05, 0.0], range(1, 4)),
    }
    first = float(truncated_bptt(model, window, box_loss_fn=loss_fn,
                                 optimizer=optimizer).detach())
    last = first
    for _ in range(20):
        last = float(truncated_bptt(model, window, box_loss_fn=loss_fn,
                                    optimizer=optimizer).detach())
    assert last < first


def test_gradient_norm_profile_has_a_length():
    torch.manual_seed(0)
    model = AppearanceRNN(cell="gru", embed_dim=8, hidden=16)
    base = {
        1: torch.nn.functional.normalize(torch.randn(8), dim=0),
        2: torch.nn.functional.normalize(torch.randn(8), dim=0),
    }
    window = {}
    for identity, vector in base.items():
        window[identity] = [
            (frame, torch.nn.functional.normalize(vector + 0.3 * torch.randn(8),
                                                  dim=0))
            for frame in (1, 2, 3, 4)
        ]
    profile = gradient_norm_profile(model, window)
    assert len(profile) >= 2
    assert all(value >= 0 for value in profile)


def _drifting_window(seed=0, n_frames=6, dim=8):
    torch.manual_seed(seed)
    window = {}
    for identity in (1, 2, 3):
        vector = torch.nn.functional.normalize(torch.randn(dim), dim=0)
        steps = []
        for frame in range(1, n_frames + 1):
            vector = torch.nn.functional.normalize(vector + 0.3 * torch.randn(dim), dim=0)
            steps.append((frame, vector))
        window[identity] = steps
    return window


def test_truncated_bptt_predictive_loss_drops():
    torch.manual_seed(0)
    model = AppearanceRNN(cell="gru", embed_dim=8, hidden=16)
    loss_fn = PredictiveContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
    window = _drifting_window()

    first = float(truncated_bptt(model, window, loss_fn, optimizer=optimizer).detach())
    last = first
    for _ in range(30):
        last = float(truncated_bptt(model, window, loss_fn, optimizer=optimizer).detach())

    assert first > 0
    assert last < first


def test_predictive_loss_reaches_the_query_head():
    # Com a perda preditiva a cabeça de consulta também aprende.
    torch.manual_seed(0)
    model = AppearanceRNN(cell="gru", embed_dim=8, hidden=16)
    loss = truncated_bptt(model, _drifting_window(), PredictiveContrastiveLoss())
    loss.backward()

    assert model.query_head.weight.grad is not None
    assert float(model.query_head.weight.grad.abs().sum()) > 0


def test_gradient_profile_with_predictive_loss():
    torch.manual_seed(0)
    model = AppearanceRNN(cell="gru", embed_dim=8, hidden=16)

    curve = gradient_norm_profile(model, _drifting_window(),
                                  loss_fn=PredictiveContrastiveLoss())

    assert len(curve) >= 2
    assert all(value >= 0 for value in curve)


def test_geometry_learns_to_beat_copying_the_last_box():
    # Velocidade constante é o caso mais simples: depois de treinar, o
    # passo previsto tem que errar menos que "a caixa não se mexe". A
    # geometria antiga (caixa absoluta) perdia exatamente para isso.
    torch.manual_seed(0)
    model = MotionRNN(cell="gru", hidden=32)
    loss_fn = SmoothL1BoxLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    velocities = [(0.004, 0.0), (-0.003, 0.001), (0.002, -0.002), (0.0, 0.003)]
    window = {i: _geometry_track([0.3 + 0.1 * i, 0.5, 0.05, 0.12], v, range(1, 13))
              for i, v in enumerate(velocities, start=1)}
    for _ in range(300):
        truncated_bptt(model, window, box_loss_fn=loss_fn, optimizer=optimizer)

    test = _geometry_track([0.5, 0.4, 0.05, 0.12], (0.003, -0.001), range(1, 13))
    model.eval()
    state = model.init_state(1, "cpu")
    err_model, err_copy = [], []
    with torch.no_grad():
        for (frame, feat), (_, nxt) in zip(test[:-1], test[1:]):
            pred, state = model.step(feat, state)
            err_model.append(float((pred[0] - nxt[:4]).abs().mean()))
            err_copy.append(float((feat[:4] - nxt[:4]).abs().mean()))

    assert sum(err_model[2:]) < 0.5 * sum(err_copy[2:])


def test_box_loss_only_between_consecutive_frames():
    # Id visto nos quadros 1, 2 e 5: só o par (1, 2) entra na perda — o
    # modelo prevê o quadro seguinte, não o de três quadros depois.
    torch.manual_seed(0)
    model = MotionRNN(cell="gru", hidden=16)
    loss_fn = SmoothL1BoxLoss()
    with_gap = {1: _geometry_track([0.2, 0.3, 0.05, 0.1], (0.01, 0.0), [1, 2, 5])}
    consecutive = {1: with_gap[1][:2]}

    loss_gap = truncated_bptt(model, with_gap, box_loss_fn=loss_fn)
    loss_pair = truncated_bptt(model, consecutive, box_loss_fn=loss_fn)

    assert torch.isclose(loss_gap, loss_pair)


def main():
    test_truncated_bptt_reduces_loss_on_toy_window()
    print("ok  test_truncated_bptt_reduces_loss_on_toy_window")


if __name__ == "__main__":
    main()
