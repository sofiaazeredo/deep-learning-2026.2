"""
Parte 2 — AppearanceRNN.

Roda com "python tests/test_model.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model import (
    AppearanceRNN,
    FusionRNN,
    MotionRNN,
    build_model,
    load_checkpoint,
    save_checkpoint,
)


def test_step_shape_and_state_persist():
    model = AppearanceRNN(cell="gru", embed_dim=8, hidden=16)
    state = model.init_state(1, "cpu")
    first, state = model.step(torch.randn(8), state)
    second, _ = model.step(torch.randn(8), state)

    assert first.shape == (1, 8)
    assert torch.allclose(first.norm(dim=-1), torch.ones(1), atol=1e-5)
    assert not torch.allclose(first, second)


def test_build_model_appearance():
    model = build_model("appearance", cell="rnn", embed_dim=4, hidden=8)
    assert isinstance(model, AppearanceRNN)
    assert model.cell == "rnn"


def test_motion_step_shape():
    model = MotionRNN(cell="gru", hidden=16)
    state = model.init_state(2, "cpu")
    pred, state = model.step(torch.rand(2, 4), state)
    assert pred.shape == (2, 4)
    assert state.shape[1] == 2


def test_fusion_checkpoint_roundtrip(tmp_path):
    model = FusionRNN(cell="gru", embed_dim=8, hidden=16)
    path = save_checkpoint(tmp_path / "fusion.pt", model,
                           {"cell": "gru", "embed_dim": 8, "hidden": 16})
    loaded, payload = load_checkpoint(path)
    assert isinstance(loaded, FusionRNN)
    assert payload["name"] == "fusion"
    features = torch.randn(8 + 4)
    first, _ = model.step(features, model.init_state(1, "cpu"))
    second, _ = loaded.step(features, loaded.init_state(1, "cpu"))
    assert torch.allclose(first[0], second[0])
    assert torch.allclose(first[1], second[1])


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
