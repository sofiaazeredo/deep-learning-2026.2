"""
Parte 2 — AppearanceRNN.

Roda com "python tests/test_model.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model import AppearanceRNN, build_model


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


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
