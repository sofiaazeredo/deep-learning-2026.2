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


def test_query_head_is_l2_and_stateless():
    # A consulta de uma detecção sai de uma cabeça própria, sem estado:
    # não é "um passo da GRU a partir do zero".
    model = AppearanceRNN(cell="gru", embed_dim=8, hidden=16)
    crops = torch.randn(3, 8)

    first = model.query(crops)
    again = model.query(crops)

    assert first.shape == (3, 8)
    assert torch.allclose(first.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(first, again)


def test_fusion_query_reads_only_the_embedding():
    model = FusionRNN(cell="gru", embed_dim=8, hidden=16)

    assert model.query(torch.randn(2, 8)).shape == (2, 8)


def test_legacy_checkpoint_without_query_head_still_loads(tmp_path):
    # Os checkpoints velhos não têm a cabeça de consulta: carregam, marcam
    # has_query=False e o tracker volta para a consulta antiga.
    model = AppearanceRNN(cell="gru", embed_dim=8, hidden=16)
    state = {k: v for k, v in model.state_dict().items()
             if not k.startswith("query_head")}
    torch.save({"name": "appearance",
                "kwargs": {"cell": "gru", "embed_dim": 8, "hidden": 16},
                "state_dict": state}, tmp_path / "old.pt")

    loaded, _ = load_checkpoint(tmp_path / "old.pt")

    assert loaded.has_query is False
    assert model.has_query is True


def test_build_model_appearance():
    model = build_model("appearance", cell="rnn", embed_dim=4, hidden=8)
    assert isinstance(model, AppearanceRNN)
    assert model.cell == "rnn"


def test_motion_step_shape():
    # Entrada: caixa atual + velocidade codificada (8); saída: caixa absoluta.
    model = MotionRNN(cell="gru", hidden=16)
    state = model.init_state(2, "cpu")
    pred, state = model.step(torch.rand(2, 8), state)
    assert model.feature_dim == 8
    assert pred.shape == (2, 4)
    assert state.shape[1] == 2


def test_motion_delta_head_starts_near_the_current_box():
    # Com a saída zerada o deslocamento é zero: prever "a caixa atual" é o
    # ponto de partida, e o treino só precisa aprender o movimento.
    model = MotionRNN(cell="gru", hidden=16)
    torch.nn.init.zeros_(model.proj.weight)
    torch.nn.init.zeros_(model.proj.bias)
    features = torch.tensor([[0.5, 0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0]])

    pred, _ = model.step(features, model.init_state(1, "cpu"))

    assert torch.allclose(pred, features[:, :4], atol=1e-6)


def test_legacy_geometry_checkpoint_loads_as_absolute(tmp_path):
    # Checkpoint da geometria antiga (caixa absoluta, entrada de 4): carrega
    # como parameterization="absolute" para a comparação antes/depois.
    old = MotionRNN(cell="gru", hidden=16, parameterization="absolute")
    torch.save({"name": "motion", "kwargs": {"cell": "gru", "hidden": 16},
                "state_dict": old.state_dict()}, tmp_path / "old_geo.pt")

    loaded, _ = load_checkpoint(tmp_path / "old_geo.pt")

    assert loaded.parameterization == "absolute"
    assert loaded.feature_dim == 4


def test_fusion_checkpoint_roundtrip(tmp_path):
    model = FusionRNN(cell="gru", embed_dim=8, hidden=16)
    path = save_checkpoint(tmp_path / "fusion.pt", model,
                           {"cell": "gru", "embed_dim": 8, "hidden": 16})
    loaded, payload = load_checkpoint(path)
    assert isinstance(loaded, FusionRNN)
    assert payload["name"] == "fusion"
    features = torch.randn(model.feature_dim)
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
