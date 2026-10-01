"""
Parte 3, Eixo 3 — parametrização da caixa da cabeça de geometria.

A RNN prevê o DESLOCAMENTO da caixa, na codificação do Faster R-CNN
(centro relativo ao tamanho, log da razão de tamanhos, pesos 10/10/5/5), e
recebe a velocidade já nessa escala. Em coordenadas normalizadas pela imagem
o movimento de um quadro é ~0,001; codificado, fica ~0,1 a 1.

Roda com "python tests/test_boxes.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.boxes import decode_delta, encode_delta, motion_features


def test_encode_decode_round_trip():
    ref = torch.tensor([[0.50, 0.40, 0.05, 0.20], [0.10, 0.90, 0.02, 0.06]])
    target = torch.tensor([[0.51, 0.40, 0.06, 0.18], [0.10, 0.88, 0.02, 0.07]])

    delta = encode_delta(ref, target)

    assert torch.allclose(decode_delta(ref, delta), target, atol=1e-6)


def test_encoding_is_relative_to_box_size():
    # O mesmo deslocamento de meia largura dá o mesmo código numa caixa
    # pequena e numa grande: é o que torna a escala aprendível.
    small = torch.tensor([0.5, 0.5, 0.02, 0.04])
    large = torch.tensor([0.5, 0.5, 0.20, 0.40])

    d_small = encode_delta(small, small + torch.tensor([0.01, 0.0, 0.0, 0.0]))
    d_large = encode_delta(large, large + torch.tensor([0.10, 0.0, 0.0, 0.0]))

    assert torch.allclose(d_small, d_large, atol=1e-5)
    assert torch.isclose(d_small[0], torch.tensor(5.0))     # 10 x 0,5 largura


def test_motion_features_first_step_has_zero_velocity():
    cur = torch.tensor([0.5, 0.5, 0.1, 0.2])

    feat = motion_features(None, cur)

    assert feat.shape == (8,)
    assert torch.equal(feat[:4], cur)
    assert torch.equal(feat[4:], torch.zeros(4))


def test_motion_features_velocity_is_per_frame():
    # Entre observações separadas por 3 quadros, a velocidade é dividida
    # pelo buraco: o modelo sempre vê deslocamento por quadro.
    prev = torch.tensor([0.50, 0.5, 0.1, 0.2])
    cur = torch.tensor([0.53, 0.5, 0.1, 0.2])

    one = motion_features(prev, cur, gap=1)
    three = motion_features(prev, cur, gap=3)

    assert torch.allclose(three[4:], one[4:] / 3)


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
