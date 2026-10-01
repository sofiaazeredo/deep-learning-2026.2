"""
Caixa MOT (x, y, w, h) <-> (cx, cy, w, h) normalizado pela imagem, e a
codificação de deslocamento da cabeça de geometria (Eixo 3).

Em coordenadas normalizadas o movimento de um quadro é ~0,001 — pequeno
demais para a RNN aprender com smooth-L1. A codificação do Faster R-CNN
(torchvision BoxCoder, pesos 10/10/5/5) mede o deslocamento do centro em
larguras/alturas da caixa e o tamanho em log da razão: um quadro de
movimento vira ~0,1 a 1, na mesma escala para caixa pequena e grande.
"""

import numpy as np
import torch


def xywh_to_norm(box, width, height):
    x, y, w, h = [float(v) for v in box]
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    return np.array([(x + 0.5 * w) / width, (y + 0.5 * h) / height,
                     w / width, h / height], dtype=np.float32)


def norm_to_xywh(box, width, height):
    cx, cy, w, h = [float(v) for v in box]
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    pw, ph = w * width, h * height
    return np.array([cx * width - 0.5 * pw, cy * height - 0.5 * ph,
                     pw, ph], dtype=np.float64)


def as_norm_tensor(box, width, height, device="cpu"):
    return torch.as_tensor(xywh_to_norm(box, width, height), device=device)


BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)


def encode_delta(ref, target, weights=BOX_WEIGHTS):
    """
    Deslocamento de `ref` para `target`, ambos (..., 4) em (cx, cy, w, h)
    normalizado.
    """

    ref = torch.as_tensor(ref, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32, device=ref.device)
    wx, wy, ww, wh = weights
    w = ref[..., 2].clamp(min=1e-6)
    h = ref[..., 3].clamp(min=1e-6)
    return torch.stack([
        wx * (target[..., 0] - ref[..., 0]) / w,
        wy * (target[..., 1] - ref[..., 1]) / h,
        ww * torch.log(target[..., 2].clamp(min=1e-6) / w),
        wh * torch.log(target[..., 3].clamp(min=1e-6) / h),
    ], dim=-1)


def decode_delta(ref, delta, weights=BOX_WEIGHTS):
    """
    Inverso de encode_delta: caixa (..., 4) a partir de `ref` e do código.
    """

    ref = torch.as_tensor(ref, dtype=torch.float32)
    delta = torch.as_tensor(delta, dtype=torch.float32, device=ref.device)
    wx, wy, ww, wh = weights
    w = ref[..., 2].clamp(min=1e-6)
    h = ref[..., 3].clamp(min=1e-6)
    # Mesmo teto do BoxCoder: exp de um código absurdo não explode a caixa.
    limit = float(np.log(1000.0 / 16))
    return torch.stack([
        ref[..., 0] + delta[..., 0] / wx * w,
        ref[..., 1] + delta[..., 1] / wy * h,
        w * torch.exp((delta[..., 2] / ww).clamp(max=limit)),
        h * torch.exp((delta[..., 3] / wh).clamp(max=limit)),
    ], dim=-1)


def motion_features(prev, cur, gap=1):
    """
    Entrada da geometria: caixa atual (4) + velocidade por quadro desde a
    observação anterior, codificada (4). Sem observação anterior, a
    velocidade é zero. `gap` é o número de quadros entre as duas.
    """

    cur = torch.as_tensor(cur, dtype=torch.float32)
    if prev is None:
        velocity = torch.zeros_like(cur)
    else:
        velocity = encode_delta(prev, cur) / max(float(gap), 1.0)
    return torch.cat([cur, velocity], dim=-1)
