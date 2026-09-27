"""
Caixa MOT (x, y, w, h) <-> (cx, cy, w, h) normalizado pela imagem.
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
