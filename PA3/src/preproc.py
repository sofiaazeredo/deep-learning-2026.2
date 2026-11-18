"""Pré-processamento único: treino/avaliação e predict.py usam EXATAMENTE este código.

Só depende de PIL/numpy (torch só dentro de to_tensor), para o predict.py ficar leve.
"""
import numpy as np
from PIL import Image

MEAN = (0.485, 0.456, 0.406)   # estatísticas do ImageNet (os backbones são pré-treinados nele)
STD = (0.229, 0.224, 0.225)
MODES = ("squash", "pad")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_MEAN = np.array(MEAN, dtype=np.float32)
_STD = np.array(STD, dtype=np.float32)


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def resize_square(im, size, mode):
    """squash: deforma para size x size. pad: mantém a proporção e completa com a cor média do ImageNet
    (vira zero depois da normalização). Como as proporções variam entre câmeras, a escolha é um
    hiperparâmetro (data.resize_mode), decidido no val."""
    if mode == "squash":
        return im.resize((size, size), Image.BILINEAR)
    if mode == "pad":
        w, h = im.size
        r = size / max(w, h)
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        im = im.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), tuple(int(round(m * 255)) for m in MEAN))
        canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
        return canvas
    raise ValueError(f"resize_mode inválido: {mode} (use {MODES})")


def to_tensor(im):
    """PIL RGB ou array HWC uint8 -> tensor CHW float32 normalizado."""
    import torch
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def preprocess_path(path, size, mode):
    return to_tensor(resize_square(load_rgb(path), size, mode))
