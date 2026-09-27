"""
Parte 2 — recorte e encoder congelado.

Roda com "python tests/test_appearance.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.appearance import CropEncoder, crop


def test_crop_output_size_and_clamp():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[5:15, 5:15] = 200
    patch = crop(frame, (-4, -4, 12, 12), output_size=(16, 8))

    assert patch.shape == (16, 8, 3)


def test_encoder_l2_and_frozen():
    encoder = CropEncoder(embed_dim=32, freeze=True)
    crops = [np.zeros((128, 64, 3), dtype=np.uint8),
             np.full((128, 64, 3), 255, dtype=np.uint8)]

    embeddings = encoder.embed(crops, device="cpu").detach()

    assert embeddings.shape == (2, 32)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert all(not parameter.requires_grad
               for parameter in encoder.backbone.parameters())


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
