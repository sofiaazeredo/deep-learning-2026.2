"""
Parte 0 — dataset sintético (teste unitário do pipeline).

Imagens 128x128 com 5 a 20 elipses de tamanhos variados, muitas delas se
tocando, com ruído e contraste variáveis. As máscaras de instância saem
de graça, então dá para validar todo o pipeline — geração de alvo,
treino, decodificação e métrica — antes de tocar nos dados reais.

As elipses NÃO se sobrepõem: o pixel disputado fica com a instância
colocada primeiro. É de propósito, para reproduzir a propriedade do
DSB2018 (máscaras disjuntas) e produzir instâncias encostadas
compartilhando fronteira — exatamente o fenômeno que as Partes 1 e 2
atacam.

A interface é a mesma do DSB2018Dataset (dicionário com image,
instance_mask, boundary_target e image_path), então o mesmo código de
treino e de avaliação serve para os dois.
"""

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from src.dataset import create_boundary_target


def _rasterize_ellipse(yy, xx, cy, cx, a, b, theta):
    """
    Máscara booleana da elipse de centro (cy, cx), semieixos (a, b) e
    rotação theta.
    """

    ct, st = np.cos(theta), np.sin(theta)

    dy, dx = yy - cy, xx - cx

    u = (dx * ct + dy * st) / a
    v = (-dx * st + dy * ct) / b

    return (u * u + v * v) <= 1.0


def make_synthetic_sample(
    rng,
    size=128,
    n_min=5,
    n_max=20,
    axis_range=(5.0, 16.0),
    touch_prob=0.75,
    min_area_frac=0.45,
    min_area_px=25,
):
    """
    Retorna (image float32 HxW em [0,1], labels int32 HxW), 0 = fundo.

    Contraste, iluminação, borramento e ruído variam por imagem, para o
    teste não ficar fácil demais.
    """

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    labels = np.zeros((size, size), np.int32)
    fg_level = np.zeros((size, size), np.float32)

    bg_level = float(rng.uniform(0.05, 0.35))
    contrast = float(rng.uniform(0.15, 0.75))

    n_target = int(rng.integers(n_min, n_max + 1))

    placed = []
    next_id = 1
    attempts = 0

    while len(placed) < n_target and attempts < 40 * n_target:

        attempts += 1

        a = float(rng.uniform(*axis_range))
        b = float(rng.uniform(*axis_range))
        theta = float(rng.uniform(0.0, np.pi))

        if placed and rng.random() < touch_prob:

            # Encosta a nova elipse numa já colocada: a distância entre
            # centros menor que a soma dos raios garante contato depois
            # do recorte por oclusão.
            py, px, pa, pb, _ = placed[int(rng.integers(len(placed)))]

            ang = float(rng.uniform(0.0, 2.0 * np.pi))
            reach = (max(pa, pb) + max(a, b)) * float(rng.uniform(0.55, 0.95))

            cy, cx = py + reach * np.sin(ang), px + reach * np.cos(ang)

        else:
            cy = float(rng.uniform(0.0, size))
            cx = float(rng.uniform(0.0, size))

        mask = _rasterize_ellipse(yy, xx, cy, cx, a, b, theta)

        full = int(mask.sum())

        if full == 0:
            continue

        mask &= labels == 0                       # sem sobreposição

        area = int(mask.sum())

        if area < min_area_px or area < min_area_frac * full:
            continue                              # sobrou pouco: descarta

        labels[mask] = next_id
        fg_level[mask] = bg_level + contrast * float(rng.uniform(0.6, 1.4))

        placed.append((cy, cx, a, b, theta))
        next_id += 1

    # Fundo com iluminação não uniforme + objetos + borramento + ruído.
    tilt_y, tilt_x = rng.uniform(-0.08, 0.08, size=2)

    ramp = tilt_y * (yy / size - 0.5) + tilt_x * (xx / size - 0.5)

    image = bg_level + ramp

    foreground = labels > 0
    image[foreground] = fg_level[foreground]

    image = ndimage.gaussian_filter(
        image,
        sigma=float(rng.uniform(0.6, 1.6)),
    )

    image = image + rng.normal(
        0.0,
        float(rng.uniform(0.01, 0.10)),
        image.shape,
    )

    return np.clip(image, 0.0, 1.0).astype(np.float32), labels


class SyntheticEllipses(Dataset):
    """
    Dataset sintético determinístico: a amostra i depende apenas de
    (seed, i), então é reprodutível e seguro com workers.

    Devolve o mesmo dicionário do DSB2018Dataset.
    """

    def __init__(
        self,
        n_samples=256,
        size=128,
        seed=0,
        boundary_width=2,
        adaptive_boundary=False,
        **kwargs
    ):
        self.n_samples = n_samples
        self.size = size
        self.seed = seed
        self.boundary_width = boundary_width
        self.adaptive_boundary = adaptive_boundary
        self.kwargs = kwargs

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        rng = np.random.default_rng([self.seed, index])

        image, labels = make_synthetic_sample(
            rng,
            size=self.size,
            **self.kwargs
        )

        boundary_target = create_boundary_target(
            labels,
            boundary_width=self.boundary_width,
            adaptive=self.adaptive_boundary,
        )

        # 1 canal -> 3, para usar o mesmo modelo dos dados reais.
        image_rgb = np.repeat(image[None], 3, axis=0)

        return {
            "image": torch.from_numpy(image_rgb).float(),
            "instance_mask": torch.from_numpy(labels.astype(np.int64)),
            "boundary_target": torch.from_numpy(boundary_target),
            "image_path": f"synthetic/seed{self.seed}/{index:05d}",
        }


def touching_pairs(labels):
    """
    Pares de instâncias que compartilham fronteira (vizinhança-8).

    Serve para mostrar que o dataset cumpre o "muitas delas se tocando"
    do enunciado.
    """

    pairs = set()

    def add(a, b):
        m = (a > 0) & (b > 0) & (a != b)
        for u, v in zip(a[m].ravel(), b[m].ravel()):
            pairs.add((min(u, v), max(u, v)))

    add(labels[1:, :], labels[:-1, :])
    add(labels[:, 1:], labels[:, :-1])
    add(labels[1:, 1:], labels[:-1, :-1])
    add(labels[1:, :-1], labels[:-1, 1:])

    return pairs
