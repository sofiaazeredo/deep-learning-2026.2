"""
Encoder de aparência (Trilha B da Parte 2, Eixo 3 da Parte 3).

Recorte da caixa -> embedding D-dimensional, com um encoder pequeno e
pré-treinado (permitido pelo enunciado). O encoder fica congelado; quem
aprende é o agregador recorrente e a perda contrastiva.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.functional import normalize, to_tensor

from src.dataset import load_frame, load_sequence, resolve_root

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CROP = (128, 64)   # altura, largura (ReID)


def crop(frame, box, output_size=DEFAULT_CROP):
    """
    Recorte HWC uint8 da caixa (x, y, w, h), preso à imagem, redimensionado
    para (altura, largura) = output_size.
    """

    image = np.asarray(frame)
    height, width = image.shape[:2]
    x, y, box_w, box_h = [float(v) for v in box]

    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = int(np.ceil(x + max(box_w, 1.0)))
    y2 = int(np.ceil(y + max(box_h, 1.0)))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    if x2 <= x1 or y2 <= y1:
        patch = np.zeros((max(y2 - y1, 1), max(x2 - x1, 1), 3), dtype=np.uint8)
    else:
        patch = image[y1:y2, x1:x2]

    from PIL import Image

    out_h, out_w = int(output_size[0]), int(output_size[1])
    return np.asarray(Image.fromarray(patch).resize((out_w, out_h)))


class CropEncoder(nn.Module):
    def __init__(self, backbone="resnet18", embed_dim=128, freeze=True):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"backbone não suportado: {backbone}")

        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.proj = nn.Linear(512, embed_dim)
        self.embed_dim = int(embed_dim)
        self.freeze = bool(freeze)

        if self.freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

        # Proj aleatória mas determinística: o cache de embeddings tem
        # que ser o mesmo no treino (gt) e na inferência (det).
        with torch.random.fork_rng():
            torch.manual_seed(0)
            nn.init.kaiming_uniform_(self.proj.weight, a=5 ** 0.5)
            if self.proj.bias is not None:
                bound = 1.0 / (self.proj.in_features ** 0.5)
                nn.init.uniform_(self.proj.bias, -bound, bound)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def embed(self, crops, device=None):
        """
        Lote de recortes HWC -> embeddings L2-normalizados (N, D).
        """

        if len(crops) == 0:
            return torch.zeros(0, self.embed_dim)

        device = device or next(self.parameters()).device
        batch = []

        for crop_image in crops:
            tensor = to_tensor(np.asarray(crop_image, dtype=np.uint8))
            batch.append(normalize(tensor, IMAGENET_MEAN, IMAGENET_STD))

        inputs = torch.stack(batch).to(device)

        with torch.set_grad_enabled(not self.freeze and self.training):
            features = self.backbone(inputs).flatten(1)
            embeddings = self.proj(features)

        return F.normalize(embeddings, dim=-1)


def encoder_tag(encoder):
    weight = encoder.proj.weight.detach().cpu().numpy().tobytes()
    return hashlib.md5(weight).hexdigest()[:8]


def cache_file(scene, kind="gt", root="data/MOT17", tag=None):
    name = f"{scene}_{kind}" if tag is None else f"{scene}_{kind}_{tag}"
    return resolve_root(root) / "cache" / "sdp_resnet18" / f"{name}.npz"


def _save_cache(path, frames, keys, embeddings):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=np.asarray(frames, dtype=np.int32),
                        keys=np.asarray(keys, dtype=np.int32),
                        embeddings=np.asarray(embeddings, dtype=np.float32))


def load_embedding_cache(path):
    data = np.load(path)
    table = {}
    for frame, key, vector in zip(data["frames"], data["keys"], data["embeddings"]):
        table[(int(frame), int(key))] = np.asarray(vector, dtype=np.float32)
    return table


@torch.no_grad()
def cache_sequence_embeddings(scene, encoder, kind="gt", detector="SDP",
                              root="data/MOT17", device="cpu",
                              score_threshold=0.0, nms_threshold=0.5,
                              batch_size=64):
    """
    kind='gt'  : chave (frame, id)
    kind='det' : chave (frame, índice da detecção depois do NMS)
    """

    tag = encoder_tag(encoder)
    path = cache_file(scene, kind=kind, root=root, tag=tag)
    if path.exists():
        return load_embedding_cache(path)

    from src.detection import public_detections

    info, gt, raw = load_sequence(scene, detector=detector, root=root)
    encoder = encoder.to(device)
    encoder.eval()

    if kind == "gt":
        rows = [(int(row[0]), int(row[1]), row[2:6]) for row in gt]
    else:
        dets = public_detections(raw, score_threshold=score_threshold,
                                 nms_threshold=nms_threshold)
        rows = [(int(row[0]), i, row[1:5]) for i, row in enumerate(dets)]

    frames, keys, vectors = [], [], []
    by_frame = {}
    for frame, key, box in rows:
        by_frame.setdefault(frame, []).append((key, box))

    frame_ids = sorted(by_frame)
    for i, frame in enumerate(frame_ids, start=1):
        image = load_frame(info["path"], frame, info["im_dir"], info["im_ext"])
        items = by_frame[frame]
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            crops = [crop(image, box) for _, box in chunk]
            embeds = encoder.embed(crops, device=device).cpu().numpy()
            for (key, _), vector in zip(chunk, embeds):
                frames.append(frame)
                keys.append(key)
                vectors.append(vector)
        if i == 1 or i % 100 == 0 or i == len(frame_ids):
            print(f"  {scene} {kind}  {i}/{len(frame_ids)} quadros  "
                  f"{len(vectors)} recortes", flush=True)

    _save_cache(path, frames, keys, vectors)
    return load_embedding_cache(path)


@torch.no_grad()
def embed_detections(scene, dets, encoder, device="cpu", batch_size=64,
                     root="data/MOT17"):
    """
    Recorta e embeda uma lista (frame, x, y, w, h, ...) na ordem dada.
    Usado na Parte 5: as caixas degradadas não cabem no cache do SDP.
    """

    info, _, _ = load_sequence(scene, root=root)
    encoder = encoder.to(device)
    encoder.eval()
    vectors = [None] * len(dets)
    by_frame = {}
    for index, row in enumerate(dets):
        by_frame.setdefault(int(row[0]), []).append((index, row[1:5]))

    for frame, items in by_frame.items():
        image = load_frame(info["path"], frame, info["im_dir"], info["im_ext"])
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            crops = [crop(image, box) for _, box in chunk]
            embeds = encoder.embed(crops, device=device).cpu().numpy()
            for (index, _), vector in zip(chunk, embeds):
                vectors[index] = vector
    return vectors
