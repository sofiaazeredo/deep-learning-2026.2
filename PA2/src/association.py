"""
Associação entre quadros (Parte 1.2 e Parte 2).

Regras diferentes dão números diferentes, então tudo que é regra vira
parâmetro explícito e entra no README e na apresentação.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.metrics import iou_matrix


def greedy_match(cost, threshold):
    """
    Casamento guloso por custo crescente, um-para-um.
    """

    cost = np.asarray(cost, dtype=np.float64)

    if cost.size == 0:
        return []

    pairs = [(float(cost[i, j]), i, j)
             for i in range(cost.shape[0])
             for j in range(cost.shape[1])
             if cost[i, j] <= threshold]
    pairs.sort()

    used_row, used_col, matches = set(), set(), []

    for _, i, j in pairs:
        if i in used_row or j in used_col:
            continue
        used_row.add(i)
        used_col.add(j)
        matches.append((int(i), int(j)))

    return matches


def hungarian_match(cost, threshold):
    """
    Casamento ótimo (scipy.optimize.linear_sum_assignment), um-para-um.
    """

    cost = np.asarray(cost, dtype=np.float64)

    if cost.size == 0:
        return []

    # SciPy não aceita inf; o portão marca pares inválidos com um
    # custo grande, e o limiar ainda os rejeita.
    finite = np.where(np.isfinite(cost), cost, 1e6)
    rows, cols = linear_sum_assignment(finite)

    return [(int(i), int(j))
            for i, j in zip(rows, cols)
            if cost[i, j] <= threshold]


def iou_cost(tracks, detections):
    """
    Custo geométrico: 1 - IoU entre caixa da track (observada ou prevista pela
    recorrência) e caixa detectada.
    """

    return 1.0 - iou_matrix(tracks, detections)


def cosine_cost(track_embeddings, detection_embeddings):
    """
    Custo de aparência: 1 - similaridade de cosseno (Trilha B).
    """

    tracks = np.asarray(track_embeddings, dtype=np.float64)
    dets = np.asarray(detection_embeddings, dtype=np.float64)

    if tracks.ndim == 1:
        tracks = tracks.reshape(1, -1)
    if dets.ndim == 1:
        dets = dets.reshape(1, -1)
    if tracks.size == 0 or dets.size == 0:
        return np.zeros((len(tracks), len(dets)))

    track_norm = np.linalg.norm(tracks, axis=1, keepdims=True)
    det_norm = np.linalg.norm(dets, axis=1, keepdims=True)
    tracks = np.divide(tracks, track_norm, out=np.zeros_like(tracks),
                       where=track_norm > 0)
    dets = np.divide(dets, det_norm, out=np.zeros_like(dets),
                     where=det_norm > 0)

    return 1.0 - tracks @ dets.T


def enlarge_boxes(boxes, scale):
    """
    Caixas (x, y, w, h) aumentadas `scale` vezes (uma escala por linha) em
    volta do próprio centro.
    """

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1, 1)
    center = boxes[:, :2] + boxes[:, 2:] / 2
    size = boxes[:, 2:] * scale
    return np.column_stack([center - size / 2, size])


def gate(cost, iou, iou_gate=0.3, row_mask=None):
    """
    Portão geométrico por cima da aparência (Trilha B): par com IoU abaixo
    de `iou_gate` ganha custo infinito e não casa, por menor que seja o
    cosseno.

    `row_mask` (n_tracks,) True aplica o portão nessa track. Sem máscara,
    aplica em todas.
    """

    gated = np.array(cost, dtype=np.float64, copy=True)
    blocked = np.asarray(iou, dtype=np.float64) < iou_gate

    if row_mask is not None:
        blocked = blocked & np.asarray(row_mask, dtype=bool).reshape(-1, 1)

    gated[blocked] = np.inf
    return gated
