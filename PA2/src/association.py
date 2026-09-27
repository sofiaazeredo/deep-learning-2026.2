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

    rows, cols = linear_sum_assignment(cost)

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

    raise NotImplementedError


def gate(cost, iou, iou_gate=0.3, sigma=None):
    """
    Portão de associação. Geométrico por cima da aparência na Trilha B;
    adaptativo pela incerteza prevista quando o modelo da Trilha A emite sigma.
    """

    raise NotImplementedError
