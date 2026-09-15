"""
Associação entre quadros (Parte 1.2 e Parte 2).

Regras diferentes dão números diferentes, então tudo que é regra vira
parâmetro explícito e entra no README e na apresentação.
"""


def greedy_match(cost, threshold):
    """
    Casamento guloso por custo crescente, um-para-um.
    """

    raise NotImplementedError


def hungarian_match(cost, threshold):
    """
    Casamento ótimo (scipy.optimize.linear_sum_assignment), um-para-um.
    """

    raise NotImplementedError


def iou_cost(tracks, detections):
    """
    Custo geométrico: 1 - IoU entre caixa da track (observada ou prevista pela
    recorrência) e caixa detectada.
    """

    raise NotImplementedError


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
