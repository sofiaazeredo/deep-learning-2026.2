"""
Métricas de rastreamento implementadas do zero (entregável obrigatório).

Proibido usar motmetrics / TrackEval / py-motmetrics. O matching por IoU é o
mesmo do PA1 (src/metrics.py de lá), reaproveitado como o enunciado permite.

IDF1 exige uma atribuição global um-para-um entre identidades previstas e
verdadeiras ao longo da sequência inteira — não é um casamento por quadro.
"""


def iou_matrix(boxes_a, boxes_b):
    """
    IoU par a par entre dois conjuntos de caixas (x, y, w, h).
    """

    raise NotImplementedError


def match_frame(gt_boxes, pred_boxes, threshold=0.5):
    """
    Casamento um-para-um dentro de um quadro (guloso por IoU decrescente ou
    Hungarian). Base de IDF1 e da contagem de switches.
    """

    raise NotImplementedError


def idf1(gt_tracks, pred_tracks, threshold=0.5):
    """
    IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN), com a atribuição global
    ID-previsto <-> ID-verdadeiro resolvida por Hungarian sobre o custo
    acumulado na sequência inteira.

    Devolve (idf1, idp, idr, detalhes).
    """

    raise NotImplementedError


def id_switches(gt_tracks, pred_tracks, threshold=0.5):
    """
    Número de vezes que uma identidade verdadeira muda de identidade prevista
    entre observações consecutivas casadas.
    """

    raise NotImplementedError


def fragmentations(gt_tracks, pred_tracks, threshold=0.5):
    """
    Número de interrupções na cobertura de uma identidade verdadeira (ela
    estava casada, deixou de estar, voltou).
    """

    raise NotImplementedError


def unique_id_count_error(gt_tracks, pred_tracks):
    """
    |identidades únicas previstas - verdadeiras| no vídeo. É o análogo
    temporal do erro de contagem do PA1.
    """

    raise NotImplementedError


def evaluate_sequence(gt_tracks, pred_tracks, threshold=0.5):
    """
    Todas as métricas acima numa chamada, no formato que os scripts gravam em
    experiments/results/.
    """

    raise NotImplementedError
