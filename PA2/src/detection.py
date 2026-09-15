"""
Fonte de detecções (Parte 1.1). Congelada a partir da Parte 2.

Duas fontes, as duas usadas:
  - as detecções públicas do MOT17 (det/det.txt), em DPM, FRCNN ou SDP;
  - um detector pré-treinado do torchvision em inferência, classe person do
    COCO.

NMS é implementação própria: torchvision.ops.nms é proibido.
"""


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-maximum suppression escrito à mão.
    """

    raise NotImplementedError


def public_detections(sequence, detector="SDP", score_threshold=0.0):
    """
    Detecções públicas já filtradas por score e passadas pelo nosso NMS.
    """

    raise NotImplementedError


def torchvision_detections(frames, score_threshold=0.5, device="cuda"):
    """
    Faster R-CNN pré-treinado do torchvision, só a classe person do COCO.
    """

    raise NotImplementedError
