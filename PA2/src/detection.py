"""
Fonte de detecções (Parte 1.1). Congelada a partir da Parte 2.

Duas fontes, as duas usadas:
  - as detecções públicas do MOT17 (det/det.txt), em DPM, FRCNN ou SDP;
  - um detector pré-treinado do torchvision em inferência, classe person do
    COCO.

NMS é implementação própria: torchvision.ops.nms é proibido.
"""

from pathlib import Path

import numpy as np

from src.metrics import iou_matrix

COCO_PERSON = 1

# Limiar de score das detecções que ENTRAM no rastreamento, por fonte. As
# públicas entram como vieram (o MOTChallenge já as limiarizou; o DPM tem
# score não calibrado e negativo em ~41% das linhas, e cortar em 0 jogava
# parte do detector fora). O Faster R-CNN do torchvision é guardado a partir
# de 0,05 (o piso do próprio modelo, para o AP ver o ranking inteiro); o
# limiar de rastreamento dele sai do sweep no treino
# (scripts/run_baseline.py --sweep-score).
TORCHVISION_CACHE_SCORE = 0.05
TRACK_SCORE_THRESHOLD = {"torchvision": 0.8}   # sweep no treino: 0,3 -> 0,366 ... 0,8 -> 0,461, 0,9 -> 0,433


def default_score_threshold(detector):
    """
    None = sem corte (detecções públicas); um número = o limiar da fonte.
    """

    return TRACK_SCORE_THRESHOLD.get(detector)


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-maximum suppression escrito à mão.

    Ordena por score decrescente e descarta caixas com IoU >= limiar contra
    uma caixa já mantida. Devolve os índices no vetor original.
    """

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    if boxes.size == 0:
        return np.zeros(0, dtype=int)

    order = np.argsort(-scores, kind="stable")
    keep = []

    while order.size:
        index = int(order[0])
        keep.append(index)

        if order.size == 1:
            break

        rest = order[1:]
        overlap = iou_matrix(boxes[index:index + 1], boxes[rest])[0]
        order = rest[overlap < iou_threshold]

    return np.asarray(keep, dtype=int)


def _rows_to_detections(rows, score_threshold, nms_threshold):
    by_frame = {}

    for row in rows:
        score = float(row[6]) if len(row) > 6 else 1.0
        if score_threshold is not None and score < score_threshold:
            continue
        frame = int(row[0])
        box = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
        by_frame.setdefault(frame, []).append((box, score))

    detections = []

    for frame in sorted(by_frame):
        boxes = np.array([item[0] for item in by_frame[frame]], dtype=np.float64)
        scores = np.array([item[1] for item in by_frame[frame]], dtype=np.float64)
        keep = nms(boxes, scores, iou_threshold=nms_threshold)

        kept = [(frame, *boxes[i], float(scores[i])) for i in keep]
        kept.sort(key=lambda item: -item[5])
        detections.extend(kept)

    return detections


def public_detections(sequence, detector="SDP", score_threshold=None,
                      nms_threshold=0.5):
    """
    Detecções públicas já filtradas por score e passadas pelo nosso NMS.

    `sequence` é uma lista de linhas MOT (gt/det) ou um caminho/nome de
    sequência; neste caso o det.txt do detector pedido é lido.
    Saída: (frame, x, y, w, h, score), por quadro e por score decrescente.
    """

    if isinstance(sequence, (str, Path)):
        from src.dataset import load_sequence
        _, _, rows = load_sequence(sequence, detector=detector)
    else:
        rows = sequence

    return _rows_to_detections(rows, score_threshold, nms_threshold)


def load_torchvision_detector(device="cuda"):
    """
    Faster R-CNN pré-treinado no COCO (torchvision), em modo de inferência.
    Carregado uma vez e reaproveitado em todos os blocos de quadros.
    """

    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    model.to(device)
    return model


def torchvision_detections(frames, score_threshold=0.5, device="cuda",
                           nms_threshold=0.5, model=None, first_frame=1):
    """
    Faster R-CNN pré-treinado do torchvision, só a classe person do COCO.
    `frames` é uma lista de imagens HWC RGB (uint8 ou float), numeradas a
    partir de `first_frame` (1 no MOT17). `model` já carregado evita
    recarregar a rede a cada bloco. Aplica o nosso NMS depois do filtro de
    score. O laço de rastreamento não chama isto: o resultado entra em cache
    no formato det.txt (ver cache_torchvision_sequence).
    """

    import torch
    from torchvision.transforms.functional import to_tensor

    if model is None:
        model = load_torchvision_detector(device)

    detections = []

    with torch.no_grad():
        for index, frame in enumerate(frames, start=first_frame):
            image = np.asarray(frame)
            if image.dtype != np.uint8:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            tensor = to_tensor(image).to(device)
            output = model([tensor])[0]

            labels = output["labels"].detach().cpu().numpy()
            boxes_xyxy = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()

            person = labels == COCO_PERSON
            boxes_xyxy = boxes_xyxy[person]
            scores = scores[person]

            keep_score = scores >= score_threshold
            boxes_xyxy = boxes_xyxy[keep_score]
            scores = scores[keep_score]

            if len(boxes_xyxy) == 0:
                continue

            boxes = np.column_stack([
                boxes_xyxy[:, 0],
                boxes_xyxy[:, 1],
                boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
                boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
            ])
            keep = nms(boxes, scores, iou_threshold=nms_threshold)

            for i in keep:
                detections.append((index, float(boxes[i, 0]), float(boxes[i, 1]),
                                   float(boxes[i, 2]), float(boxes[i, 3]),
                                   float(scores[i])))

    detections.sort(key=lambda item: (item[0], -item[5]))
    return detections


def write_det_txt(path, detections):
    """
    Cache no formato MOT: frame, id, x, y, w, h, conf.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as handle:
        for row in detections:
            frame, x, y, w, h, score = row[:6]
            handle.write(f"{int(frame)},-1,{x:.3f},{y:.3f},{w:.3f},{h:.3f},"
                         f"{score:.6f}\n")


def cache_torchvision_sequence(sequence_dir, output_path, score_threshold=0.5,
                               device="cuda", nms_threshold=0.5, chunk_size=64):
    """
    Roda o Faster R-CNN numa pasta img1/ e grava det.txt. Tracking só lê
    o arquivo — o detector não volta a rodar. Os quadros são lidos em blocos
    de `chunk_size`: uma sequência 1080p inteira não cabe na RAM.
    """

    from PIL import Image

    image_dir = Path(sequence_dir) / "img1"
    paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"nenhum quadro em {image_dir}")

    model = load_torchvision_detector(device)
    detections = []

    for start in range(0, len(paths), chunk_size):
        frames = [np.array(Image.open(path).convert("RGB"))
                  for path in paths[start:start + chunk_size]]
        detections.extend(torchvision_detections(
            frames, score_threshold=score_threshold, device=device,
            nms_threshold=nms_threshold, model=model, first_frame=start + 1))

    write_det_txt(output_path, detections)
    return detections
