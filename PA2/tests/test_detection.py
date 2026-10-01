"""
Parte 1 — NMS próprio e detecções públicas.

Roda com "python tests/test_detection.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import torch

import src.detection as detection
from src.detection import (
    default_score_threshold,
    nms,
    public_detections,
    torchvision_detections,
)
from src.metrics import iou_matrix


def test_nms_known_keep_and_drop():
    # A e B se sobrepõem (IoU = 90/110 > 0.5); C está longe.
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 0.0, 10.0, 10.0],
        [50.0, 50.0, 10.0, 10.0],
    ])
    scores = np.array([0.9, 0.8, 0.7])

    assert np.isclose(iou_matrix(boxes[0:1], boxes[1:2])[0, 0], 90 / 110)
    keep = nms(boxes, scores, iou_threshold=0.5)

    assert list(keep) == [0, 2]


def test_nms_threshold_is_inclusive():
    # IoU exatamente 0.5: inter 200 / união 400. Deve suprimir.
    boxes = np.array([
        [0.0, 0.0, 30.0, 10.0],
        [10.0, 0.0, 30.0, 10.0],
    ])
    scores = np.array([0.9, 0.8])

    assert np.isclose(iou_matrix(boxes[0:1], boxes[1:2])[0, 0], 0.5)
    assert list(nms(boxes, scores, iou_threshold=0.5)) == [0]


def test_nms_empty():
    assert nms(np.zeros((0, 4)), np.zeros(0)).size == 0


def test_public_detections_filters_score_and_applies_nms():
    rows = [
        (1, -1, 0.0, 0.0, 10.0, 10.0, 0.9, None, None),
        (1, -1, 1.0, 0.0, 10.0, 10.0, 0.8, None, None),
        (1, -1, 50.0, 50.0, 10.0, 10.0, 0.2, None, None),
        (2, -1, 0.0, 0.0, 10.0, 10.0, 0.6, None, None),
    ]

    detections = public_detections(rows, score_threshold=0.5, nms_threshold=0.5)

    assert [row[0] for row in detections] == [1, 2]
    assert detections[0][1:6] == (0.0, 0.0, 10.0, 10.0, 0.9)
    assert detections[1][5] == 0.6


def test_public_detections_sorted_by_score_within_frame():
    rows = [
        (1, -1, 0.0, 0.0, 10.0, 10.0, 0.4, None, None),
        (1, -1, 50.0, 0.0, 10.0, 10.0, 0.9, None, None),
    ]

    detections = public_detections(rows)

    assert detections[0][5] == 0.9
    assert detections[1][5] == 0.4


class FakeFasterRCNN:
    """
    Faz o papel do Faster R-CNN: devolve, para cada imagem, uma pessoa
    boa, uma pessoa com score baixo e um carro (label 3).
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, images):
        self.calls += 1
        out = []
        for _ in images:
            out.append({
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 60.0],
                                       [50.0, 20.0, 70.0, 60.0],
                                       [0.0, 0.0, 5.0, 5.0]]),
                "labels": torch.tensor([1, 1, 3]),
                "scores": torch.tensor([0.9, 0.3, 0.95]),
            })
        return out


def test_torchvision_detections_uses_a_loaded_model():
    # O modelo entra carregado (a inferência em blocos não recarrega a
    # rede a cada bloco). Só a pessoa acima do limiar sai, em xywh.
    model = FakeFasterRCNN()
    frames = [np.zeros((80, 80, 3), dtype=np.uint8) for _ in range(3)]

    dets = torchvision_detections(frames, score_threshold=0.5, device="cpu",
                                  model=model)

    assert model.calls == 3
    assert dets == [(i, 10.0, 20.0, 20.0, 40.0, dets[0][5]) for i in (1, 2, 3)]


def test_cache_torchvision_sequence_streams_in_chunks(tmp_path, monkeypatch):
    # Quadros lidos em blocos (sem carregar a sequência inteira na RAM) e a
    # numeração continua entre blocos.
    from PIL import Image

    (tmp_path / "img1").mkdir()
    for frame in range(1, 6):
        Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8)).save(
            tmp_path / "img1" / f"{frame:06d}.jpg")
    model = FakeFasterRCNN()
    monkeypatch.setattr(detection, "load_torchvision_detector",
                        lambda device="cpu": model)

    dets = detection.cache_torchvision_sequence(
        tmp_path, tmp_path / "det.txt", device="cpu", chunk_size=2)

    assert [row[0] for row in dets] == [1, 2, 3, 4, 5]
    assert (tmp_path / "det.txt").read_text().count("\n") == 5


def test_public_detections_keep_negative_scores_by_default():
    # O DPM tem score não calibrado (negativo em ~41% das linhas): cortar
    # em 0 jogava fora parte do detector. Sem limiar, tudo entra.
    rows = [(1, -1, 0.0, 0.0, 10.0, 10.0, -0.4),
            (1, -1, 50.0, 0.0, 10.0, 10.0, 1.3)]

    assert len(public_detections(rows)) == 2
    assert len(public_detections(rows, score_threshold=0.0)) == 1


def test_default_score_threshold_per_source():
    # Detecção pública entra como veio; o torchvision tem o limiar de
    # rastreamento escolhido no treino.
    assert default_score_threshold("SDP") is None
    assert default_score_threshold("DPM") is None
    assert 0.0 < default_score_threshold("torchvision") < 1.0


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
