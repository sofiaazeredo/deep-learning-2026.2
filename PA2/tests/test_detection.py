"""
Parte 1 — NMS próprio e detecções públicas.

Roda com "python tests/test_detection.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.detection import nms, public_detections
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


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
