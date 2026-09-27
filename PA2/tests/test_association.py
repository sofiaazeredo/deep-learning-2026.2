"""
Parte 1 — custo IoU e casamento guloso / Hungarian.

Roda com "python tests/test_association.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.association import cosine_cost, gate, greedy_match, hungarian_match, iou_cost


def test_iou_cost_of_identical_boxes_is_zero():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 10.0, 10.0]])

    assert np.allclose(iou_cost(boxes, boxes), np.eye(2) * 0 + (1 - np.eye(2)))
    assert iou_cost(boxes[:1], boxes[:1])[0, 0] == 0.0


def test_greedy_and_hungarian_are_one_to_one():
    cost = np.array([
        [0.1, 0.2, 0.05],
        [0.15, 0.3, 0.4],
    ])

    for matcher in (greedy_match, hungarian_match):
        pairs = matcher(cost, threshold=0.5)
        rows = [i for i, _ in pairs]
        cols = [j for _, j in pairs]
        assert len(rows) == len(set(rows))
        assert len(cols) == len(set(cols))


def test_cost_gate_rejects_pairs():
    cost = np.array([[0.9, 0.1], [0.2, 0.95]])

    assert greedy_match(cost, threshold=0.3) == [(0, 1), (1, 0)]
    assert hungarian_match(cost, threshold=0.05) == []


def test_cosine_cost_of_parallel_vectors_is_zero():
    tracks = np.array([[1.0, 0.0], [0.0, 1.0]])
    dets = np.array([[2.0, 0.0], [0.0, 3.0]])

    cost = cosine_cost(tracks, dets)

    assert np.allclose(np.diag(cost), 0.0)
    assert cost[0, 1] == 1.0


def test_gate_blocks_low_iou_only_on_masked_rows():
    cost = np.array([[0.1, 0.2], [0.1, 0.2]])
    iou = np.array([[0.9, 0.0], [0.9, 0.0]])

    gated = gate(cost, iou, iou_gate=0.3, row_mask=[True, False])

    assert np.isinf(gated[0, 1])
    assert gated[1, 1] == 0.2


def test_empty_cost():
    empty = np.zeros((0, 3))

    assert greedy_match(empty, 0.5) == []
    assert hungarian_match(empty, 0.5) == []


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
