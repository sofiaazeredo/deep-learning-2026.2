"""
Parte 1 — tracker ingênuo: última caixa, nascimento, morte, min_hits.

Roda com "python tests/test_tracker.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.tracker import Tracker


class FakeAppearance:
    cell = "gru"

    def init_state(self, batch=1, device="cpu"):
        return None

    def step(self, embedding, state):
        return embedding, embedding


def boxes_at(frames, x, y):
    return [(t, x, y, 10.0, 10.0, 1.0) for t in frames]


def test_ids_persist_across_frames():
    tracker = Tracker(iou_threshold=0.3, max_age=5, min_hits=1,
                      matcher="hungarian")
    detections = boxes_at(range(1, 6), 0.0, 0.0) + boxes_at(range(1, 6), 50.0, 0.0)

    tracks = tracker.run(detections)
    ids = {row[1] for row in tracks}

    assert len(ids) == 2
    assert all(len(row) == 6 for row in tracks)


def test_unmatched_detection_starts_new_id():
    tracker = Tracker(min_hits=1, max_age=5, iou_threshold=0.3)
    detections = (boxes_at([1], 0.0, 0.0)
                  + boxes_at([2], 0.0, 0.0)
                  + boxes_at([2], 80.0, 0.0))

    tracks = tracker.run(detections)
    ids_frame2 = {row[1] for row in tracks if row[0] == 2}

    assert len(ids_frame2) == 2


def test_track_dies_after_max_age_misses():
    tracker = Tracker(min_hits=1, max_age=2, iou_threshold=0.3)
    detections = boxes_at([1, 2], 0.0, 0.0) + boxes_at([6], 0.0, 0.0)

    tracks = tracker.run(detections)
    ids = {row[1] for row in tracks}

    assert len(ids) == 2


def test_no_output_while_unmatched():
    tracker = Tracker(min_hits=1, max_age=10, iou_threshold=0.3)
    detections = boxes_at([1, 2], 0.0, 0.0)

    tracks = tracker.run(detections)

    assert [row[0] for row in tracks] == [1, 2]


def test_min_hits_drops_unconfirmed_frames():
    tracker = Tracker(min_hits=3, max_age=5, iou_threshold=0.3)
    detections = boxes_at(range(1, 6), 0.0, 0.0)

    tracks = tracker.run(detections)

    assert [row[0] for row in tracks] == [3, 4, 5]
    assert len({row[1] for row in tracks}) == 1


def test_appearance_keeps_ids_after_occlusion_far_from_last_box():
    # Depois do buraco as pessoas reaparecem no lugar uma da outra.
    # IoU puro trocaria os ids; a aparência (portão desligado no miss)
    # segue o embedding.
    tracker = Tracker(iou_threshold=0.3, max_age=5, min_hits=1,
                      matcher="hungarian", motion=FakeAppearance(),
                      appearance_threshold=0.5)
    detections = [(1, 0.0, 0.0, 10.0, 10.0, 1.0),
                  (1, 80.0, 0.0, 10.0, 10.0, 1.0),
                  (4, 80.0, 0.0, 10.0, 10.0, 1.0),
                  (4, 0.0, 0.0, 10.0, 10.0, 1.0)]
    embeddings = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]

    tracks = tracker.run(detections, embeddings=embeddings)
    first = {row[2:4]: row[1] for row in tracks if row[0] == 1}
    later = {row[2:4]: row[1] for row in tracks if row[0] == 4}

    assert first[(0.0, 0.0)] == later[(80.0, 0.0)]
    assert first[(80.0, 0.0)] == later[(0.0, 0.0)]

    iou_only = Tracker(iou_threshold=0.3, max_age=5, min_hits=1).run(detections)
    first_iou = {row[2:4]: row[1] for row in iou_only if row[0] == 1}
    later_iou = {row[2:4]: row[1] for row in iou_only if row[0] == 4}
    assert first_iou[(0.0, 0.0)] == later_iou[(0.0, 0.0)]


class FakeMotion:
    kind = "geometry"

    def init_state(self, batch=1, device="cpu"):
        return None

    def step(self, box, state):
        pred = torch.as_tensor(box, dtype=torch.float32).clone()
        if pred.dim() == 1:
            pred[0] = pred[0] + 0.25
        else:
            pred = pred.clone()
            pred[:, 0] = pred[:, 0] + 0.25
        return pred, state


def test_geometry_rematches_on_predicted_box_after_miss():
    # A RNN de mentira empurra cx em +0.25 por passo. Depois de um miss o
    # casamento usa a caixa prevista, não a última observada.
    tracker = Tracker(iou_threshold=0.3, max_age=5, min_hits=1,
                      matcher="hungarian", motion=FakeMotion(),
                      image_size=(200.0, 40.0))
    detections = [(1, 0.0, 0.0, 10.0, 10.0, 1.0),
                  (1, 80.0, 0.0, 10.0, 10.0, 1.0),
                  (3, 100.0, 0.0, 10.0, 10.0, 1.0),
                  (3, 180.0, 0.0, 10.0, 10.0, 1.0)]

    tracks = tracker.run(detections, image_size=(200.0, 40.0))
    first = {row[2:4]: row[1] for row in tracks if row[0] == 1}
    later = {row[2:4]: row[1] for row in tracks if row[0] == 3}
    assert first[(0.0, 0.0)] == later[(100.0, 0.0)]
    assert first[(80.0, 0.0)] == later[(180.0, 0.0)]

    iou_only = Tracker(iou_threshold=0.3, max_age=5, min_hits=1).run(detections)
    first_iou = {row[2:4]: row[1] for row in iou_only if row[0] == 1}
    later_iou = {row[2:4]: row[1] for row in iou_only if row[0] == 3}
    if (100.0, 0.0) in later_iou:
        assert first_iou[(0.0, 0.0)] != later_iou[(100.0, 0.0)]


def test_miss_prefer_iou_keeps_nearby_box_against_swapped_embeddings():
    # Depois do miss as caixas quase não andam, mas os embeddings vêm
    # trocados. Sem a correção a aparência troca os ids; com
    # miss_prefer_iou o IoU da última caixa ganha.
    detections = [(1, 0.0, 0.0, 10.0, 10.0, 1.0),
                  (1, 80.0, 0.0, 10.0, 10.0, 1.0),
                  (3, 1.0, 0.0, 10.0, 10.0, 1.0),
                  (3, 81.0, 0.0, 10.0, 10.0, 1.0)]
    embeddings = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
    ]
    app_tracks = Tracker(iou_threshold=0.3, max_age=5, min_hits=1,
                         motion=FakeAppearance(),
                         appearance_threshold=0.5).run(detections, embeddings)
    fix_tracks = Tracker(iou_threshold=0.3, max_age=5, min_hits=1,
                         motion=FakeAppearance(), appearance_threshold=0.5,
                         miss_prefer_iou=True).run(detections, embeddings)
    first = {row[2:4]: row[1] for row in app_tracks if row[0] == 1}
    later_app = {row[2:4]: row[1] for row in app_tracks if row[0] == 3}
    later_fix = {row[2:4]: row[1] for row in fix_tracks if row[0] == 3}
    assert first[(0.0, 0.0)] == later_app.get((81.0, 0.0))
    assert first[(0.0, 0.0)] == later_fix.get((1.0, 0.0))


def test_miss_resets_consecutive_hits_before_confirmation():
    tracker = Tracker(min_hits=3, max_age=5, iou_threshold=0.3)
    detections = boxes_at([1, 2, 4, 5, 6], 0.0, 0.0)

    tracks = tracker.run(detections)

    assert [row[0] for row in tracks] == [6]


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
