"""
Parte 1 — tracker ingênuo: última caixa, nascimento, morte, min_hits.

Roda com "python tests/test_tracker.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracker import Tracker


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
