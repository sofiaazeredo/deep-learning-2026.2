"""
Parte 0.4 — o piso fácil do baseline.

Poucas elipses, lentas, sem oclusão, detecções = caixas verdadeiras sem id.
O mesmo tracker da Parte 1 tem que ficar com IDF1 muito perto de 1.

Roda com "python tests/test_baseline_synthetic.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector_sim import degrade
from src.metrics import evaluate_sequence
from src.synthetic import make_sequence
from src.tracker import Tracker

EASY = dict(n_objects=5, speed=0.5, occlusion_frames=0, n_frames=45)
TRACKER = dict(iou_threshold=0.2, max_age=20, min_hits=2, matcher="hungarian")


def _run(seed, **params):
    settings = dict(EASY)
    settings.update(params)
    _, tracks = make_sequence(seed=seed, **settings)
    pred = Tracker(**TRACKER).run(degrade(tracks, seed=seed))
    return evaluate_sequence(tracks, pred)


def test_easy_floor_idf1_near_one():
    scores = [_run(seed)["idf1"] for seed in (42, 123, 7)]

    assert min(scores) > 0.95
    assert sum(scores) / len(scores) > 0.98


def test_easy_floor_almost_no_switches():
    result = _run(42)

    assert result["id_switches"] <= 1
    assert result["id_count_error"] <= 1


def test_long_occlusion_hurts_identity():
    easy = _run(42)
    hard = _run(42, n_objects=3, speed=2.0, occlusion_frames=16)

    assert hard["idf1"] < easy["idf1"]


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
