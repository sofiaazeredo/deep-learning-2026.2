"""
Parte 0.2 — testes do simulador de detector.

O simulador recebe as caixas verdadeiras e as estraga de propósito:
descarta p% delas, adiciona ruído nas coordenadas e injeta falsos positivos.
Os testes conferem cada botão separado, com a resposta conhecida, e que a
saída não vaza identidade.

Roda com "python tests/test_detector_sim.py" ou sob pytest.
"""

import sys
from pathlib import Path

# Rodar "python tests/x.py" coloca tests/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.detector_sim import degrade, INTENSITIES
from src.synthetic import make_sequence

IMAGE_SIZE = (128, 128)


def grid_boxes(n_frames=200, per_frame=10):
    """
    Caixas verdadeiras fáceis de conferir: per_frame caixas 10x20 por quadro,
    no formato do gerador (frame, id, x, y, w, h, visibility).
    """

    return [(t, i + 1, 10.0 * i, 50.0, 10.0, 20.0, 1.0)
            for t in range(n_frames) for i in range(per_frame)]


def test_no_degradation_keeps_every_box():
    boxes = grid_boxes(n_frames=3)

    detections = degrade(boxes, seed=0)

    assert sorted(d[:5] for d in detections) == \
        sorted((b[0], *b[2:6]) for b in boxes)


def test_output_has_no_identity():
    detections = degrade(grid_boxes(n_frames=3), seed=0)

    assert all(len(d) == 6 for d in detections)   # frame, x, y, w, h, score


def test_detections_sorted_by_score_within_frame():
    detections = degrade(grid_boxes(n_frames=5), false_positive_rate=0.5,
                         image_size=IMAGE_SIZE, seed=0)

    frames = [d[0] for d in detections]
    assert frames == sorted(frames)

    for t in set(frames):
        scores = [d[5] for d in detections if d[0] == t]
        assert scores == sorted(scores, reverse=True)


def test_scores_do_not_separate_false_positives():
    boxes = grid_boxes()
    true_scores = [d[5] for d in degrade(boxes, seed=0)]

    only_fp = degrade(boxes, drop_rate=1.0, false_positive_rate=1.0,
                      image_size=IMAGE_SIZE, seed=0)
    fp_scores = [d[5] for d in only_fp]

    assert abs(np.mean(true_scores) - np.mean(fp_scores)) < 0.02
    assert min(true_scores + fp_scores) >= 0.5
    assert max(true_scores + fp_scores) <= 1.0


def test_drop_rate_removes_that_fraction():
    boxes = grid_boxes()

    kept = len(degrade(boxes, drop_rate=0.3, seed=0)) / len(boxes)

    assert abs(kept - 0.7) < 0.03, kept


def test_drop_everything_leaves_nothing():
    assert degrade(grid_boxes(n_frames=3), drop_rate=1.0, seed=0) == []


def test_coord_noise_is_relative_to_box_size():
    # Caixas 10x20 com ruído de 10%: o centro erra ~1 px em x e ~2 px em y.
    boxes = grid_boxes()
    detections = degrade(boxes, coord_noise=0.1, seed=0)

    dx, dy = [], []
    for t, x, y, w, h, _ in detections:
        cx, cy = x + w / 2, y + h / 2
        # A caixa verdadeira mais próxima (espaçadas de 10 px em x).
        i = int(round((cx - 5.0) / 10.0))
        dx.append(cx - (10.0 * i + 5.0))
        dy.append(cy - 60.0)

    assert abs(np.std(dx) - 1.0) < 0.1, np.std(dx)
    assert abs(np.std(dy) - 2.0) < 0.2, np.std(dy)
    assert abs(np.mean(dy)) < 0.1


def test_coord_noise_keeps_positive_size():
    detections = degrade(grid_boxes(), coord_noise=2.0, seed=0)

    assert all(d[3] >= 1.0 and d[4] >= 1.0 for d in detections)


def test_false_positive_rate_is_per_true_box():
    boxes = grid_boxes()

    detections = degrade(boxes, false_positive_rate=0.2,
                         image_size=IMAGE_SIZE, seed=0)
    n_fp = len(detections) - len(boxes)

    assert abs(n_fp / len(boxes) - 0.2) < 0.03, n_fp / len(boxes)


def test_false_positives_inside_image():
    width, height = IMAGE_SIZE

    detections = degrade(grid_boxes(), drop_rate=1.0, false_positive_rate=1.0,
                         image_size=IMAGE_SIZE, seed=0)

    assert detections
    for _, x, y, w, h, _ in detections:
        assert x >= 0 and y >= 0 and x + w <= width and y + h <= height


def test_false_positives_require_image_size():
    try:
        degrade(grid_boxes(n_frames=3), false_positive_rate=0.1, seed=0)
    except ValueError:
        return

    raise AssertionError("falso positivo sem image_size deveria falhar")


def test_invalid_rates_fail():
    for kwargs in ({"drop_rate": 1.5}, {"drop_rate": -0.1},
                   {"coord_noise": -1.0}, {"false_positive_rate": -0.1}):
        try:
            degrade(grid_boxes(n_frames=3), image_size=IMAGE_SIZE, **kwargs)
        except ValueError:
            continue

        raise AssertionError(f"{kwargs} deveria falhar")


def test_same_seed_same_detections():
    kwargs = dict(drop_rate=0.2, coord_noise=0.05, false_positive_rate=0.1,
                  image_size=IMAGE_SIZE)

    assert degrade(grid_boxes(), seed=3, **kwargs) == \
        degrade(grid_boxes(), seed=3, **kwargs)


def test_input_is_not_modified():
    boxes = grid_boxes(n_frames=3)
    copy = list(boxes)

    degrade(boxes, drop_rate=0.5, coord_noise=0.1, false_positive_rate=0.5,
            image_size=IMAGE_SIZE, seed=0)

    assert boxes == copy


def test_runs_on_synthetic_ground_truth():
    _, tracks = make_sequence(seed=0)

    detections = degrade(tracks, **INTENSITIES["media"],
                         image_size=IMAGE_SIZE, seed=0)

    assert detections
    assert {d[0] for d in detections} <= {row[0] for row in tracks}


def test_intensities_grow_from_light_to_strong():
    light, medium, strong = (INTENSITIES[k] for k in ("leve", "media", "forte"))

    for knob in ("drop_rate", "coord_noise", "false_positive_rate"):
        assert 0 < light[knob] < medium[knob] < strong[knob], knob


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
