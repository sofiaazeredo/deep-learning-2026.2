"""
Parte 1 — leitura MOT17 e split por sequência.

Roda com "python tests/test_dataset.py" ou sob pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import (
    CAMERA,
    TEST_SCENES,
    TRAIN_SCENES,
    create_splits,
    median_occlusion,
    read_mot_file,
    scene_id,
    split_ground_truth,
)


def test_read_mot_file_types(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("1,7,10,20,30,40,1,1,0.9\n2,7,11,21,30,40,0,7,1.0\n")

    rows = read_mot_file(path)

    assert rows[0] == (1, 7, 10.0, 20.0, 30.0, 40.0, 1.0, 1, 0.9)
    assert rows[1][6] == 0.0 and rows[1][7] == 7


def test_read_det_file_without_class(tmp_path):
    path = tmp_path / "det.txt"
    path.write_text("1,-1,10,20,30,40,0.8\n")

    rows = read_mot_file(path)

    assert rows[0][:7] == (1, -1, 10.0, 20.0, 30.0, 40.0, 0.8)
    assert rows[0][7] is None and rows[0][8] is None


def test_split_ground_truth_keeps_only_real_pedestrians():
    rows = [
        (1, 1, 0, 0, 10, 10, 1, 1, 1.0),
        (1, 2, 0, 0, 10, 10, 0, 1, 1.0),
        (1, 3, 0, 0, 10, 10, 1, 7, 1.0),
        (1, 4, 0, 0, 10, 10, 1, 8, 0.2),
    ]

    pedestrians, distractors = split_ground_truth(rows)

    assert [row[1] for row in pedestrians] == [1]
    assert [row[1] for row in distractors] == [2, 3, 4]


def test_create_splits_is_fixed_and_ignores_seed():
    sequences = ["MOT17-02-SDP", "04", "05", "09", "10", "11", "13"]

    first = create_splits(sequences, seed=1)
    second = create_splits(sequences, seed=99)

    assert first == second
    assert first["train"] == list(TRAIN_SCENES)
    assert first["test"] == list(TEST_SCENES)


def test_create_splits_never_breaks_a_scene():
    splits = create_splits(TRAIN_SCENES + TEST_SCENES)
    train, test = set(splits["train"]), set(splits["test"])

    assert train.isdisjoint(test)
    assert train | test == set(TRAIN_SCENES + TEST_SCENES)


def test_scene_id_and_camera():
    assert scene_id("MOT17-11-FRCNN") == "11"
    assert CAMERA["09"] == "static" and CAMERA["11"] == "moving"


def test_median_occlusion_from_visibility_runs():
    # identidade 1: 3 quadros ocluídos, depois 2. Mediana 2.5.
    tracks = [(t, 1, 0, 0, 10, 10, 1, 1, 0.1 if t < 3 or t >= 5 else 1.0)
              for t in range(7)]
    tracks += [(t, 2, 0, 20, 10, 10, 1, 1, 1.0) for t in range(7)]

    assert median_occlusion(tracks) == 2.5


def main():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_read_mot_file_types(Path(tmp))
        test_read_det_file_without_class(Path(tmp))

    for test in (test_split_ground_truth_keeps_only_real_pedestrians,
                 test_create_splits_is_fixed_and_ignores_seed,
                 test_create_splits_never_breaks_a_scene,
                 test_scene_id_and_camera,
                 test_median_occlusion_from_visibility_runs):
        test()
        print(f"ok  {test.__name__}")

    print("ok  test_read_mot_file_types")
    print("ok  test_read_det_file_without_class")


if __name__ == "__main__":
    main()
