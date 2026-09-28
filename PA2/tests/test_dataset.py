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
    TrackWindowDataset,
    create_splits,
    load_sequence,
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
        (1, 1, 0, 0, 10, 10, 1, 1, 1.0),    # pedestre
        (1, 2, 0, 0, 10, 10, 0, 1, 1.0),    # pedestre com conf=0: fora do gt
        (1, 3, 0, 0, 10, 10, 1, 7, 1.0),    # pessoa estática
        (1, 4, 0, 0, 10, 10, 1, 8, 0.2),    # distractor
    ]

    pedestrians, distractors = split_ground_truth(rows)

    assert [row[1] for row in pedestrians] == [1]
    assert [row[1] for row in distractors] == [3, 4]


def test_distractors_are_only_the_official_classes():
    # TrackEval / MOT17: só 2 (pessoa em veículo), 7 (pessoa estática),
    # 8 (distractor) e 12 (reflexo) são distractores. Carro (3), bicicleta
    # (4), moto (5), veículo (6) e oclusores (9, 10, 11) só ficam de fora do
    # gt: predição em cima deles é FP, não some.
    rows = [(1, k, 0, 0, 10, 10, 0, k, 1.0) for k in range(2, 13)]

    _, distractors = split_ground_truth(rows)

    assert sorted(row[7] for row in distractors) == [2, 7, 8, 12]


def test_load_sequence_exposes_every_gt_row():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "train" / "MOT17-02-SDP"
        (folder / "gt").mkdir(parents=True)
        (folder / "seqinfo.ini").write_text(
            "[Sequence]\nname=MOT17-02-SDP\nimDir=img1\nframeRate=30\n"
            "seqLength=2\nimWidth=100\nimHeight=100\nimExt=.jpg\n")
        (folder / "gt" / "gt.txt").write_text(
            "1,1,0,0,10,10,1,1,1\n1,2,50,0,10,10,0,3,1\n2,1,1,0,10,10,1,1,1\n")

        info, gt, _ = load_sequence("02", root=tmp)

    assert len(gt) == 2
    assert len(info["gt_all"]) == 3          # o carro também, para o matching
    assert "distractors" not in info         # quem usar o nome velho quebra alto


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


def test_track_window_dataset_length_and_no_test_scene(tmp_path):
    root = tmp_path / "MOT17" / "train" / "MOT17-02-SDP"
    (root / "gt").mkdir(parents=True)
    (root / "det").mkdir()
    (root / "seqinfo.ini").write_text(
        "name=MOT17-02-SDP\nframeRate=30\nseqLength=8\n"
        "imWidth=64\nimHeight=64\nimDir=img1\nimExt=.jpg\n")
    lines = []
    for t in range(1, 9):
        lines.append(f"{t},1,0,0,10,10,1,1,1.0")
        lines.append(f"{t},2,20,0,10,10,1,1,1.0")
    (root / "gt" / "gt.txt").write_text("\n".join(lines) + "\n")
    (root / "det" / "det.txt").write_text("1,-1,0,0,10,10,1\n")

    dataset = TrackWindowDataset(["02"], window=4, stride=2, root=tmp_path / "MOT17")

    assert len(dataset) == 3
    assert dataset[0]["window"] == 4
    assert set(dataset[0]["identities"]) == {1, 2}

    try:
        TrackWindowDataset(["09"], window=4, root=tmp_path / "MOT17")
    except ValueError:
        return
    raise AssertionError("teste não pode entrar no dataset de janelas")


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
        test_track_window_dataset_length_and_no_test_scene(Path(tmp))
        print("ok  test_read_mot_file_types")
        print("ok  test_read_det_file_without_class")
        print("ok  test_track_window_dataset_length_and_no_test_scene")

    for test in (test_split_ground_truth_keeps_only_real_pedestrians,
                 test_create_splits_is_fixed_and_ignores_seed,
                 test_create_splits_never_breaks_a_scene,
                 test_scene_id_and_camera,
                 test_median_occlusion_from_visibility_runs):
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
