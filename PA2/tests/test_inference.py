"""
Entregável inferencia.ipynb — src/inference.py.

Recebe o caminho de uma sequência qualquer, devolve as trajetórias, a
contagem de objetos únicos e os quadros com as identidades coloridas, sem
retreinar. Os testes montam uma sequência mínima em disco (quadros gerados +
det.txt), então não dependem do MOT17 nem de rede.

Roda sob pytest (usa tmp_path e monkeypatch).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import src.inference as inference
from src.inference import (
    color_for,
    draw_tracks,
    resolve_checkpoint,
    track_sequence,
    write_video,
)


def make_sequence(root, n_frames=5, with_seqinfo=True, with_det=True):
    """
    Dois quadrados andando 2 px por quadro, bem separados. O det.txt tem
    exatamente as duas caixas em cada quadro.
    """

    folder = Path(root) / "minha_sequencia"
    (folder / "img1").mkdir(parents=True)
    rows = []
    for frame in range(1, n_frames + 1):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        for k, x0 in enumerate((10, 60)):
            x = x0 + 2 * (frame - 1)
            image[20:40, x:x + 10] = (200, 50 + 100 * k, 50)
            rows.append(f"{frame},-1,{x},20,10,20,0.9")
        Image.fromarray(image).save(folder / "img1" / f"{frame:06d}.jpg")
    if with_det:
        (folder / "det").mkdir()
        (folder / "det" / "det.txt").write_text("\n".join(rows) + "\n")
    if with_seqinfo:
        (folder / "seqinfo.ini").write_text(
            "[Sequence]\nname=minha_sequencia\nimDir=img1\nframeRate=15\n"
            f"seqLength={n_frames}\nimWidth=100\nimHeight=60\nimExt=.jpg\n")
    return folder


def test_baseline_tracks_an_arbitrary_folder(tmp_path):
    folder = make_sequence(tmp_path)

    result = track_sequence(folder, checkpoint=False, device="cpu")

    assert result["count"] == 2
    assert result["fps"] == 15
    assert {row[1] for row in result["tracks"]} == {1, 2}
    # min_hits=2: o primeiro quadro de cada track não sai.
    assert min(row[0] for row in result["tracks"]) == 2


def test_frames_are_lazy_and_annotated(tmp_path):
    folder = make_sequence(tmp_path)

    result = track_sequence(folder, checkpoint=False, device="cpu")
    frames = result["frames"]

    assert len(frames) == 5
    plain = np.asarray(Image.open(folder / "img1" / "000003.jpg").convert("RGB"))
    drawn = frames[2]                       # quadro 3
    assert drawn.shape == plain.shape
    assert not np.array_equal(drawn, plain)  # tem caixa desenhada


def test_folder_without_seqinfo_still_works(tmp_path):
    folder = make_sequence(tmp_path, with_seqinfo=False)

    result = track_sequence(folder, checkpoint=False, device="cpu")

    assert result["count"] == 2
    assert result["fps"] == 30              # sem seqinfo.ini: padrão


def test_without_det_txt_falls_back_to_torchvision(tmp_path, monkeypatch):
    folder = make_sequence(tmp_path, with_det=False)
    calls = []

    def fake_detector(frames, score_threshold=0.5, device="cpu",
                      nms_threshold=0.5, model=None, first_frame=1):
        calls.append(len(frames))
        return [(i, 10.0, 20.0, 10.0, 20.0, 0.9)
                for i in range(first_frame, first_frame + len(frames))]

    monkeypatch.setattr(inference, "torchvision_detections", fake_detector)
    monkeypatch.setattr(inference, "load_torchvision_detector",
                        lambda device="cpu": None)

    result = track_sequence(folder, checkpoint=False, device="cpu")

    assert calls                            # o detector rodou
    assert result["detector"] == "torchvision"
    assert result["count"] == 1



def test_torchvision_fallback_uses_the_threshold_chosen_on_train(tmp_path, monkeypatch):
    # O Faster R-CNN entra no rastreamento com o limiar varrido no treino
    # (0,8), o mesmo dos resultados baseline_torchvision_*. A pessoa com
    # score 0,6 não pode virar track.
    from src.detection import default_score_threshold

    folder = make_sequence(tmp_path, with_det=False)

    def fake_detector(frames, score_threshold=0.5, device="cpu",
                      nms_threshold=0.5, model=None, first_frame=1):
        rows = []
        for i in range(first_frame, first_frame + len(frames)):
            rows.append((i, 10.0, 20.0, 10.0, 20.0, 0.9))
            rows.append((i, 60.0, 20.0, 10.0, 20.0, 0.6))
        return [row for row in rows if row[5] >= score_threshold]

    monkeypatch.setattr(inference, "torchvision_detections", fake_detector)
    monkeypatch.setattr(inference, "load_torchvision_detector",
                        lambda device="cpu": None)

    result = track_sequence(folder, checkpoint=False, device="cpu")

    assert default_score_threshold("torchvision") == 0.8
    assert result["count"] == 1
    assert all(row[2] == 10.0 for row in result["tracks"])

def test_colors_are_stable_and_distinct():
    assert color_for(7) == color_for(7)
    assert len({color_for(i) for i in range(1, 21)}) == 20


def test_draw_tracks_does_not_touch_the_input():
    frame = np.zeros((30, 30, 3), dtype=np.uint8)

    out = draw_tracks(frame, [(5.0, 5.0, 10.0, 10.0)], [3])

    assert frame.sum() == 0
    assert out.sum() > 0


def test_write_video_creates_a_file(tmp_path):
    frames = [np.full((32, 48, 3), 40 * i, dtype=np.uint8) for i in range(4)]

    path = write_video(frames, tmp_path / "out" / "video.mp4", fps=10)

    assert Path(path).exists() and Path(path).stat().st_size > 0


def test_resolve_checkpoint_survives_a_moved_repo(tmp_path):
    # best_model.json gravado noutra máquina, com caminho absoluto velho: o
    # nome do arquivo em checkpoints/ ainda resolve.
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "temporal_best.pt").write_bytes(b"x")
    best = tmp_path / "best_model.json"
    best.write_text(json.dumps(
        {"checkpoint": "/outra/maquina/PA2/checkpoints/temporal_best.pt"}))

    path = resolve_checkpoint(None, best_model=best, root=tmp_path)

    assert path == ckpt_dir / "temporal_best.pt"


def test_resolve_checkpoint_accepts_a_path_relative_to_the_project(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "m.pt").write_bytes(b"x")
    best = tmp_path / "best_model.json"
    best.write_text(json.dumps({"checkpoint": "checkpoints/m.pt"}))

    assert resolve_checkpoint(None, best_model=best, root=tmp_path) == \
        tmp_path / "checkpoints" / "m.pt"


def test_windowed_inference_is_a_discussion_not_code():
    try:
        inference.windowed_inference("qualquer")
    except NotImplementedError as error:
        assert "Parte 2" in str(error)
        return

    raise AssertionError("windowed_inference deveria explicar que é discussão")
