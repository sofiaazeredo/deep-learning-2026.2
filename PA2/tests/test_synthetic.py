"""
Parte 0.1 — testes do gerador sintético.

O que o enunciado exige do gerador, verificado com a resposta conhecida:
  - vídeos T x 128 x 128 com o número pedido de elipses;
  - determinismo por seed (os mesmos vídeos em todas as execuções);
  - oclusão de verdade: com occlusion_frames=N, uma identidade some do
    ground truth por N quadros e volta com o mesmo id;
  - as funções auxiliares (ordem de profundidade, duração dos buracos).

Roda com "python tests/test_synthetic.py" ou sob pytest.
"""

import sys
from pathlib import Path

# Rodar "python tests/x.py" coloca tests/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.synthetic import make_sequence, depth_order, occlusion_durations


def test_frames_shape_and_range():
    frames, _ = make_sequence(n_frames=40, size=128, seed=0)

    assert frames.shape == (40, 128, 128)
    assert frames.dtype == np.float32
    assert frames.min() >= 0.0 and frames.max() <= 1.0


def test_number_of_identities():
    _, tracks = make_sequence(n_objects=11, occlusion_frames=0, seed=1)

    assert len({row[1] for row in tracks}) == 11


def test_same_seed_same_sequence():
    frames_a, tracks_a = make_sequence(seed=7)
    frames_b, tracks_b = make_sequence(seed=7)

    assert np.array_equal(frames_a, frames_b)
    assert tracks_a == tracks_b


def test_boxes_inside_frame_and_visibility_in_range():
    size = 128
    _, tracks = make_sequence(n_objects=15, speed=4.0, size=size, seed=3)

    for frame, _, x, y, w, h, visibility in tracks:
        assert x >= 0 and y >= 0
        assert x + w <= size and y + h <= size
        assert 0.0 < visibility <= 1.0


def test_occluded_identity_disappears_and_returns():
    # Buraco medido pelo ground truth: pode passar de N por no máximo um
    # quadro de cada lado (a entrada/saída de trás do oclusor).
    for n in (5, 10, 20):
        _, tracks = make_sequence(n_frames=60, occlusion_frames=n, seed=n)

        holes = occlusion_durations(tracks)
        longest = max(max(h, default=0) for h in holes.values())

        assert n <= longest <= n + 2, (n, longest)


def test_hidden_object_leaves_no_pixels():
    # Sem ruído, cada objeto é desenhado com uma intensidade própria. Lida
    # num quadro em que ele está 100% visível, essa intensidade não pode
    # aparecer em nenhum pixel dos quadros em que ele está escondido.
    frames, tracks = make_sequence(n_frames=60, occlusion_frames=10,
                                   noise=0.0, seed=4)

    holes = occlusion_durations(tracks)
    hidden_id = max(holes, key=lambda i: max(holes[i], default=0))

    rows = [row for row in tracks if row[1] == hidden_id]
    t, _, x, y, w, h, _ = next(row for row in rows if row[6] == 1.0)
    intensity = frames[t, int(y + h / 2), int(x + w / 2)]

    present = {row[0] for row in rows}
    hidden = [t for t in range(min(present), max(present))
              if t not in present]

    assert hidden
    for t in hidden:
        assert not np.any(frames[t] == intensity), t


def test_occlusion_requires_motion():
    try:
        make_sequence(speed=0.0, occlusion_frames=10, seed=0)
    except ValueError:
        return

    raise AssertionError("speed=0 com oclusão deveria falhar")


def test_occlusion_longer_than_video_fails():
    try:
        make_sequence(n_frames=30, occlusion_frames=30, seed=0)
    except ValueError:
        return

    raise AssertionError("oclusão do tamanho do vídeo deveria falhar")


def test_depth_order_back_to_front():
    objects = [{"depth": 2.0}, {"depth": 0.0}, {"depth": 1.0}]

    assert depth_order(objects) == [1, 2, 0]


def test_occlusion_durations_counts_interior_gaps():
    tracks = [
        (0, 1, 0, 0, 1, 1, 1.0),
        (1, 1, 0, 0, 1, 1, 1.0),
        (4, 1, 0, 0, 1, 1, 1.0),   # buraco de 2 quadros (2 e 3)
        (5, 1, 0, 0, 1, 1, 1.0),
        (7, 1, 0, 0, 1, 1, 1.0),   # buraco de 1 quadro (6)
        (0, 2, 0, 0, 1, 1, 1.0),
        (1, 2, 0, 0, 1, 1, 1.0),
    ]

    assert occlusion_durations(tracks) == {1: [2, 1], 2: []}


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
