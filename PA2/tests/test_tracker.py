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


def _appearance_tracker(**kwargs):
    base = dict(iou_threshold=0.3, max_age=5, min_hits=1, matcher="hungarian",
                motion=FakeAppearance(), appearance_threshold=0.5)
    base.update(kwargs)
    return Tracker(**base)


def test_cosine_match_inside_the_gate_keeps_the_id():
    # Depois de 2 misses a pessoa volta deslocada 8 px: IoU com a última
    # caixa (0,11) não passa do limiar 0,3, mas cabe no portão que cresceu
    # com os misses — o cosseno mantém o id.
    detections = [(1, 0.0, 0.0, 10.0, 10.0, 1.0),
                  (4, 8.0, 0.0, 10.0, 10.0, 1.0)]
    embeddings = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]

    tracks = _appearance_tracker().run(detections, embeddings=embeddings)

    assert len({row[1] for row in tracks}) == 1


def test_cosine_match_needs_the_geometric_gate():
    # O enunciado pede cosseno COM portão de IoU por cima: o mesmo
    # embedding reaparecendo a 80 px (fora do portão) é outra track.
    detections = [(1, 0.0, 0.0, 10.0, 10.0, 1.0),
                  (4, 80.0, 0.0, 10.0, 10.0, 1.0)]
    embeddings = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]

    tracks = _appearance_tracker().run(detections, embeddings=embeddings)

    assert len({row[1] for row in tracks}) == 2


def test_gate_grows_with_the_misses():
    # O mesmo deslocamento de 20 px: bloqueado depois de 1 miss (portão de
    # 16 px, de -3 a 13), aceito depois de 4 (34 px, de -12 a 22). Portão
    # fixado aqui para o teste não depender dos defaults.
    embeddings = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    soon = [(1, 0.0, 0.0, 10.0, 10.0, 1.0), (3, 20.0, 0.0, 10.0, 10.0, 1.0)]
    late = [(1, 0.0, 0.0, 10.0, 10.0, 1.0), (6, 20.0, 0.0, 10.0, 10.0, 1.0)]

    ids_soon = {row[1] for row in _appearance_tracker(gate_growth=0.6, gate_iou=0.01).run(soon, embeddings=embeddings)}
    ids_late = {row[1] for row in _appearance_tracker(gate_growth=0.6, gate_iou=0.01).run(late, embeddings=embeddings)}

    assert len(ids_soon) == 2
    assert len(ids_late) == 1


def test_miss_prefer_iou_is_the_default():
    # A cascata (IoU da última caixa nas tracks perdidas antes do cosseno)
    # é o padrão da Trilha B; o antigo comportamento é opt-out.
    assert Tracker().miss_prefer_iou is True


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
    # trocados. Sem a correção só o cosseno decide e, com o portão
    # geométrico impedindo a troca através dos 80 px, a pessoa perde o id;
    # com miss_prefer_iou o IoU da última caixa o mantém.
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
                         motion=FakeAppearance(), appearance_threshold=0.5,
                         miss_prefer_iou=False).run(detections, embeddings)
    fix_tracks = Tracker(iou_threshold=0.3, max_age=5, min_hits=1,
                         motion=FakeAppearance(), appearance_threshold=0.5,
                         miss_prefer_iou=True).run(detections, embeddings)
    first = {row[2:4]: row[1] for row in app_tracks if row[0] == 1}
    later_app = {row[2:4]: row[1] for row in app_tracks if row[0] == 3}
    later_fix = {row[2:4]: row[1] for row in fix_tracks if row[0] == 3}
    assert later_app.get((1.0, 0.0)) != first[(0.0, 0.0)]
    assert later_app.get((81.0, 0.0)) != first[(0.0, 0.0)]
    assert first[(0.0, 0.0)] == later_fix.get((1.0, 0.0))


def test_miss_resets_consecutive_hits_before_confirmation():
    tracker = Tracker(min_hits=3, max_age=5, iou_threshold=0.3)
    detections = boxes_at([1, 2, 4, 5, 6], 0.0, 0.0)

    tracks = tracker.run(detections)

    assert [row[0] for row in tracks] == [6]


class RecordingFusion:
    """
    Braço 'both' de mentira: guarda o que recebe em cada passo.
    """

    kind = "both"
    cell = "gru"
    embed_dim = 2
    has_query = False
    parameterization = "absolute"

    def __init__(self):
        self.inputs = []

    def init_state(self, batch=1, device="cpu"):
        return None

    def step(self, features, state):
        features = torch.as_tensor(features, dtype=torch.float32)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        self.inputs.append(features.clone())
        vector = torch.tensor([[0.0, -1.0]]).repeat(len(features), 1)  # "saída" da rede
        return (vector, features[:, 2:6]), state


def test_both_arm_rolls_forward_with_the_last_observed_crop():
    # No miss o braço 'both' roda para frente sem observação: a aparência
    # que entra é o último recorte VISTO, não o vetor de saída da própria
    # rede (que ele nunca viu como entrada no treino).
    model = RecordingFusion()
    tracker = Tracker(iou_threshold=0.3, max_age=5, min_hits=1, motion=model,
                      image_size=(100.0, 100.0))
    detections = [(1, 10.0, 10.0, 10.0, 10.0, 1.0)]
    embeddings = [np.array([1.0, 0.0])]

    tracker.run(detections + [(3, 80.0, 80.0, 10.0, 10.0, 1.0)],
                embeddings=embeddings + [np.array([0.0, 1.0])])

    miss_step = model.inputs[1]              # quadro 2: a track 1 não casou
    assert torch.allclose(miss_step[0, :2], torch.tensor([1.0, 0.0]))



def test_prefer_confirmed_keeps_the_real_track_over_a_newborn_duplicate():
    # Diagnóstico da Parte 4: a track confirmada perde a detecção por um
    # quadro (IoU 0,11 < 0,2), a detecção sobrante vira uma track nova, e
    # no quadro seguinte a recém-nascida — "vista no quadro anterior" —
    # casa antes da confirmada, que só concorre como perdida. Com
    # prefer_confirmed, as confirmadas perdidas casam por IoU antes das
    # tentativas.
    detections = [(t, 0.0, 0.0, 10.0, 10.0, 1.0) for t in (1, 2, 3)]
    detections += [(4, 8.0, 0.0, 10.0, 10.0, 1.0),    # a track 1 não alcança
                   (5, 3.0, 0.0, 10.0, 10.0, 1.0)]    # IoU 0,54 com a 1, 0,33 com a nova
    embeddings = [np.array([1.0, 0.0])] * len(detections)

    def id_at_frame_5(**flags):
        tracker = Tracker(iou_threshold=0.2, max_age=5, min_hits=2,
                          motion=FakeAppearance(), appearance_threshold=0.5,
                          **flags)
        rows = tracker.run(detections, embeddings=embeddings)
        first_id = next(row[1] for row in rows if row[0] == 2)
        return first_id, next(row[1] for row in rows if row[0] == 5)

    first, before = id_at_frame_5()
    assert before != first                       # a duplicata ganhou

    first, after = id_at_frame_5(prefer_confirmed=True)
    assert after == first                        # a confirmada recupera


def test_prefer_confirmed_is_off_by_default():
    assert Tracker().prefer_confirmed is False


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
