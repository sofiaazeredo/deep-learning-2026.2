"""
Parte 0.3 — testes da métrica.

Três casos construídos à mão, com a resposta conhecida:
  (a) predição = ground truth          -> IDF1 = 1 e zero switches;
  (b) duas identidades trocadas no quadro k -> o número exato de switches;
  (c) uma track partida em duas no meio -> o efeito em IDF1, que NÃO é o
      mesmo de (b) — partir não troca identidade, fragmenta.

E os testes das peças: IoU, casamento por quadro, a regra de continuidade do
CLEAR MOT, buracos do ground truth, contagem de identidades e MOTA.

Roda com "python tests/test_metrics.py" ou sob pytest.
"""

import sys
from pathlib import Path

# Rodar "python tests/x.py" coloca tests/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import numpy as np

from src.metrics import (
    iou_matrix,
    match_frame,
    idf1,
    id_switches,
    fragmentations,
    mota,
    unique_id_count_error,
    evaluate_sequence,
)

T = 20   # quadros dos casos à mão
K = 8    # quadro do evento (troca ou quebra)


def track(track_id, frames, y):
    """
    Uma trajetória 10x10 andando 1 px por quadro em x, na altura y.
    Linhas (frame, id, x, y, w, h). Com y diferentes as trajetórias nunca
    se sobrepõem.
    """

    return [(t, track_id, float(t), float(y), 10.0, 10.0) for t in frames]


def relabel(rows, new_id):
    return [(t, new_id, *rest) for t, _, *rest in rows]


GT = track(1, range(T), y=0) + track(2, range(T), y=50)


# --- os três casos à mão -------------------------------------------------


def test_perfect_prediction():
    score, idp, idr, _ = idf1(GT, GT)

    assert score == 1.0 and idp == 1.0 and idr == 1.0
    assert id_switches(GT, GT) == 0
    assert fragmentations(GT, GT) == 0


def test_swapped_identities():
    # A partir do quadro K a predição troca os rótulos das duas trajetórias:
    # cada identidade verdadeira muda de id previsto uma vez -> 2 switches.
    first, second = track(1, range(T), y=0), track(2, range(T), y=50)
    pred = (relabel(first[:K], 10) + relabel(first[K:], 20)
            + relabel(second[:K], 20) + relabel(second[K:], 10))

    score, _, _, details = idf1(GT, pred)

    assert id_switches(GT, pred) == 2
    assert fragmentations(GT, pred) == 0
    # A atribuição global fica com o trecho mais longo (T - K = 12 quadros)
    # de cada identidade: IDF1 = 2*24 / (2*24 + 16 + 16) = 12/20.
    assert details["idtp"] == 2 * (T - K)
    assert np.isclose(score, (T - K) / T)


def test_broken_track():
    # A identidade 1 é partida em duas no quadro K (id 10 e depois 30), sem
    # buraco; a identidade 2 sai perfeita. Um switch só, e o IDF1 só perde o
    # trecho curto de uma identidade: (20 + 12) / 40 — maior que o de (b).
    first = track(1, range(T), y=0)
    pred = (relabel(first[:K], 10) + relabel(first[K:], 30)
            + relabel(track(2, range(T), y=50), 20))

    score, _, _, _ = idf1(GT, pred)

    assert id_switches(GT, pred) == 1
    assert fragmentations(GT, pred) == 0
    assert np.isclose(score, (T + T - K) / (2 * T))
    assert score > (T - K) / T                     # não é o mesmo de (b)


def test_broken_track_with_gap():
    # A mesma quebra, mas a track morre por 2 quadros antes de renascer com
    # outro id: agora também é uma fragmentação.
    first = track(1, range(T), y=0)
    pred = (relabel(first[:K], 10) + relabel(first[K + 2:], 30)
            + relabel(track(2, range(T), y=50), 20))

    score, _, _, details = idf1(GT, pred)

    assert id_switches(GT, pred) == 1
    assert fragmentations(GT, pred) == 1
    # IDTP = 10 (id 30 na identidade 1) + 20; predições: 8 + 10 + 20 = 38.
    assert details["idtp"] == 30
    assert np.isclose(score, 60 / (40 + 38))


# --- IoU e casamento por quadro ------------------------------------------


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [50, 50, 10, 10]],
                 dtype=float)

    iou = iou_matrix(a, b)

    assert iou.shape == (2, 3)
    assert np.allclose(iou[0], [1.0, 50 / 150, 0.0])


def test_iou_empty():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_match_frame_one_to_one_above_threshold():
    gt = np.array([[0, 0, 10, 10], [50, 0, 10, 10]], dtype=float)
    pred = np.array([[1, 0, 10, 10], [0, 0, 10, 10], [80, 0, 10, 10]],
                    dtype=float)

    pairs = match_frame(gt, pred, threshold=0.5)

    assert pairs == [(0, 1)]   # a melhor para o gt 0; o gt 1 não tem par


def test_match_frame_maximizes_total_iou():
    # Guloso pegaria (gt0, pred0) = 0.9 primeiro e deixaria o gt1 sem par;
    # o ótimo casa os dois.
    iou_target = np.array([[0.9, 0.6], [0.7, 0.0]])
    gt = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)

    pairs = match_frame(gt, gt, threshold=0.5, iou=iou_target)

    assert sorted(pairs) == [(0, 1), (1, 0)]


# --- a regra de continuidade e os buracos --------------------------------


def test_crossing_tracks_keep_previous_match():
    # Duas identidades se cruzando: no quadro 1 cada predição fica um pouco
    # mais perto da OUTRA identidade, mas ainda com IoU >= 0.5 com a sua.
    # O Hungarian puro trocaria (2 switches); o CLEAR MOT mantém o par do
    # quadro anterior enquanto ele continua válido -> 0 switches.
    gt = [(0, 1, 0.0, 0, 10, 10), (0, 2, 2.0, 0, 10, 10),
          (1, 1, 0.0, 0, 10, 10), (1, 2, 2.0, 0, 10, 10)]
    pred = [(0, 10, 0.0, 0, 10, 10), (0, 20, 2.0, 0, 10, 10),
            (1, 10, 1.2, 0, 10, 10), (1, 20, 0.8, 0, 10, 10)]

    assert id_switches(gt, pred) == 0


def test_occlusion_in_ground_truth_is_not_a_fragment():
    # O objeto sai do ground truth (ocluído) e a predição também some:
    # nada foi perdido, nada fragmentou.
    frames = [t for t in range(T) if t not in (8, 9)]
    gt = track(1, frames, y=0)

    assert fragmentations(gt, relabel(gt, 10)) == 0
    assert id_switches(gt, relabel(gt, 10)) == 0
    assert idf1(gt, relabel(gt, 10))[0] == 1.0


def test_same_id_after_gap_is_fragment_not_switch():
    gt = track(1, range(T), y=0)
    pred = relabel([row for row in gt if row[0] not in (8, 9)], 10)

    assert id_switches(gt, pred) == 0
    assert fragmentations(gt, pred) == 1


def test_unmatched_start_is_not_a_fragment():
    # A track só nasce no quadro 5: é perda (IDFN), não interrupção.
    gt = track(1, range(T), y=0)
    pred = relabel([row for row in gt if row[0] >= 5], 10)

    assert fragmentations(gt, pred) == 0


# --- IDF1 nas bordas e contagem ------------------------------------------


def test_idf1_empty_prediction_is_zero():
    score, idp, idr, details = idf1(GT, [])

    assert score == 0.0 and idr == 0.0
    assert details["idfn"] == len(GT)


def test_idf1_nothing_to_track_is_one():
    assert idf1([], [])[0] == 1.0


def test_idf1_threshold_is_inclusive():
    # IoU exatamente 0.5 conta como casamento com threshold=0.5.
    gt = [(0, 1, 0.0, 0.0, 30.0, 10.0)]
    pred = [(0, 10, 10.0, 0.0, 30.0, 10.0)]   # inter 200 / união 400

    assert idf1(gt, pred, threshold=0.5)[0] == 1.0


def test_duplicate_id_in_frame_fails():
    pred = GT + [GT[0]]

    try:
        idf1(GT, pred)
    except ValueError:
        return

    raise AssertionError("id repetido no mesmo quadro deveria falhar")


def test_unique_id_count_error():
    pred = relabel(GT[:5], 10) + relabel(GT[5:], 11)   # 2 ids previstos
    pred += track(12, range(3), y=90)                  # + 1 id espúrio

    assert unique_id_count_error(GT, pred) == 1


# --- MOTA ------------------------------------------------------------------


def test_mota_perfect_prediction():
    score, details = mota(GT, GT)

    assert score == 1.0
    assert details == {"tp": 40, "fp": 0, "fn": 0, "id_switches": 0,
                       "n_gt": 40}


def test_mota_swap_costs_only_the_switches():
    # Caso (b): todas as caixas casam, só os 2 switches pesam.
    first, second = track(1, range(T), y=0), track(2, range(T), y=50)
    pred = (relabel(first[:K], 10) + relabel(first[K:], 20)
            + relabel(second[:K], 20) + relabel(second[K:], 10))

    assert np.isclose(mota(GT, pred)[0], 1 - 2 / 40)


def test_mota_broken_track_with_gap():
    # Caso (c'): 2 caixas perdidas no buraco + 1 switch.
    first = track(1, range(T), y=0)
    pred = (relabel(first[:K], 10) + relabel(first[K + 2:], 30)
            + relabel(track(2, range(T), y=50), 20))

    score, details = mota(GT, pred)

    assert details["fn"] == 2 and details["fp"] == 0
    assert details["id_switches"] == 1
    assert np.isclose(score, 1 - 3 / 40)


def test_mota_can_be_negative():
    # 50 caixas espúrias longe de tudo contra 40 verdadeiras.
    spurious = [(t % T, 99 + t // T, 200.0, 200.0, 10.0, 10.0)
                for t in range(50)]

    score, details = mota(GT, GT + spurious)

    assert details["fp"] == 50
    assert np.isclose(score, 1 - 50 / 40)


def test_mota_counts_prediction_in_frame_without_ground_truth():
    pred = GT + [(100, 10, 0.0, 0.0, 10.0, 10.0)]

    assert mota(GT, pred)[1]["fp"] == 1


def test_mota_empty_prediction_is_zero():
    assert mota(GT, [])[0] == 0.0


def test_mota_without_ground_truth_is_undefined():
    assert math.isnan(mota([], GT)[0])


def test_evaluate_sequence_reports_everything():
    first = track(1, range(T), y=0)
    pred = (relabel(first[:K], 10) + relabel(first[K:], 30)
            + relabel(track(2, range(T), y=50), 20))

    result = evaluate_sequence(GT, pred)

    assert result["id_switches"] == 1
    assert result["fragmentations"] == 0
    assert result["n_gt_ids"] == 2 and result["n_pred_ids"] == 3
    assert result["id_count_error"] == 1
    assert np.isclose(result["id_ratio"], 1.5)
    assert np.isclose(result["id_switches_per_gt"], 0.5)
    assert np.isclose(result["idf1"], idf1(GT, pred)[0])
    assert np.isclose(result["mota"], mota(GT, pred)[0])
    assert (result["tp"], result["fp"], result["fn"]) == (40, 0, 0)


def test_accepts_extra_columns():
    # gt.txt do MOT17 tem 9 colunas, o gerador sintético 7: só as 6
    # primeiras importam.
    gt = [(*row, 1.0, 1, 1.0) for row in GT]

    assert idf1(gt, GT)[0] == 1.0


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
