"""
Métricas de rastreamento implementadas do zero (entregável obrigatório).

Proibido usar motmetrics / TrackEval / py-motmetrics. Do PA1 vem a ideia do
casamento um-para-um por IoU com limiar; aqui ele é Hungarian (o do PA1 era
guloso) e ganha a regra de continuidade do CLEAR MOT.

IDF1 exige uma atribuição global um-para-um entre identidades previstas e
verdadeiras ao longo da sequência inteira — não é um casamento por quadro.

Entrada de todas as funções: linhas (frame, id, x, y, w, h, ...). Só as seis
primeiras colunas são lidas, então servem o gt.txt do MOT17 (9 colunas), o
ground truth do gerador sintético (7) e a saída do tracker (6).

Definições (as mesmas do CLEAR MOT e do IDF1 da literatura):

  casamento por quadro   CLEAR MOT (Bernardin & Stiefelhagen, 2008): o par
                         (gt, predição) casado antes continua casado se o IoU
                         ainda passa do limiar; o resto vai para o Hungarian,
                         maximizando o IoU total. Sem a continuidade, duas
                         trajetórias que se cruzam trocariam de par só porque
                         o Hungarian achou uma soma de IoU um pouco maior.
  ID switch              uma identidade verdadeira casa com um id previsto
                         diferente do último com que ela casou (a memória
                         atravessa buracos).
  fragmentação           a identidade verdadeira estava casada, deixa de estar
                         num quadro em que ELA ESTÁ no ground truth, e volta
                         a casar depois. Quadro em que ela não está no gt
                         (ocluída) não interrompe nada.
  IDF1                   Ristani et al. (2016): um par (gt, predição) conta
                         num quadro se o IoU >= limiar; o Hungarian sobre
                         essas contagens dá a atribuição global. IDTP é a
                         soma atribuída, IDFP = caixas previstas - IDTP,
                         IDFN = caixas verdadeiras - IDTP.
  MOTA                   1 - (FN + FP + IDSW) / caixas verdadeiras, sobre o
                         mesmo casamento do CLEAR MOT. Opcional no enunciado:
                         com o detector congelado FP e FN quase não mudam
                         entre configurações, então ela esconde o que muda.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _by_frame(tracks):
    """
    frame -> (ids, caixas (N, 4)). Falha se um id aparece duas vezes no
    mesmo quadro — isso é bug de quem gerou as trajetórias, e contaria em
    dobro no IDF1.
    """

    rows = {}

    for row in tracks:
        rows.setdefault(int(row[0]), []).append(row)

    frames = {}

    for frame, frame_rows in rows.items():
        ids = [int(row[1]) for row in frame_rows]

        if len(set(ids)) != len(ids):
            raise ValueError(f"id repetido no quadro {frame}")

        boxes = np.array([row[2:6] for row in frame_rows], dtype=np.float64)
        frames[frame] = (ids, boxes)

    return frames


def iou_matrix(boxes_a, boxes_b):
    """
    IoU par a par entre dois conjuntos de caixas (x, y, w, h).
    """

    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)

    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]

    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h

    union = a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter

    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_frame(gt_boxes, pred_boxes, threshold=0.5, iou=None):
    """
    Casamento um-para-um dentro de um quadro por Hungarian, maximizando o IoU
    total entre os pares com IoU >= threshold. Base de IDF1 e da contagem de
    switches.

    `iou` aceita a matriz já calculada (gt x pred). Devolve [(i_gt, i_pred)].
    """

    if iou is None:
        iou = iou_matrix(gt_boxes, pred_boxes)

    if iou.size == 0:
        return []

    valid = iou >= threshold

    # Par abaixo do limiar ganha custo proibitivo; o Hungarian ainda pode
    # escolhê-lo por falta de opção, então ele é filtrado depois.
    cost = np.where(valid, 1.0 - iou, 1e6)
    rows, cols = linear_sum_assignment(cost)

    return [(int(i), int(j)) for i, j in zip(rows, cols) if valid[i, j]]


def _clear_mot_matches(gt_tracks, pred_tracks, threshold):
    """
    Casamentos quadro a quadro com a regra de continuidade do CLEAR MOT.

    Devolve (gt_frames, matches): gt_frames[gt_id] é a lista ordenada de
    quadros em que a identidade está no ground truth; matches[frame] é
    {gt_id: pred_id}.
    """

    gt = _by_frame(gt_tracks)
    pred = _by_frame(pred_tracks)

    gt_frames = {}
    matches = {}
    last = {}                      # gt_id -> último pred_id casado

    for frame in sorted(gt):
        gt_ids, gt_boxes = gt[frame]

        for gt_id in gt_ids:
            gt_frames.setdefault(gt_id, []).append(frame)

        pred_ids, pred_boxes = pred.get(frame, ([], np.zeros((0, 4))))
        iou = iou_matrix(gt_boxes, pred_boxes)

        pred_index = {pred_id: j for j, pred_id in enumerate(pred_ids)}
        frame_matches = {}
        used_gt, used_pred = set(), set()

        # 1) continuidade: o par anterior segue se ainda passa do limiar.
        for i, gt_id in enumerate(gt_ids):
            j = pred_index.get(last.get(gt_id))

            if j is not None and j not in used_pred and iou[i, j] >= threshold:
                frame_matches[gt_id] = pred_ids[j]
                used_gt.add(i)
                used_pred.add(j)

        # 2) Hungarian no que sobrou.
        free_gt = [i for i in range(len(gt_ids)) if i not in used_gt]
        free_pred = [j for j in range(len(pred_ids)) if j not in used_pred]

        pairs = match_frame(None, None, threshold,
                            iou=iou[np.ix_(free_gt, free_pred)])

        for a, b in pairs:
            frame_matches[gt_ids[free_gt[a]]] = pred_ids[free_pred[b]]

        last.update(frame_matches)
        matches[frame] = frame_matches

    return gt_frames, matches


def _clear_mot_counts(gt_tracks, pred_tracks, threshold):
    """
    ({gt_id: switches}, {gt_id: fragmentações}, casamentos) a partir de uma
    única passada do CLEAR MOT — switches, fragmentações e MOTA saem do mesmo
    casamento.
    """

    gt_frames, matches = _clear_mot_matches(gt_tracks, pred_tracks, threshold)

    switches = {}
    fragments = {}

    for gt_id, frames in gt_frames.items():
        matched = [matches[f].get(gt_id) for f in frames]

        previous = None
        n_switches = 0

        for pred_id in matched:
            if pred_id is None:
                continue
            if previous is not None and pred_id != previous:
                n_switches += 1
            previous = pred_id

        # Interrupções entre o primeiro e o último casamento: transições
        # casado -> não casado (que, por estarem antes do último casamento,
        # sempre voltam).
        hits = [pred_id is not None for pred_id in matched]
        n_fragments = 0

        if any(hits):
            first = hits.index(True)
            last = len(hits) - 1 - hits[::-1].index(True)
            span = hits[first:last + 1]
            n_fragments = sum(1 for a, b in zip(span, span[1:]) if a and not b)

        switches[gt_id] = n_switches
        fragments[gt_id] = n_fragments

    n_matches = sum(len(frame_matches) for frame_matches in matches.values())

    return switches, fragments, n_matches


def idf1(gt_tracks, pred_tracks, threshold=0.5):
    """
    IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN), com a atribuição global
    ID-previsto <-> ID-verdadeiro resolvida por Hungarian sobre o custo
    acumulado na sequência inteira.

    Devolve (idf1, idp, idr, detalhes). Sem nada para rastrear e nada
    previsto, IDF1 = 1 (não há erro nenhum).
    """

    gt = _by_frame(gt_tracks)
    pred = _by_frame(pred_tracks)

    gt_ids = sorted({i for ids, _ in gt.values() for i in ids})
    pred_ids = sorted({i for ids, _ in pred.values() for i in ids})
    gt_row = {i: k for k, i in enumerate(gt_ids)}
    pred_col = {i: k for k, i in enumerate(pred_ids)}

    # overlap[g, p] = quadros em que g e p estão presentes com IoU >= limiar.
    overlap = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)

    for frame, (frame_gt, gt_boxes) in gt.items():
        if frame not in pred:
            continue

        frame_pred, pred_boxes = pred[frame]
        hit = iou_matrix(gt_boxes, pred_boxes) >= threshold

        for i, j in zip(*np.nonzero(hit)):
            overlap[gt_row[frame_gt[i]], pred_col[frame_pred[j]]] += 1

    n_gt = sum(len(ids) for ids, _ in gt.values())
    n_pred = sum(len(ids) for ids, _ in pred.values())

    assignment = {}
    idtp = 0

    if overlap.size:
        rows, cols = linear_sum_assignment(-overlap)

        for r, c in zip(rows, cols):
            if overlap[r, c] > 0:
                assignment[gt_ids[r]] = pred_ids[c]
                idtp += int(overlap[r, c])

    idfp = n_pred - idtp
    idfn = n_gt - idtp

    details = {"idtp": idtp, "idfp": idfp, "idfn": idfn,
               "assignment": assignment}

    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0, details

    score = 2 * idtp / (2 * idtp + idfp + idfn)
    idp = idtp / n_pred if n_pred else 0.0
    idr = idtp / n_gt if n_gt else 0.0

    return score, idp, idr, details


def id_switches(gt_tracks, pred_tracks, threshold=0.5):
    """
    Número de vezes que uma identidade verdadeira muda de identidade prevista
    entre observações consecutivas casadas.
    """

    switches, _, _ = _clear_mot_counts(gt_tracks, pred_tracks, threshold)

    return sum(switches.values())


def fragmentations(gt_tracks, pred_tracks, threshold=0.5):
    """
    Número de interrupções na cobertura de uma identidade verdadeira (ela
    estava casada, deixou de estar, voltou).
    """

    _, fragments, _ = _clear_mot_counts(gt_tracks, pred_tracks, threshold)

    return sum(fragments.values())


def _mota_from_counts(tp, n_pred, n_gt, n_switches):
    fp = n_pred - tp
    fn = n_gt - tp
    score = 1.0 - (fn + fp + n_switches) / n_gt if n_gt else float("nan")

    return score, {"tp": tp, "fp": fp, "fn": fn, "id_switches": n_switches,
                   "n_gt": n_gt}


def mota(gt_tracks, pred_tracks, threshold=0.5):
    """
    MOTA = 1 - (FN + FP + IDSW) / caixas verdadeiras. Pode ser negativa
    (mais erros que caixas). Sem ground truth fica indefinida (nan).

    Devolve (mota, {tp, fp, fn, id_switches, n_gt}). Predição em quadro sem
    ground truth conta como FP.
    """

    switches, _, tp = _clear_mot_counts(gt_tracks, pred_tracks, threshold)

    return _mota_from_counts(tp, len(pred_tracks), len(gt_tracks),
                             sum(switches.values()))


def unique_id_count_error(gt_tracks, pred_tracks):
    """
    |identidades únicas previstas - verdadeiras| no vídeo. É o análogo
    temporal do erro de contagem do PA1.
    """

    n_gt = len({int(row[1]) for row in gt_tracks})
    n_pred = len({int(row[1]) for row in pred_tracks})

    return abs(n_pred - n_gt)


def evaluate_sequence(gt_tracks, pred_tracks, threshold=0.5):
    """
    Todas as métricas acima numa chamada, no formato que os scripts gravam em
    experiments/results/.

    id_ratio (ids previstos / verdadeiros) e id_switches_per_gt são os dois
    eixos do painel de baixo do gráfico da Parte 1.5. MOTA vai junto, mas é
    reportada à parte (ver o docstring do módulo).
    """

    score, idp, idr, details = idf1(gt_tracks, pred_tracks, threshold)
    switches, fragments, tp = _clear_mot_counts(gt_tracks, pred_tracks,
                                                threshold)

    n_gt_ids = len({int(row[1]) for row in gt_tracks})
    n_pred_ids = len({int(row[1]) for row in pred_tracks})
    n_switches = sum(switches.values())

    mota_score, counts = _mota_from_counts(tp, len(pred_tracks),
                                           len(gt_tracks), n_switches)

    return {
        "idf1": score,
        "idp": idp,
        "idr": idr,
        "idtp": details["idtp"],
        "idfp": details["idfp"],
        "idfn": details["idfn"],
        "id_switches": n_switches,
        "fragmentations": sum(fragments.values()),
        "n_gt_ids": n_gt_ids,
        "n_pred_ids": n_pred_ids,
        "id_count_error": abs(n_pred_ids - n_gt_ids),
        "id_ratio": n_pred_ids / n_gt_ids if n_gt_ids else float("nan"),
        "id_switches_per_gt": (n_switches / n_gt_ids if n_gt_ids
                               else float("nan")),
        "mota": mota_score,
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
    }
