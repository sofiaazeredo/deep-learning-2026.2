"""
Gestão de tracks: nascimento, morte e o laço de rastreamento (Parte 1.2/1.4).

O mesmo laço serve ao baseline ingênuo e ao modelo temporal da Parte 2 — o que
muda é quem prevê a caixa (ou o embedding) do próximo quadro.
"""

from collections import defaultdict

import numpy as np

from src.association import greedy_match, hungarian_match, iou_cost


class Track:
    """
    Estado de um objeto: identidade, última caixa, idade, quadros sem
    observação e o estado recorrente (Parte 2).
    """

    def __init__(self, track_id, box, frame):
        self.track_id = int(track_id)
        self.box = np.asarray(box, dtype=np.float64).reshape(4)
        self.start_frame = int(frame)
        self.last_frame = int(frame)
        self.hits = 1
        self.time_since_update = 0
        self.confirmed = False
        self.state = None


class Tracker:
    """
    Parâmetros de regra:
      iou_threshold   limiar de associação
      max_age         k quadros sem observação antes de matar a track
      min_hits        observações consecutivas antes de a track ser emitida
      matcher         'greedy' ou 'hungarian'
      motion          None (baseline); Kalman / RNN ficam para depois
    """

    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=3,
                 matcher="hungarian", motion=None):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.matcher = matcher
        self.motion = motion
        self.tracks = []
        self._next_id = 1

    def _match(self, detections):
        if not self.tracks or len(detections) == 0:
            return []

        track_boxes = np.stack([track.box for track in self.tracks])
        cost = iou_cost(track_boxes, detections)
        gate = 1.0 - self.iou_threshold
        matcher = hungarian_match if self.matcher == "hungarian" else greedy_match
        return matcher(cost, gate)

    def update(self, detections, frame):
        """
        Um quadro: associar à última caixa observada, atualizar, nascer, morrer.

        Só emite a caixa do quadro em que houve match, e só depois de
        min_hits matches consecutivos. Miss não escreve caixa (sem coasting).
        """

        boxes = np.asarray(detections, dtype=np.float64)
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float64)
        else:
            boxes = boxes.reshape(-1, 4)

        pairs = self._match(boxes)
        matched_tracks = {i for i, _ in pairs}
        matched_dets = {j for _, j in pairs}
        emitted = []

        for i, j in pairs:
            track = self.tracks[i]
            consecutive = track.time_since_update == 0
            if track.confirmed:
                track.hits += 1
            elif consecutive:
                track.hits += 1
            else:
                track.hits = 1
            track.box = boxes[j].copy()
            track.time_since_update = 0
            track.last_frame = int(frame)
            if track.hits >= self.min_hits:
                track.confirmed = True
                emitted.append((int(frame), track.track_id,
                                float(track.box[0]), float(track.box[1]),
                                float(track.box[2]), float(track.box[3])))

        survivors = []

        for i, track in enumerate(self.tracks):
            if i in matched_tracks:
                survivors.append(track)
                continue
            track.time_since_update += 1
            if not track.confirmed:
                track.hits = 0
            if track.time_since_update <= self.max_age:
                survivors.append(track)

        self.tracks = survivors

        for j, box in enumerate(boxes):
            if j in matched_dets:
                continue
            track = Track(self._next_id, box, frame)
            self._next_id += 1
            if track.hits >= self.min_hits:
                track.confirmed = True
                emitted.append((int(frame), track.track_id,
                                float(track.box[0]), float(track.box[1]),
                                float(track.box[2]), float(track.box[3])))
            self.tracks.append(track)

        return emitted

    def run(self, sequence):
        """
        Sequência inteira -> pred_tracks no formato que src/metrics.py consome.

        Aceita linhas (frame, x, y, w, h[, score]) ou (frame, id, x, y, w, h, ...).
        """

        by_frame = defaultdict(list)

        for row in sequence:
            frame = int(row[0])
            if len(row) == 6:
                box = row[1:5]
            else:
                box = row[2:6]
            by_frame[frame].append(box)

        if not by_frame:
            return []

        tracks = []

        for frame in range(min(by_frame), max(by_frame) + 1):
            tracks.extend(self.update(by_frame.get(frame, []), frame))

        return tracks


class KalmanMotion:
    """
    Filtro de Kalman de velocidade constante. Permitido APENAS como baseline de
    comparação — não pode ser o modelo temporal da Parte 2.
    """

    def __init__(self):
        raise NotImplementedError
