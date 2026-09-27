"""
Gestão de tracks: nascimento, morte e o laço de rastreamento (Parte 1.2/1.4).

O mesmo laço serve ao baseline ingênuo e ao modelo temporal da Parte 2 — o que
muda é quem prevê a caixa (ou o embedding) do próximo quadro.
"""

from collections import defaultdict

import numpy as np

import torch

from src.association import cosine_cost, greedy_match, hungarian_match, iou_cost
from src.boxes import norm_to_xywh, xywh_to_norm
from src.model import unpack_step


class Track:
    """
    Estado de um objeto: identidade, última caixa, idade, quadros sem
    observação e o estado recorrente (Parte 2).
    """

    def __init__(self, track_id, box, frame):
        self.track_id = int(track_id)
        self.box = np.asarray(box, dtype=np.float64).reshape(4)
        self.predicted_box = self.box.copy()
        self.start_frame = int(frame)
        self.last_frame = int(frame)
        self.hits = 1
        self.time_since_update = 0
        self.confirmed = False
        self.state = None
        self.embedding = None


class Tracker:
    """
    Parâmetros de regra:
      iou_threshold   limiar de associação
      max_age         k quadros sem observação antes de matar a track
      min_hits        observações consecutivas antes de a track ser emitida
      matcher         'greedy' ou 'hungarian'
      motion          None (baseline); RNN de aparência / geometria / fusão
    """

    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=3,
                 matcher="hungarian", motion=None, appearance_threshold=0.5,
                 image_size=None, miss_prefer_iou=False):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.matcher = matcher
        self.motion = motion
        self.appearance_threshold = float(appearance_threshold)
        self.image_size = image_size or (1.0, 1.0)
        self.miss_prefer_iou = bool(miss_prefer_iou)
        self.tracks = []
        self._next_id = 1

    def _kind(self):
        if self.motion is None:
            return None
        return getattr(self.motion, "kind", "appearance")

    def _uses_model(self):
        return self.motion is not None and hasattr(self.motion, "step")

    def _device(self):
        if hasattr(self.motion, "parameters"):
            return next(self.motion.parameters()).device
        return "cpu"

    def _queries(self, embeddings, boxes=None):
        """
        Recorte no espaço da memória: um passo a partir do estado zero.
        """

        device = self._device()
        kind = self._kind()
        if kind == "both":
            width, height = self.image_size
            feats = []
            for i, vector in enumerate(embeddings):
                box = xywh_to_norm(boxes[i], width, height)
                feats.append(np.concatenate([np.asarray(vector, dtype=np.float32),
                                             box]))
            tensor = torch.as_tensor(np.stack(feats), device=device,
                                     dtype=torch.float32)
        else:
            tensor = torch.as_tensor(np.asarray(embeddings), device=device,
                                     dtype=torch.float32)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)

        state = self.motion.init_state(len(tensor), device)
        with torch.no_grad():
            output, _ = self.motion.step(tensor, state)
        vector, _ = unpack_step(output, kind)
        if torch.is_tensor(vector):
            return vector.detach().cpu().numpy().reshape(len(tensor), -1)
        return np.asarray(vector, dtype=np.float32).reshape(len(tensor), -1)

    def _step_track(self, track, embedding=None, box=None):
        if not self._uses_model():
            return

        kind = self._kind()
        width, height = self.image_size
        device = self._device()
        if box is None:
            box = track.predicted_box if track.predicted_box is not None else track.box
        box_norm = xywh_to_norm(box, width, height)

        if kind == "appearance":
            if embedding is None:
                return
            features = np.asarray(embedding, dtype=np.float32).reshape(-1)
        elif kind == "geometry":
            features = box_norm
        else:
            vector = embedding
            if vector is None:
                vector = track.embedding
            if vector is None:
                dim = getattr(self.motion, "embed_dim", 128)
                vector = np.zeros(dim, dtype=np.float32)
            features = np.concatenate([np.asarray(vector, dtype=np.float32).reshape(-1),
                                       box_norm])

        tensor = torch.as_tensor(features, device=device, dtype=torch.float32)
        state = track.state
        if state is None:
            state = self.motion.init_state(1, device)
        with torch.no_grad():
            output, track.state = self.motion.step(tensor, state)
        vector, pred = unpack_step(output, kind)
        if vector is not None:
            if torch.is_tensor(vector):
                track.embedding = vector.detach().cpu().numpy().reshape(-1)
            else:
                track.embedding = np.asarray(vector, dtype=np.float32).reshape(-1)
        if pred is not None:
            if torch.is_tensor(pred):
                pred = pred.detach().cpu().numpy().reshape(-1)
            track.predicted_box = norm_to_xywh(pred, width, height)

    def _match(self, detections, embeddings=None):
        if not self.tracks or len(detections) == 0:
            return []

        matcher = hungarian_match if self.matcher == "hungarian" else greedy_match
        kind = self._kind()

        if not self._uses_model():
            track_boxes = np.stack([track.box for track in self.tracks])
            return matcher(iou_cost(track_boxes, detections),
                           1.0 - self.iou_threshold)

        live = [i for i, track in enumerate(self.tracks)
                if track.time_since_update == 0]
        lost = [i for i, track in enumerate(self.tracks)
                if track.time_since_update > 0]

        pairs = []
        used_tracks, used_dets = set(), set()

        if live:
            boxes = np.stack([self.tracks[i].box for i in live])
            for a, j in matcher(iou_cost(boxes, detections),
                                1.0 - self.iou_threshold):
                pairs.append((live[a], j))
                used_tracks.add(live[a])
                used_dets.add(j)

        remain_tracks = [i for i in lost if i not in used_tracks]
        remain_dets = [j for j in range(len(detections)) if j not in used_dets]
        if self.miss_prefer_iou and remain_tracks and remain_dets:
            last_boxes = np.stack([self.tracks[i].box for i in remain_tracks])
            leftover = detections[remain_dets]
            for a, b in matcher(iou_cost(last_boxes, leftover),
                                1.0 - self.iou_threshold):
                pairs.append((remain_tracks[a], remain_dets[b]))
                used_tracks.add(remain_tracks[a])
                used_dets.add(remain_dets[b])
            remain_tracks = [i for i in lost if i not in used_tracks]
            remain_dets = [j for j in range(len(detections)) if j not in used_dets]

        if not remain_tracks or not remain_dets:
            return pairs

        leftover = detections[remain_dets]
        pred_boxes = np.stack([
            self.tracks[i].predicted_box if self.tracks[i].predicted_box is not None
            else self.tracks[i].box
            for i in remain_tracks
        ])
        iou_c = iou_cost(pred_boxes, leftover)

        if kind == "geometry" or embeddings is None:
            for a, b in matcher(iou_c, 1.0 - self.iou_threshold):
                pairs.append((remain_tracks[a], remain_dets[b]))
            return pairs

        queries = self._queries(embeddings[remain_dets], boxes=leftover)
        dim = queries.shape[1]
        memory = np.stack([
            self.tracks[i].embedding if self.tracks[i].embedding is not None
            else np.zeros(dim, dtype=np.float64)
            for i in remain_tracks
        ])
        app_c = cosine_cost(memory, queries)
        cost = np.minimum(app_c, iou_c) if kind == "both" else app_c
        threshold = 0.5 if kind == "both" else self.appearance_threshold
        for a, b in matcher(cost, threshold):
            pairs.append((remain_tracks[a], remain_dets[b]))
        return pairs

    def update(self, detections, frame, embeddings=None):
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

        if embeddings is not None:
            embeddings = np.asarray(embeddings, dtype=np.float64)
            if embeddings.size == 0:
                embeddings = None
            else:
                embeddings = embeddings.reshape(len(boxes), -1)

        pairs = self._match(boxes, embeddings)
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
            vector = embeddings[j] if embeddings is not None else None
            self._step_track(track, embedding=vector, box=boxes[j])
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
                self._step_track(track)
                survivors.append(track)

        self.tracks = survivors

        for j, box in enumerate(boxes):
            if j in matched_dets:
                continue
            track = Track(self._next_id, box, frame)
            self._next_id += 1
            vector = embeddings[j] if embeddings is not None else None
            self._step_track(track, embedding=vector, box=box)
            if track.hits >= self.min_hits:
                track.confirmed = True
                emitted.append((int(frame), track.track_id,
                                float(track.box[0]), float(track.box[1]),
                                float(track.box[2]), float(track.box[3])))
            self.tracks.append(track)

        return emitted

    def run(self, sequence, embeddings=None, image_size=None):
        """
        Sequência inteira -> pred_tracks no formato que src/metrics.py consome.

        Aceita linhas (frame, x, y, w, h[, score]) ou (frame, id, x, y, w, h, ...).
        `embeddings` alinha 1:1 com `sequence` quando a Trilha B está ligada.
        """

        if image_size is not None:
            self.image_size = image_size

        by_frame = defaultdict(list)
        emb_by_frame = defaultdict(list)

        for index, row in enumerate(sequence):
            frame = int(row[0])
            if len(row) == 6:
                box = row[1:5]
            else:
                box = row[2:6]
            by_frame[frame].append(box)
            if embeddings is not None:
                emb_by_frame[frame].append(embeddings[index])

        if not by_frame:
            return []

        tracks = []

        for frame in range(min(by_frame), max(by_frame) + 1):
            frame_emb = emb_by_frame.get(frame) if embeddings is not None else None
            tracks.extend(self.update(by_frame.get(frame, []), frame,
                                      embeddings=frame_emb))

        return tracks


class KalmanMotion:
    """
    Filtro de Kalman de velocidade constante. Permitido APENAS como baseline de
    comparação — não pode ser o modelo temporal da Parte 2.
    """

    def __init__(self):
        raise NotImplementedError
