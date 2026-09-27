"""
Gestão de tracks: nascimento, morte e o laço de rastreamento (Parte 1.2/1.4).

O mesmo laço serve ao baseline ingênuo e ao modelo temporal da Parte 2 — o que
muda é quem prevê a caixa (ou o embedding) do próximo quadro.
"""

from collections import defaultdict

import numpy as np

import torch

from src.association import cosine_cost, greedy_match, hungarian_match, iou_cost


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
        self.embedding = None


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
                 matcher="hungarian", motion=None, appearance_threshold=0.5):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.matcher = matcher
        self.motion = motion
        self.appearance_threshold = float(appearance_threshold)
        self.tracks = []
        self._next_id = 1

    def _uses_appearance(self):
        return self.motion is not None and hasattr(self.motion, "step")

    def _queries(self, embeddings):
        """
        Recorte no espaço da memória: um passo de GRU a partir do estado
        zero, para o cosseno comparar memória com consulta — não memória
        com embedding cru do ResNet.
        """

        if hasattr(self.motion, "parameters"):
            device = next(self.motion.parameters()).device
        else:
            device = "cpu"

        tensor = torch.as_tensor(np.asarray(embeddings), device=device,
                                 dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        state = self.motion.init_state(len(tensor), device)
        with torch.no_grad():
            output, _ = self.motion.step(tensor, state)
        if torch.is_tensor(output):
            return output.detach().cpu().numpy().reshape(len(tensor), -1)
        return np.asarray(output, dtype=np.float32).reshape(len(tensor), -1)

    def _match(self, detections, embeddings=None):
        if not self.tracks or len(detections) == 0:
            return []

        matcher = hungarian_match if self.matcher == "hungarian" else greedy_match

        if not self._uses_appearance() or embeddings is None:
            track_boxes = np.stack([track.box for track in self.tracks])
            return matcher(iou_cost(track_boxes, detections),
                           1.0 - self.iou_threshold)

        # Dois estágios: IoU nas tracks recém-vistas (não piorar o
        # baseline), aparência só depois do miss.
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
        if not remain_tracks or not remain_dets:
            return pairs

        queries = self._queries(embeddings[remain_dets])
        dim = queries.shape[1]
        memory = np.stack([
            self.tracks[i].embedding if self.tracks[i].embedding is not None
            else np.zeros(dim, dtype=np.float64)
            for i in remain_tracks
        ])
        for a, b in matcher(cosine_cost(memory, queries),
                            self.appearance_threshold):
            pairs.append((remain_tracks[a], remain_dets[b]))
        return pairs

    def _update_appearance(self, track, embedding):
        if embedding is None or not self._uses_appearance():
            return

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if hasattr(self.motion, "parameters"):
            device = next(self.motion.parameters()).device
        else:
            device = "cpu"

        tensor = torch.as_tensor(vector, device=device)
        state = track.state
        if state is None:
            state = self.motion.init_state(1, device)

        with torch.no_grad():
            output, track.state = self.motion.step(tensor, state)
        if torch.is_tensor(output):
            track.embedding = output.detach().cpu().numpy().reshape(-1)
        else:
            track.embedding = np.asarray(output, dtype=np.float32).reshape(-1)

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
            if embeddings is not None:
                self._update_appearance(track, embeddings[j])
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
            if embeddings is not None:
                self._update_appearance(track, embeddings[j])
            if track.hits >= self.min_hits:
                track.confirmed = True
                emitted.append((int(frame), track.track_id,
                                float(track.box[0]), float(track.box[1]),
                                float(track.box[2]), float(track.box[3])))
            self.tracks.append(track)

        return emitted

    def run(self, sequence, embeddings=None):
        """
        Sequência inteira -> pred_tracks no formato que src/metrics.py consome.

        Aceita linhas (frame, x, y, w, h[, score]) ou (frame, id, x, y, w, h, ...).
        `embeddings` alinha 1:1 com `sequence` quando a Trilha B está ligada.
        """

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
