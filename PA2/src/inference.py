"""
Inferência numa sequência qualquer (entregável inferencia.ipynb).

Recebe o caminho de uma sequência, devolve o vídeo com as identidades
coloridas de forma consistente e a contagem de objetos únicos. Roda sem
retreinar.

"Qualquer" quer dizer qualquer pasta com os quadros:
  - seqinfo.ini é opcional (dá o fps e a pasta dos quadros; sem ele, img1/
    ou a própria pasta, 30 fps);
  - det/det.txt é opcional: sem ele (ou com detector="torchvision") o
    Faster R-CNN do torchvision roda nos quadros, em blocos;
  - os embeddings dos recortes são calculados na hora, não saem do cache
    por cena do treino.

As regras do tracker são as congeladas no README (IoU 0,2, min_hits 2,
max_age 20, Hungarian, cascata com portão geométrico) mais a correção da
Parte 4 (prefer_confirmed).
"""

import colorsys
import configparser
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.detection import (
    TORCHVISION_CACHE_SCORE,
    default_score_threshold,
    load_torchvision_detector,
    public_detections,
    torchvision_detections,
)
from src.tracker import Tracker

PA2_ROOT = Path(__file__).resolve().parent.parent
BEST_MODEL = PA2_ROOT / "experiments" / "results" / "best_model.json"
# O modelo final: as regras congeladas + a correção da Parte 4.
TRACKER_KW = dict(iou_threshold=0.2, max_age=20, min_hits=2,
                  matcher="hungarian", appearance_threshold=0.5,
                  prefer_confirmed=True)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def resolve_checkpoint(checkpoint=None, best_model=BEST_MODEL, root=PA2_ROOT):
    """
    Caminho do checkpoint. None lê best_model.json. Caminho relativo é
    relativo à raiz do PA2 (não ao diretório atual). Um caminho absoluto
    gravado noutra máquina que não existe mais cai para o mesmo nome dentro
    de checkpoints/.
    """

    root = Path(root)

    if checkpoint is None:
        with open(best_model) as handle:
            checkpoint = json.load(handle)["checkpoint"]

    path = Path(checkpoint)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        fallback = root / "checkpoints" / path.name
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"checkpoint não encontrado: {checkpoint}")
    return path


def read_sequence(sequence_path):
    """
    Quadros (ordenados), fps, tamanho e det.txt de uma pasta qualquer.
    """

    folder = Path(sequence_path)
    fps, im_dir = 30, None

    seqinfo = folder / "seqinfo.ini"
    if seqinfo.exists():
        parser = configparser.ConfigParser()
        parser.read(seqinfo)
        section = parser["Sequence"]
        fps = int(float(section.get("frameRate", fps)))
        im_dir = section.get("imDir")

    candidates = [folder / im_dir] if im_dir else []
    candidates += [folder / "img1", folder]
    frame_dir = next((c for c in candidates
                      if c.is_dir() and _image_paths(c)), None)
    if frame_dir is None:
        raise FileNotFoundError(f"nenhum quadro em {folder}")

    paths = _image_paths(frame_dir)
    det = folder / "det" / "det.txt"
    return {"folder": folder, "frames": paths, "fps": fps,
            "det_path": det if det.exists() else None}


def _image_paths(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS)


def _load_image(path):
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def load_detections(sequence, detector="SDP", device="cuda", chunk_size=32):
    """
    det.txt da pasta (qualquer que seja o detector que o gerou) ou, sem ele
    ou com detector="torchvision", o Faster R-CNN nos quadros. Devolve
    (detecções (frame, x, y, w, h, score), nome da fonte).
    """

    if detector != "torchvision" and sequence["det_path"] is not None:
        from src.dataset import read_mot_file

        rows = read_mot_file(sequence["det_path"])
        return public_detections(rows), detector

    model = load_torchvision_detector(device)
    detections = []
    paths = sequence["frames"]
    for start in range(0, len(paths), chunk_size):
        frames = [_load_image(p) for p in paths[start:start + chunk_size]]
        detections.extend(torchvision_detections(
            frames, score_threshold=TORCHVISION_CACHE_SCORE, device=device,
            model=model, first_frame=start + 1))
    # A mesma regra dos resultados baseline_torchvision_*: o detector guarda
    # a partir do piso do modelo e o rastreamento entra com o limiar varrido
    # no treino (default_score_threshold), depois do nosso NMS.
    rows = [(f, -1, x, y, w, h, s) for f, x, y, w, h, s in detections]
    return public_detections(
        rows, score_threshold=default_score_threshold("torchvision")), "torchvision"


def embed_detections(sequence, detections, encoder, device="cuda",
                     batch_size=64):
    """
    Embedding de cada detecção, quadro a quadro, alinhado 1:1 com
    `detections`.
    """

    from src.appearance import crop

    by_frame = {}
    for index, row in enumerate(detections):
        by_frame.setdefault(int(row[0]), []).append(index)

    vectors = [None] * len(detections)
    pending_crops, pending_index = [], []

    def flush():
        if pending_crops:
            out = encoder.embed(pending_crops, device=device).cpu().numpy()
            for index, vector in zip(pending_index, out):
                vectors[index] = vector
            pending_crops.clear()
            pending_index.clear()

    for frame, indices in sorted(by_frame.items()):
        image = _load_image(sequence["frames"][frame - 1])
        for index in indices:
            pending_crops.append(crop(image, detections[index][1:5]))
            pending_index.append(index)
        if len(pending_crops) >= batch_size:
            flush()
    flush()
    return vectors


def _load_model(checkpoint, device):
    """
    (modelo, encoder) ou (None, None) no baseline. O encoder só existe para
    os braços que usam aparência.
    """

    if checkpoint is False:
        return None, None

    from src.appearance import CropEncoder
    from src.model import load_checkpoint

    model, payload = load_checkpoint(resolve_checkpoint(checkpoint),
                                     device=device)
    encoder = None
    if getattr(model, "kind", "appearance") in {"appearance", "both"}:
        embed_dim = payload.get("kwargs", {}).get("embed_dim", 128)
        encoder = CropEncoder(embed_dim=embed_dim, freeze=True).to(device)
        if "encoder_proj" in payload:
            encoder.proj.load_state_dict(payload["encoder_proj"])
    return model, encoder


def track_sequence(sequence_path, checkpoint=None, detector="SDP",
                   device=None, tracker_kwargs=None):
    """
    Devolve um dicionário com:
      tracks    trajetórias previstas (frame, id, x, y, w, h)
      count     número de objetos únicos no vídeo
      frames    quadros anotados, coloridos por identidade (carregados sob
                demanda: len(), indexação e iteração)
      fps       taxa de quadros da sequência
      detector  fonte de detecções usada

    checkpoint: None = modelo final (best_model.json); um caminho; ou False
    para o baseline da Parte 1 (só IoU, sem modelo).
    """

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    sequence = read_sequence(sequence_path)
    detections, source = load_detections(sequence, detector=detector,
                                         device=device)
    model, encoder = _load_model(checkpoint, device)

    embeddings = None
    if encoder is not None and detections:
        embeddings = embed_detections(sequence, detections, encoder,
                                      device=device)

    height, width = _load_image(sequence["frames"][0]).shape[:2]
    kwargs = dict(TRACKER_KW)
    kwargs.update(tracker_kwargs or {})
    tracker = Tracker(**kwargs, motion=model, image_size=(width, height))
    tracks = tracker.run(detections, embeddings=embeddings,
                         image_size=(width, height))

    return {
        "tracks": tracks,
        "count": len({row[1] for row in tracks}),
        "frames": AnnotatedFrames(sequence["frames"], tracks),
        "fps": sequence["fps"],
        "detector": source,
    }


class AnnotatedFrames(Sequence):
    """
    Quadros com as caixas desenhadas, lidos do disco só quando pedidos: um
    vídeo 1080p de mil quadros não cabe na memória.
    """

    def __init__(self, paths, tracks):
        self.paths = list(paths)
        self.by_frame = {}
        for frame, track_id, x, y, w, h in tracks:
            self.by_frame.setdefault(int(frame), []).append(((x, y, w, h),
                                                             track_id))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        items = self.by_frame.get(index + 1, [])
        return draw_tracks(_load_image(self.paths[index]),
                           [box for box, _ in items],
                           [track_id for _, track_id in items])


def color_for(track_id):
    """
    Cor RGB estável por identidade: matiz pela razão áurea, então ids
    vizinhos ficam com cores bem diferentes.
    """

    hue = (int(track_id) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def draw_tracks(frame, boxes, ids):
    """
    Uma cor estável por identidade, para o olho conferir o que a métrica diz.
    Devolve uma cópia; o quadro de entrada não muda.
    """

    import cv2

    image = np.ascontiguousarray(np.array(frame, dtype=np.uint8, copy=True))
    thickness = max(1, image.shape[0] // 400)

    for (x, y, w, h), track_id in zip(boxes, ids):
        color = color_for(track_id)
        p1 = (int(round(x)), int(round(y)))
        p2 = (int(round(x + w)), int(round(y + h)))
        cv2.rectangle(image, p1, p2, color, thickness)
        cv2.putText(image, str(track_id), (p1[0], max(p1[1] - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thickness, color,
                    thickness)
    return image


def write_video(frames, path, fps=30):
    """
    Quadros RGB -> mp4 (mp4v). Aceita lista ou AnnotatedFrames.
    """

    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for frame in frames:
            image = np.asarray(frame, dtype=np.uint8)
            if writer is None:
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(str(path),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         float(fps), (width, height))
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
    return path


def windowed_inference(sequence_path, window=64, overlap=8, **kwargs):
    """
    Um vídeo de duas horas não cabe na memória, então a inferência roda em
    janelas de T quadros. A costura de identidades entre janelas é o análogo
    temporal da fusão entre tiles do mosaico do PA1 (pergunta da Parte 2).
    """

    raise NotImplementedError(
        "A inferência em janelas é a pergunta de discussão da Parte 2 "
        "(o enunciado pede para responder sem implementar); "
        "track_sequence processa a sequência inteira.")
