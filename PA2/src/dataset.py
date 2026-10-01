"""
MOT17 / MOTChallenge: leitura, split e janelas temporais.

Formato do gt.txt e do det.txt:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

O campo visibility alimenta a análise de oclusão da Parte 4.

Split POR SEQUÊNCIA, nunca por quadro: separar quadros aleatoriamente coloca o
quadro t no treino e o t+1 na validação, e o modelo temporal seria avaliado em
cima do que praticamente já viu. Pelo menos uma sequência inteira fica fora.
"""

from pathlib import Path

import numpy as np

SEQUENCE_ROOT = "data/MOT17"
DETECTORS = ("DPM", "FRCNN", "SDP")
# A segunda fonte da Parte 1: Faster R-CNN pré-treinado do torchvision, em
# cache (scripts/detect_torchvision.py). Usa o gt e os quadros da pasta SDP.
DETECTION_SOURCES = DETECTORS + ("torchvision",)
TORCHVISION_CACHE = Path("cache") / "torchvision"

# Pedestres reais (conf=1, class=1) formam o gt. Das outras classes, só as
# distractoras do protocolo oficial (TrackEval, MOT17) — 2 pessoa em veículo,
# 7 pessoa estática, 8 distractor, 12 reflexo — removem a predição que casa
# com elas. Carro, bicicleta, moto, veículo e oclusores (3, 4, 5, 6, 9, 10,
# 11) só ficam fora do gt: predição em cima deles é FP.
PEDESTRIAN_CLASS = 1
DISTRACTOR_CLASSES = (2, 7, 8, 12)

SCENES = ("02", "04", "05", "09", "10", "11", "13")
TRAIN_SCENES = ("02", "04", "05", "10", "13")
TEST_SCENES = ("09", "11")

CAMERA = {
    "02": "static",
    "04": "static",
    "05": "moving",
    "09": "static",
    "10": "moving",
    "11": "moving",
    "13": "moving",
}

# Mediana das corridas com visibility < 0.25 (campo do gt, não buracos).
# 13 (3) → 05 (5) → 10 (7) → 11 (9) → 09 (12) → 04 (20) → 02 (29)
OCCLUSION_ORDER = ("13", "05", "10", "11", "09", "04", "02")
VISIBILITY_OCCLUDED = 0.25

_PA2_ROOT = Path(__file__).resolve().parent.parent


def resolve_root(root=SEQUENCE_ROOT):
    path = Path(root)
    if path.exists():
        return path
    fallback = _PA2_ROOT / root
    if fallback.exists():
        return fallback
    return path


def scene_id(name):
    token = Path(str(name)).name.replace("MOT17-", "")
    return token.split("-")[0].zfill(2)


def sequence_folder(name, detector="SDP"):
    raw = Path(str(name)).name
    if raw.startswith("MOT17-") and raw.count("-") >= 2:
        return raw
    # O torchvision não tem pasta própria no MOT17: gt, seqinfo e quadros são
    # os mesmos em DPM/FRCNN/SDP.
    folder_detector = "SDP" if detector == "torchvision" else detector
    return f"MOT17-{scene_id(name)}-{folder_detector}"


def torchvision_det_path(scene, root=SEQUENCE_ROOT):
    return resolve_root(root) / TORCHVISION_CACHE / f"MOT17-{scene_id(scene)}" / "det.txt"


def read_mot_file(path):
    """
    gt.txt ou det.txt -> lista de linhas já tipadas.

    gt:  (frame, id, x, y, w, h, conf, class, visibility)
    det: (frame, id, x, y, w, h, conf)  — class/visibility viram None
    """

    rows = []

    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [item.strip() for item in line.split(",")]
            frame = int(float(parts[0]))
            identity = int(float(parts[1]))
            x, y, w, h = (float(parts[2]), float(parts[3]),
                          float(parts[4]), float(parts[5]))
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            klass = int(float(parts[7])) if len(parts) > 7 else None
            visibility = float(parts[8]) if len(parts) > 8 else None
            rows.append((frame, identity, x, y, w, h, conf, klass, visibility))

    return rows


def _read_seqinfo(path):
    info = {}

    with open(path) as handle:
        for line in handle:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            info[key] = value

    return {
        "name": info.get("name", path.parent.name),
        "frame_rate": float(info.get("frameRate", 30)),
        "seq_length": int(info.get("seqLength", 0)),
        "im_width": int(info.get("imWidth", 0)),
        "im_height": int(info.get("imHeight", 0)),
        "im_dir": info.get("imDir", "img1"),
        "im_ext": info.get("imExt", ".jpg"),
    }


def split_ground_truth(rows):
    """
    Pedestres reais (conf=1, class=1) e distractores oficiais (classes 2, 7,
    8, 12). As outras linhas não entram em nenhuma das duas listas.
    """

    pedestrians, distractors = [], []

    for row in rows:
        conf, klass = row[6], row[7]
        if klass == PEDESTRIAN_CLASS and conf == 1:
            pedestrians.append(row)
        elif klass in DISTRACTOR_CLASSES:
            distractors.append(row)

    return pedestrians, distractors


def occlusion_runs(gt_tracks, visibility_threshold=VISIBILITY_OCCLUDED):
    """
    Por identidade, as corridas de quadros consecutivos com visibility
    abaixo do limiar. MOT17 não remove ocluídos: o campo visibility é o
    sinal, não a ausência da linha.
    """

    by_id = {}

    for row in gt_tracks:
        if row[8] is None:
            continue
        by_id.setdefault(int(row[1]), []).append((int(row[0]), float(row[8])))

    runs = {}

    for identity, frames in by_id.items():
        frames.sort()
        current = 0
        identity_runs = []

        for _, visibility in frames:
            if visibility < visibility_threshold:
                current += 1
            elif current:
                identity_runs.append(current)
                current = 0

        if current:
            identity_runs.append(current)

        runs[identity] = identity_runs

    return runs


def occlusion_intervals(gt_tracks, visibility_threshold=VISIBILITY_OCCLUDED):
    """
    Por identidade, intervalos (quadro_inicial, quadro_final, duração) com
    visibility abaixo do limiar.
    """

    by_id = {}

    for row in gt_tracks:
        if row[8] is None:
            continue
        by_id.setdefault(int(row[1]), []).append((int(row[0]), float(row[8])))

    intervals = {}

    for identity, items in by_id.items():
        items.sort()
        start = None
        found = []
        for frame, visibility in items:
            if visibility < visibility_threshold:
                if start is None:
                    start = frame
            elif start is not None:
                found.append((start, frame - 1, frame - start))
                start = None
        if start is not None:
            found.append((start, items[-1][0], items[-1][0] - start + 1))
        intervals[identity] = found

    return intervals


def median_occlusion(gt_tracks, visibility_threshold=VISIBILITY_OCCLUDED):
    lengths = [length
               for runs in occlusion_runs(gt_tracks, visibility_threshold).values()
               for length in runs]

    if not lengths:
        return 0.0

    ordered = sorted(lengths)
    mid = len(ordered) // 2

    if len(ordered) % 2:
        return float(ordered[mid])

    return 0.5 * (ordered[mid - 1] + ordered[mid])


def load_sequence(name, detector="SDP", root=SEQUENCE_ROOT):
    """
    Devolve (info, gt_tracks, detections) de uma sequência.

    gt_tracks são só pedestres (conf=1, class=1). Todas as linhas do gt.txt
    (todas as classes) ficam em info["gt_all"], que é o que
    src.metrics.drop_distractor_matches precisa para casar como o TrackEval.
    detections são as linhas cruas do det.txt; o NMS e o limiar de score
    entram em src.detection.public_detections.
    """

    root = resolve_root(root)
    folder = sequence_folder(name, detector)
    scene = scene_id(folder)

    candidates = [root / "train" / folder, root / "test" / folder, root / folder]
    path = next((item for item in candidates if item.exists()), None)

    if path is None:
        raise FileNotFoundError(f"sequência {folder} não encontrada em {root}")

    info = _read_seqinfo(path / "seqinfo.ini")
    raw_gt = read_mot_file(path / "gt" / "gt.txt") if (path / "gt" / "gt.txt").exists() else []
    pedestrians, _ = split_ground_truth(raw_gt)

    if detector == "torchvision":
        det_path = torchvision_det_path(scene, root)
        if not det_path.exists():
            raise FileNotFoundError(
                f"sem detecções do torchvision para {scene}: rode "
                f"python scripts/detect_torchvision.py (grava {det_path})")
    else:
        det_path = path / "det" / "det.txt"
    detections = read_mot_file(det_path) if det_path.exists() else []

    n_frames = info["seq_length"] or (max((row[0] for row in raw_gt), default=0))
    n_ids = len({row[1] for row in pedestrians})
    density = len(pedestrians) / n_frames if n_frames else 0.0

    info.update({
        "path": str(path),
        "scene": scene,
        "detector": detector,
        "camera": CAMERA.get(scene, "unknown"),
        "gt_all": raw_gt,
        "n_frames": n_frames,
        "n_ids": n_ids,
        "density": density,
        "occlusion": median_occlusion(pedestrians),
    })

    return info, pedestrians, detections


def list_scenes(root=SEQUENCE_ROOT, detector="SDP"):
    root = resolve_root(root)
    train = root / "train"

    if not train.exists():
        return list(SCENES)

    found = []
    if detector == "torchvision":
        for path in sorted(train.glob("MOT17-*-SDP")):
            if torchvision_det_path(path.name, root).exists():
                found.append(scene_id(path.name))
        return found

    for path in sorted(train.glob(f"MOT17-*-{detector}")):
        found.append(scene_id(path.name))

    return found or list(SCENES)


def create_splits(sequences, seed=42):
    """
    Split por sequência, idêntico para qualquer seed. test = 09 + 11
    (uma câmera parada de rua, uma móvel indoor); train = o resto.
    """

    del seed
    scenes = [scene_id(name) for name in sequences]

    return {
        "train": [scene for scene in TRAIN_SCENES if scene in scenes],
        "test": [scene for scene in TEST_SCENES if scene in scenes],
    }


def split_report(splits, detector="SDP", root=SEQUENCE_ROOT):
    """
    Imprime quantos quadros, identidades, densidade e câmera por split.
    """

    lines = []

    for split_name, scenes in splits.items():
        print(f"\n[{split_name}] {len(scenes)} sequências")
        print(f"{'scene':<8} {'camera':<8} {'frames':>7} {'ids':>5} "
              f"{'density':>8} {'occlusion':>10}")

        total_frames = total_ids = 0
        densities = []

        for scene in scenes:
            info, _, _ = load_sequence(scene, detector=detector, root=root)
            print(f"{scene:<8} {info['camera']:<8} {info['n_frames']:>7} "
                  f"{info['n_ids']:>5} {info['density']:>8.2f} "
                  f"{info['occlusion']:>10.1f}")
            total_frames += info["n_frames"]
            total_ids += info["n_ids"]
            densities.append(info["density"])
            lines.append((split_name, scene, info))

        mean_density = sum(densities) / len(densities) if densities else 0.0
        print(f"{'total':<8} {'':<8} {total_frames:>7} {total_ids:>5} "
              f"{mean_density:>8.2f}")

    return lines


def load_frame(sequence_path, frame, im_dir="img1", im_ext=".jpg"):
    from PIL import Image

    path = Path(sequence_path) / im_dir / f"{int(frame):06d}{im_ext}"
    return np.asarray(Image.open(path).convert("RGB"))


class TrackWindowDataset:
    """
    Trajetórias do ground truth fatiadas em janelas de T quadros, que é a
    unidade de BPTT truncado do treino da Parte 2 e do Eixo 1 da Parte 3.

    `min_visibility` tira da janela as observações de pessoa escondida: o
    recorte de quem tem visibility < limiar mostra o oclusor, não ela, e
    viraria positivo errado na perda contrastiva. A memória só dá passo nas
    observações que ficam, como na inferência.
    """

    def __init__(self, sequences, window=16, stride=1, detector="SDP",
                 root=SEQUENCE_ROOT, min_visibility=0.0):
        if window < 2:
            raise ValueError("window precisa ser >= 2")
        if set(scene_id(name) for name in sequences) & set(TEST_SCENES):
            raise ValueError("TrackWindowDataset não pode ver o split de teste")

        self.window = int(window)
        self.stride = int(stride)
        self.items = []

        for name in sequences:
            info, gt, _ = load_sequence(name, detector=detector, root=root)
            by_frame = {}
            for row in gt:
                if (row[8] is not None and min_visibility > 0
                        and float(row[8]) < min_visibility):
                    continue
                by_frame.setdefault(int(row[0]), []).append(row)

            if not by_frame:
                continue

            first, last = min(by_frame), max(by_frame)

            for start in range(first, last - window + 2, self.stride):
                identities = {}
                for frame in range(start, start + window):
                    for row in by_frame.get(frame, []):
                        identities.setdefault(int(row[1]), []).append(
                            (frame, (float(row[2]), float(row[3]),
                                     float(row[4]), float(row[5]))))
                if len(identities) < 2:
                    continue
                self.items.append({
                    "scene": info["scene"],
                    "path": info["path"],
                    "start": start,
                    "window": window,
                    "identities": identities,
                    "im_width": info["im_width"],
                    "im_height": info["im_height"],
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
