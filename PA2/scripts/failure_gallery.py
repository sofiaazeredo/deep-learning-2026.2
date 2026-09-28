"""
Parte 4 — galeria de falhas.

Três trechos em que o modelo final erra feio. Cada um com a tira de quadros
(ground truth e predição coloridos por identidade) mais o mapa intermediário
relevante — a matriz de similaridade dos embeddings — e o diagnóstico.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.appearance import CropEncoder, cache_sequence_embeddings
from src.association import cosine_cost
from src.dataset import SCENES, load_frame, load_sequence
from src.detection import public_detections
from src.metrics import drop_distractor_matches, switch_events
from src.model import load_checkpoint

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
FIGURES = Path(__file__).resolve().parents[1] / "experiments" / "figures"


def read_tracks(path):
    by_sequence = defaultdict(list)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            by_sequence[str(row["sequence"]).zfill(2)].append((
                int(float(row["frame"])), int(float(row["id"])),
                float(row["x"]), float(row["y"]),
                float(row["w"]), float(row["h"]),
            ))
    return by_sequence


def collect_events(tracks_path):
    by_sequence = read_tracks(tracks_path)
    events = []
    for scene in SCENES:
        info, gt, _ = load_sequence(scene)
        pred = drop_distractor_matches(by_sequence.get(scene, []),
                                       info["gt_all"])
        for event in switch_events(gt, pred):
            event["scene"] = scene
            event["camera"] = info["camera"]
            event["occlusion"] = info["occlusion"]
            event["density"] = info["density"]
            events.append(event)
    events.sort(key=lambda item: (-item["gap"], -item["density"]))
    return events


def pick_failures(events, n=3):
    """
    Três falhas distintas, não as três oclusões mais longas:
      1. miss curto em cena densa (o diagnóstico da correção)
      2. oclusão média (buraco comparável a T=16 / max_age=20)
      3. switch no teste (09 ou 11), se existir
    """

    used_scenes = set()
    chosen = []

    def take(pool, key):
        leftover = [event for event in pool if event["scene"] not in used_scenes]
        leftover.sort(key=key)
        if not leftover:
            return
        event = leftover[0]
        chosen.append(event)
        used_scenes.add(event["scene"])

    short_dense = [e for e in events
                   if e["gap"] <= 5 and e["scene"] in {"02", "04"}]
    mid = [e for e in events if 8 <= e["gap"] <= 24]
    test = [e for e in events if e["scene"] in {"09", "11"} and e["gap"] <= 40]
    take(short_dense, key=lambda e: (-e["density"], e["gap"]))
    take(mid, key=lambda e: (-e["gap"], -e["density"]))
    take(test, key=lambda e: (abs(e["gap"] - 16), -e["density"]))
    if len(chosen) < n:
        take(events, key=lambda e: (-min(e["gap"], 40), -e["density"]))
    return chosen[:n]


def _color(track_id):
    cmap = plt.get_cmap("tab20")
    return cmap(int(track_id) % 20)


def _focus_box(rows, identity):
    for row in rows:
        if int(row[1]) == int(identity):
            return row[2], row[3], row[4], row[5]
    return None


def _crop_view(image, boxes, pad=90):
    height, width = image.shape[:2]
    valid = [box for box in boxes if box is not None]
    if not valid:
        return image, (0, 0)
    x0 = max(0, int(min(box[0] for box in valid) - pad))
    y0 = max(0, int(min(box[1] for box in valid) - pad))
    x1 = min(width, int(max(box[0] + box[2] for box in valid) + pad))
    y1 = min(height, int(max(box[1] + box[3] for box in valid) + pad))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return image, (0, 0)
    return image[y0:y1, x0:x1], (x0, y0)


# pad amplo o bastante para caber o vizinho que roubou o id
_CROP_PAD = 160


def _draw_boxes(ax, image, rows, title, highlight=None, origin=(0, 0)):
    ax.imshow(image)
    ox, oy = origin
    for frame, identity, x, y, w, h in rows:
        thick = 2.4 if highlight is not None and int(identity) == int(highlight) else 1.2
        ax.add_patch(Rectangle((x - ox, y - oy), w, h, fill=False,
                               edgecolor=_color(identity), linewidth=thick))
        ax.text(x - ox, y - oy - 4, str(identity), color="white", fontsize=7,
                bbox=dict(facecolor=_color(identity), edgecolor="none", pad=1))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def render_failure(event, pred_rows, encoder, device, path):
    info, gt, raw = load_sequence(event["scene"])
    dets = public_detections(raw)
    cache = cache_sequence_embeddings(event["scene"], encoder, kind="det",
                                      device=device)
    center = event["frame"]
    frames = [t for t in range(center - 3, center + 4)
              if 1 <= t <= info["n_frames"]]
    fig, axes = plt.subplots(3, len(frames), figsize=(2.6 * len(frames), 7.6),
                             constrained_layout=True)
    if len(frames) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    for col, t in enumerate(frames):
        image = load_frame(info["path"], t, info["im_dir"], info["im_ext"])
        gt_rows = [(r[0], r[1], r[2], r[3], r[4], r[5])
                   for r in gt if r[0] == t and int(r[1]) == int(event["gt_id"])]
        pred_t = [row for row in pred_rows if row[0] == t
                  and int(row[1]) in {int(event["pred_from"]),
                                      int(event["pred_to"])}]
        focus = [
            _focus_box(gt_rows, event["gt_id"]),
            _focus_box(pred_t, event["pred_from"]),
            _focus_box(pred_t, event["pred_to"]),
        ]
        crop, origin = _crop_view(image, focus, pad=_CROP_PAD)
        _draw_boxes(axes[0, col], crop, gt_rows, f"gt {t}",
                    highlight=event["gt_id"], origin=origin)
        _draw_boxes(axes[1, col], crop, pred_t, f"pred {t}",
                    highlight=event["pred_to"], origin=origin)

        frame_dets = [row for row in dets if row[0] == t]
        if len(frame_dets) >= 2:
            start = next(i for i, row in enumerate(dets) if row[0] == t)
            vectors = [cache[(t, start + i)] for i in range(len(frame_dets))]
            sim = 1.0 - cosine_cost(vectors, vectors)
            axes[2, col].imshow(sim, vmin=0, vmax=1, cmap="magma")
        else:
            axes[2, col].imshow(np.zeros((1, 1)), vmin=0, vmax=1, cmap="magma")
        axes[2, col].set_title("cosseno dets", fontsize=8)
        axes[2, col].axis("off")

    fig.suptitle(
        f"MOT17-{event['scene']}  gt={event['gt_id']}  "
        f"id {event['pred_from']}→{event['pred_to']}  "
        f"gap={event['gap']}  quadro {event['frame']}",
        fontsize=11)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def diagnose(event):
    if event["scene"] in {"04", "02"} and event["gap"] <= 5:
        return (
            f"Cena {event['scene']} (densidade {event['density']:.1f}): depois "
            f"de um miss de {event['gap']} quadro(s) a aparência casou o gt "
            f"{event['gt_id']} com o id {event['pred_to']} em vez de "
            f"{event['pred_from']}. Recortes de pedestres próximos são "
            f"parecidos; o cosseno ganha do IoU da última caixa."
        )
    if event["gap"] >= 8:
        return (
            f"Cena {event['scene']}: o gt {event['gt_id']} some {event['gap']} "
            f"quadros e volta com outro id. A janela de BPTT é T=16 e max_age=20; "
            f"se o buraco passa do que o estado da GRU ainda distingue, a track "
            f"morre ou a aparência pega outra pessoa."
        )
    return (
        f"Cena {event['scene']} ({event['camera']}): switch do gt "
        f"{event['gt_id']} no quadro {event['frame']} após gap {event['gap']}."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tracks", default=None)
    parser.add_argument("--n-failures", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="failures")
    args = parser.parse_args()

    import torch

    tracks_path = Path(args.tracks) if args.tracks else RESULTS / "input_appearance_seed42_tracks.csv"
    if not tracks_path.exists():
        tracks_path = RESULTS / "temporal_tracks.csv"
    events = collect_events(tracks_path)
    chosen = pick_failures(events, args.n_failures)
    if not chosen:
        raise SystemExit("nenhum ID switch nas trajetórias")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or str(
        Path(__file__).resolve().parents[1] / "checkpoints" / "input_appearance_seed42_best.pt")
    encoder = CropEncoder(embed_dim=128, freeze=True).to(device)
    if Path(checkpoint).exists():
        _, payload = load_checkpoint(checkpoint, device=device)
        if "encoder_proj" in payload:
            encoder.proj.load_state_dict(payload["encoder_proj"])

    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    by_sequence = read_tracks(tracks_path)
    report = []

    for index, event in enumerate(chosen, start=1):
        event["diagnosis"] = diagnose(event)
        fig_path = FIGURES / f"{args.name}_{index}.png"
        render_failure(event, by_sequence[event["scene"]], encoder, device,
                       fig_path)
        event["figure"] = str(fig_path)
        report.append(event)
        print(f"[{index}] {event['scene']} gt={event['gt_id']} "
              f"gap={event['gap']}  {fig_path}")
        print(f"    {event['diagnosis']}")

    with open(RESULTS / f"{args.name}.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"gravado {RESULTS / f'{args.name}.json'}")


if __name__ == "__main__":
    main()
