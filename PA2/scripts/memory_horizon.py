"""
Parte 4 — horizonte de memória efetivo (obrigatório), das duas formas.

1. Analítica: a norma de dL_t/dh_{t-k} em função de k, no nosso modelo e nos
   nossos dados. É a curva de gradiente que some dos slides.
2. Empírica: quantos quadros o estado sobrevive a uma oclusão antes de a track
   morrer ou trocar de ID, comparado com a distribuição de duração de oclusão
   do dataset.
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
import torch

from src.appearance import CropEncoder, cache_sequence_embeddings
from src.dataset import (
    SCENES,
    TRAIN_SCENES,
    TrackWindowDataset,
    load_sequence,
    occlusion_intervals,
)
from src.metrics import _clear_mot_matches, drop_distractor_matches
from src.model import load_checkpoint
from src.training import gradient_norm_profile

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
FIGURES = Path(__file__).resolve().parents[1] / "experiments" / "figures"
ROOT = Path(__file__).resolve().parents[1]


def _window_from_item(item, cache):
    tensors = {}
    for identity, steps in item["identities"].items():
        prepared = []
        for frame, _ in steps:
            vector = cache.get((int(frame), int(identity)))
            if vector is None:
                continue
            prepared.append((int(frame), torch.as_tensor(vector)))
        if prepared:
            tensors[int(identity)] = prepared
    return tensors


def analytic(checkpoints, name, device, n_windows=12):
    encoder = CropEncoder(embed_dim=128, freeze=True).to(device)
    caches = {}
    for scene in TRAIN_SCENES:
        caches[scene] = cache_sequence_embeddings(scene, encoder, kind="gt",
                                                  device=device)
    dataset = TrackWindowDataset(TRAIN_SCENES, window=16, stride=32)
    profiles = {}

    for ckpt in checkpoints:
        model, payload = load_checkpoint(ckpt, device=device)
        if "encoder_proj" in payload:
            encoder.proj.load_state_dict(payload["encoder_proj"])
        curves = []
        for item in dataset:
            window = _window_from_item(item, caches[item["scene"]])
            if len(window) < 2:
                continue
            curve = gradient_norm_profile(model, window)
            if curve:
                curves.append(curve)
            if len(curves) >= n_windows:
                break
        if not curves:
            continue
        width = max(len(curve) for curve in curves)
        padded = np.array([curve + [np.nan] * (width - len(curve))
                           for curve in curves])
        profiles[Path(ckpt).stem] = np.nanmean(padded, axis=0)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for label, curve in profiles.items():
        scale = curve[0] if curve[0] else 1.0
        ax.plot(range(len(curve)), curve / scale, marker="o", label=label)
    ax.set_xlabel("k (passos no passado)")
    ax.set_ylabel(r"$||\partial L_t / \partial h_{t-k}||$  (rel. a $k=0$)")
    ax.set_ylim(0, 1.15)
    ax.legend()
    fig.tight_layout()
    fig_path = FIGURES / f"{name}_analytic.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    rows = []
    for label, curve in profiles.items():
        for k, value in enumerate(curve):
            rows.append({"checkpoint": label, "k": k, "grad_norm": float(value)})
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{name}_analytic.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint", "k", "grad_norm"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"analítico  { {k: [round(v, 4) for v in curve[:6]] for k, curve in profiles.items()} }")
    print(f"gravado {fig_path}")
    return profiles


def _read_tracks(path):
    by_sequence = defaultdict(list)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            by_sequence[str(row["sequence"]).zfill(2)].append((
                int(float(row["frame"])), int(float(row["id"])),
                float(row["x"]), float(row["y"]),
                float(row["w"]), float(row["h"]),
            ))
    return by_sequence


def empirical(tracks_path, name):
    by_sequence = _read_tracks(tracks_path)
    occlusions = []
    survived = []

    for scene in SCENES:
        info, gt, _ = load_sequence(scene)
        pred = drop_distractor_matches(by_sequence.get(scene, []),
                                       info["gt_all"])
        _, matches = _clear_mot_matches(gt, pred, 0.5)
        intervals = occlusion_intervals(gt)
        for gt_id, items in intervals.items():
            for start, end, length in items:
                occlusions.append(length)
                before = None
                after = None
                for frame in range(start - 1, 0, -1):
                    if gt_id in matches.get(frame, {}):
                        before = matches[frame][gt_id]
                        break
                for frame in range(end + 1, info["n_frames"] + 1):
                    if gt_id in matches.get(frame, {}):
                        after = matches[frame][gt_id]
                        break
                if before is not None and before == after:
                    survived.append(length)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    cutoff = 60
    bins = range(1, cutoff + 2)
    ax.hist([length for length in occlusions if length <= cutoff], bins=bins,
            alpha=0.5, label="oclusões no gt")
    ax.hist([length for length in survived if length <= cutoff], bins=bins,
            alpha=0.7, label="id sobreviveu")
    ax.set_xlabel("duração (quadros, visibility < 0,25; cauda > 60 omitida)")
    ax.set_ylabel("contagem")
    ax.legend()
    fig.tight_layout()
    fig_path = FIGURES / f"{name}_empirical.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    summary = {
        "n_occlusions": len(occlusions),
        "n_survived": len(survived),
        "survive_rate": len(survived) / len(occlusions) if occlusions else 0.0,
        "median_occlusion": float(np.median(occlusions)) if occlusions else 0.0,
        "median_survived": float(np.median(survived)) if survived else 0.0,
        "max_survived": int(max(survived)) if survived else 0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{name}_empirical.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"empírico  {summary['n_survived']}/{summary['n_occlusions']} "
          f"sobreviveram  mediana oclusão={summary['median_occlusion']:.1f}  "
          f"mediana sobrevivida={summary['median_survived']:.1f}")
    print(f"gravado {fig_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["analytic", "empirical", "both"],
                        default="both")
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--tracks", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="memory_horizon")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    default_ckpt = ROOT / "checkpoints" / "input_appearance_seed42_best.pt"
    checkpoints = args.checkpoints or ([str(default_ckpt)] if default_ckpt.exists()
                                       else [str(ROOT / "checkpoints" / "temporal_best.pt")])
    tracks = args.tracks or str(RESULTS / "input_appearance_seed42_tracks.csv")
    if not Path(tracks).exists():
        tracks = str(RESULTS / "temporal_tracks.csv")

    if args.mode in {"analytic", "both"}:
        analytic(checkpoints, args.name, device)
    if args.mode in {"empirical", "both"}:
        empirical(tracks, args.name)


if __name__ == "__main__":
    main()
