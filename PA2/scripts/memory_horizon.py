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
from src.metrics import drop_distractor_matches, occlusion_survival
from src.losses import PredictiveContrastiveLoss
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
    dataset = TrackWindowDataset(TRAIN_SCENES, window=16, stride=32,
                                 min_visibility=0.25)
    # Janelas sorteadas de todas as cenas de treino (semente fixa): na ordem
    # natural as primeiras 12 eram todas da 02.
    order = np.random.default_rng(0).permutation(len(dataset))
    profiles = {}

    for ckpt in checkpoints:
        model, payload = load_checkpoint(ckpt, device=device)
        if "encoder_proj" in payload:
            encoder.proj.load_state_dict(payload["encoder_proj"])
        # A mesma perda do treino: preditiva no modelo novo, a antiga nos
        # checkpoints sem cabeça de consulta.
        loss_fn = (PredictiveContrastiveLoss()
                   if getattr(model, "has_query", False) else None)
        curves = []
        for index in order:
            item = dataset[int(index)]
            window = _window_from_item(item, caches[item["scene"]])
            if len(window) < 2:
                continue
            curve = gradient_norm_profile(model, window, loss_fn=loss_fn)
            if curve:
                curves.append(curve)
            if len(curves) >= n_windows:
                break
        if not curves:
            continue
        width = max(len(curve) for curve in curves)
        padded = np.array([curve + [np.nan] * (width - len(curve))
                           for curve in curves])
        label = ("antigo (contrastiva)" if "legacy_contrastive" in str(ckpt)
                 else f"final ({Path(ckpt).stem})")
        profiles[label] = np.nanmean(padded, axis=0)

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


def empirical(tracks_path, name, baseline_path=None):
    """
    Quantos quadros a identidade atravessa: sobrevivência em cada oclusão
    (src.metrics.occlusion_survival, a mesma definição do Eixo 3), contra a
    distribuição de duração das oclusões do dataset e contra o baseline sem
    memória nas mesmas oclusões.
    """

    runs = {"modelo": tracks_path}
    if baseline_path and Path(baseline_path).exists():
        runs["baseline"] = baseline_path

    events = {label: [] for label in runs}
    for label, path in runs.items():
        by_sequence = _read_tracks(path)
        for scene in SCENES:
            info, gt, _ = load_sequence(scene)
            pred = drop_distractor_matches(by_sequence.get(scene, []),
                                           info["gt_all"])
            events[label].extend(occlusion_survival(gt, pred))

    occlusions = [e["length"] for e in events["modelo"]]
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cutoff = 60
    bins = range(1, cutoff + 2)
    ax.hist([length for length in occlusions if length <= cutoff], bins=bins,
            alpha=0.35, color="gray", label="oclusões no gt")
    for label, color in (("baseline", "tab:blue"), ("modelo", "tab:green")):
        if label in events:
            ax.hist([e["length"] for e in events[label]
                     if e["survived"] and e["length"] <= cutoff], bins=bins,
                    histtype="step", linewidth=1.8, color=color,
                    label=f"id sobreviveu ({label})")
    ax.set_xlabel("duração (quadros, visibility < 0,25; cauda > 60 omitida)")
    ax.set_ylabel("contagem")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = FIGURES / f"{name}_empirical.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    summary = {"median_occlusion": float(np.median(occlusions)) if occlusions else 0.0,
               "n_occlusions": len(occlusions)}
    for label, items in events.items():
        scorable = [e for e in items if e["scorable"]]
        survived = [e["length"] for e in scorable if e["survived"]]
        summary[label] = {
            "n_scorable": len(scorable),
            "n_survived": len(survived),
            "survive_rate": len(survived) / len(scorable) if scorable else 0.0,
            "median_survived": float(np.median(survived)) if survived else 0.0,
            "p90_survived": float(np.percentile(survived, 90)) if survived else 0.0,
            "max_survived": int(max(survived)) if survived else 0,
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{name}_empirical.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"empírico  {summary['n_occlusions']} oclusões, mediana "
          f"{summary['median_occlusion']:.0f} quadros")
    for label in events:
        row = summary[label]
        print(f"  {label:<8} sobreviveram {row['n_survived']}/{row['n_scorable']} "
              f"({100 * row['survive_rate']:.0f}% das comparáveis)  mediana "
              f"{row['median_survived']:.0f}, p90 {row['p90_survived']:.0f}, "
              f"máx {row['max_survived']}")
    print(f"gravado {fig_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["analytic", "empirical", "both"],
                        default="both")
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--tracks", default=None)
    parser.add_argument("--baseline", default=str(RESULTS / "baseline_tracks.csv"),
                        help="baseline sem memória, nas mesmas oclusões")
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="memory_horizon")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # O modelo final e, se o backup existir, o antigo (perda contrastiva que
    # a GRU vencia copiando o primeiro recorte): as duas curvas lado a lado.
    from src.inference import resolve_checkpoint

    legacy = ROOT / "checkpoints" / "legacy_contrastive" / "temporal_best.pt"
    checkpoints = args.checkpoints or (
        [str(resolve_checkpoint(None))] + ([str(legacy)] if legacy.exists() else []))
    tracks = args.tracks or str(RESULTS / "temporal_tracks.csv")
    if not Path(tracks).exists():
        tracks = str(RESULTS / "temporal_tracks.csv")

    if args.mode in {"analytic", "both"}:
        analytic(checkpoints, args.name, device)
    if args.mode in {"empirical", "both"}:
        empirical(tracks, args.name, baseline_path=args.baseline)


if __name__ == "__main__":
    main()
