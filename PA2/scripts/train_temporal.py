"""
Parte 2 — treino do modelo temporal. Parte 3 — Eixo 3 (--input).

A fonte de detecções está congelada daqui em diante; o que muda é o que
acontece entre os quadros. Treina em trajetórias do ground truth, em janelas de
T quadros (BPTT truncado).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random

import numpy as np
import torch

from src.appearance import CropEncoder, cache_sequence_embeddings
from src.boxes import xywh_to_norm
from src.dataset import SCENES, TRAIN_SCENES, TrackWindowDataset
from src.losses import ContrastiveIdentityLoss, SmoothL1BoxLoss, TripletIdentityLoss
from src.model import CELLS, build_model, save_checkpoint
from src.training import REGIMES, truncated_bptt

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def window_appearance(item, cache):
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


def window_geometry(item):
    width, height = item["im_width"], item["im_height"]
    tensors = {}
    for identity, steps in item["identities"].items():
        prepared = [(int(frame), torch.as_tensor(xywh_to_norm(box, width, height)))
                    for frame, box in steps]
        if prepared:
            tensors[int(identity)] = prepared
    return tensors


def window_both(item, cache):
    width, height = item["im_width"], item["im_height"]
    tensors = {}
    for identity, steps in item["identities"].items():
        prepared = []
        for frame, box in steps:
            vector = cache.get((int(frame), int(identity)))
            if vector is None:
                continue
            feat = np.concatenate([np.asarray(vector, dtype=np.float32),
                                   xywh_to_norm(box, width, height)])
            prepared.append((int(frame), torch.as_tensor(feat)))
        if prepared:
            tensors[int(identity)] = prepared
    return tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["a", "b"], default="b")
    parser.add_argument("--input", choices=["appearance", "geometry", "both"],
                        default="appearance")
    parser.add_argument("--cell", choices=list(CELLS), default="gru")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8,
                        help="passo entre janelas (1 = máximo overlap)")
    parser.add_argument("--regime", choices=list(REGIMES),
                        default="teacher_forcing")
    parser.add_argument("--clip-grad", type=float, default=1.0,
                        help="0 desliga o clipping (Eixo 2)")
    parser.add_argument("--bidirectional", action="store_true",
                        help="modo offline do Eixo 4")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--loss", choices=["contrastive", "triplet"],
                        default="contrastive")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    if args.track != "b":
        raise SystemExit("Parte 2 implementa só a Trilha B (--track b); "
                         "geometria entra pelo --input do Eixo 3")
    if args.bidirectional:
        raise SystemExit("bidirecional é o Eixo 4 da Parte 3")

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  input={args.input}  cell={args.cell}  "
          f"window={args.window}  epochs={args.epochs}", flush=True)

    encoder = None
    caches = {}
    needs_cache = args.input in {"appearance", "both"}
    if needs_cache:
        encoder = CropEncoder(embed_dim=args.embed_dim, freeze=True).to(device)
        for scene in TRAIN_SCENES:
            print(f"cache GT {scene}...", flush=True)
            caches[scene] = cache_sequence_embeddings(
                scene, encoder, kind="gt", device=device)
        for scene in SCENES:
            print(f"cache det {scene}...", flush=True)
            cache_sequence_embeddings(scene, encoder, kind="det", device=device)

    dataset = TrackWindowDataset(TRAIN_SCENES, window=args.window,
                                 stride=args.stride)
    print(f"{len(dataset)} janelas de treino", flush=True)

    model_name = {"appearance": "appearance", "geometry": "motion",
                  "both": "fusion"}[args.input]
    kwargs = {"cell": args.cell, "hidden": args.hidden}
    if args.input != "geometry":
        kwargs["embed_dim"] = args.embed_dim
    model = build_model(model_name, **kwargs).to(device)

    loss_fn = None
    box_loss_fn = None
    if args.input in {"appearance", "both"}:
        loss_fn = (ContrastiveIdentityLoss() if args.loss == "contrastive"
                   else TripletIdentityLoss())
    if args.input in {"geometry", "both"}:
        box_loss_fn = SmoothL1BoxLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    clip = None if args.clip_grad == 0 else args.clip_grad

    best = float("inf")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    extra_base = {"track": "b", "input": args.input, "window": args.window,
                  "name": model_name}
    if encoder is not None:
        extra_base["encoder_proj"] = encoder.proj.state_dict()

    for epoch in range(args.epochs):
        order = np.random.permutation(len(dataset))
        running = []

        for index in order:
            item = dataset[int(index)]
            if args.input == "appearance":
                window = window_appearance(item, caches[item["scene"]])
            elif args.input == "geometry":
                window = window_geometry(item)
            else:
                window = window_both(item, caches[item["scene"]])
            if args.input == "geometry":
                if len(window) < 1:
                    continue
            elif len(window) < 2:
                continue
            loss = truncated_bptt(model, window, loss_fn, args.regime,
                                  clip_grad=clip, optimizer=optimizer,
                                  box_loss_fn=box_loss_fn)
            running.append(float(loss.detach().cpu()))

        mean = float(np.mean(running)) if running else float("nan")
        print(f"epoch {epoch + 1:03d}/{args.epochs}  loss={mean:.4f}",
              flush=True)

        if mean < best:
            best = mean
            save_checkpoint(CHECKPOINTS / f"{args.name}_best.pt", model, kwargs,
                            extra={**extra_base, "epoch": epoch, "loss": best})

    save_checkpoint(CHECKPOINTS / f"{args.name}_last.pt", model, kwargs,
                    extra={**extra_base, "epoch": args.epochs, "loss": mean})

    payload = {
        "checkpoint": str(CHECKPOINTS / f"{args.name}_best.pt"),
        "track": "b",
        "input": args.input,
        "cell": args.cell,
        "window": args.window,
        "loss": args.loss,
        "seed": args.seed,
        "best_loss": best,
    }
    with open(RESULTS / f"{args.name}.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    if not args.name.startswith("input_"):
        with open(RESULTS / "best_model.json", "w") as handle:
            json.dump(payload, handle, indent=2)
    print(f"melhor loss={best:.4f}  {payload['checkpoint']}")


if __name__ == "__main__":
    main()
