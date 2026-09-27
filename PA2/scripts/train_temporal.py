"""
Parte 2 — treino do modelo temporal.

A fonte de detecções está congelada daqui em diante; o que muda é o que
acontece entre os quadros. Treina em trajetórias do ground truth, em janelas de
T quadros (BPTT truncado).

--track a  : RNN como modelo de movimento (caixa prevista, perda smooth-L1 ou
             NLL gaussiana)
--track b  : RNN como memória de aparência (embedding agregado, perda triplet
             ou contrastiva)
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
from src.dataset import SCENES, TRAIN_SCENES, TrackWindowDataset
from src.losses import ContrastiveIdentityLoss, TripletIdentityLoss
from src.model import AppearanceRNN, CELLS, save_checkpoint
from src.training import REGIMES, truncated_bptt

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def window_from_cache(item, cache):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["a", "b"], default="b")
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
        raise SystemExit("Parte 2 implementa só a Trilha B (--track b)")
    if args.bidirectional:
        raise SystemExit("bidirecional é o Eixo 4 da Parte 3")

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  cell={args.cell}  window={args.window}  "
          f"epochs={args.epochs}", flush=True)

    encoder = CropEncoder(embed_dim=args.embed_dim, freeze=True).to(device)
    caches = {}
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

    model = AppearanceRNN(cell=args.cell, embed_dim=args.embed_dim,
                          hidden=args.hidden).to(device)
    loss_fn = (ContrastiveIdentityLoss() if args.loss == "contrastive"
               else TripletIdentityLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    clip = None if args.clip_grad == 0 else args.clip_grad
    kwargs = {"cell": args.cell, "embed_dim": args.embed_dim,
              "hidden": args.hidden}

    best = float("inf")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        order = np.random.permutation(len(dataset))
        running = []

        for index in order:
            item = dataset[int(index)]
            window = window_from_cache(item, caches[item["scene"]])
            if len(window) < 2:
                continue
            loss = truncated_bptt(model, window, loss_fn, args.regime,
                                  clip_grad=clip, optimizer=optimizer)
            running.append(float(loss.detach().cpu()))

        mean = float(np.mean(running)) if running else float("nan")
        print(f"epoch {epoch + 1:03d}/{args.epochs}  loss={mean:.4f}",
              flush=True)

        if mean < best:
            best = mean
            save_checkpoint(CHECKPOINTS / f"{args.name}_best.pt", model, kwargs,
                            extra={"epoch": epoch, "loss": best,
                                   "track": "b", "window": args.window,
                                   "encoder_proj": encoder.proj.state_dict()})

    save_checkpoint(CHECKPOINTS / f"{args.name}_last.pt", model, kwargs,
                    extra={"epoch": args.epochs, "loss": mean, "track": "b",
                           "window": args.window,
                           "encoder_proj": encoder.proj.state_dict()})

    payload = {
        "checkpoint": str(CHECKPOINTS / f"{args.name}_best.pt"),
        "track": "b",
        "cell": args.cell,
        "window": args.window,
        "loss": args.loss,
        "seed": args.seed,
        "best_loss": best,
    }
    with open(RESULTS / "best_model.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"melhor loss={best:.4f}  {payload['checkpoint']}")


if __name__ == "__main__":
    main()
