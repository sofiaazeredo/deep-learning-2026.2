"""
Rastreamento com o modelo temporal da Parte 2, sobre a mesma fonte de
detecções congelada do baseline. Sai no mesmo formato de run_baseline.py, para
as métricas ficarem lado a lado.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json

import torch

from src.appearance import CropEncoder, cache_sequence_embeddings
from src.dataset import SCENES, TEST_SCENES, TRAIN_SCENES, load_sequence
from src.detection import public_detections
from src.model import load_checkpoint
from src.tracker import Tracker

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def resolve_checkpoint(path):
    """
    --checkpoint ou o modelo final de best_model.json, resolvido como na
    inferência (caminho relativo à raiz do PA2, ou o nome em checkpoints/
    se o caminho gravado for de outra máquina).
    """

    from src.inference import resolve_checkpoint as resolve

    if not path and not (RESULTS / "best_model.json").exists():
        raise SystemExit("passe --checkpoint ou grave experiments/results/best_model.json")
    return resolve(path or None)


def embeddings_for_detections(dets, cache):
    vectors = []
    for index, row in enumerate(dets):
        vectors.append(cache[(int(row[0]), index)])
    return vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="None usa o modelo final de experiments/results/best_model.json")
    parser.add_argument("--split", default="all",
                        choices=["all", "train", "test"])
    parser.add_argument("--window", type=int, default=None,
                        help="inferência em janelas; None processa a sequência inteira")
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--max-age", type=int, default=20)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--appearance-threshold", type=float, default=0.5)
    parser.add_argument("--miss-prefer-iou", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="IoU da última caixa nas tracks perdidas antes do "
                             "cosseno (padrão; --no-miss-prefer-iou desliga)")
    parser.add_argument("--prefer-confirmed", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="correção da Parte 4: tentativas casam depois das "
                             "confirmadas (desligada = modelo das Partes 2 e 3)")
    parser.add_argument("--gate-growth", type=float, default=0.6,
                        help="portão: a última caixa cresce isto por quadro perdido")
    parser.add_argument("--gate-iou", type=float, default=0.1,
                        help="portão: IoU mínimo com a caixa aumentada")
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    if args.window is not None:
        print("aviso: --window é discussão da Parte 2, a inferência roda a "
              "sequência inteira")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint(args.checkpoint)
    model, payload = load_checkpoint(checkpoint, device=device)
    kind = getattr(model, "kind", payload.get("input", "appearance"))
    embed_dim = payload.get("kwargs", {}).get("embed_dim", 128)
    encoder = None
    if kind in {"appearance", "both"}:
        encoder = CropEncoder(embed_dim=embed_dim, freeze=True).to(device)
        if "encoder_proj" in payload:
            encoder.proj.load_state_dict(payload["encoder_proj"])

    if args.split == "train":
        scenes = list(TRAIN_SCENES)
    elif args.split == "test":
        scenes = list(TEST_SCENES)
    else:
        scenes = list(SCENES)

    rows = []

    for scene in scenes:
        info, _, raw = load_sequence(scene, detector="SDP")
        dets = public_detections(raw)
        embeddings = None
        if encoder is not None:
            print(f"cache det {scene}...")
            cache = cache_sequence_embeddings(scene, encoder, kind="det",
                                              device=device)
            embeddings = embeddings_for_detections(dets, cache)
        tracker = Tracker(iou_threshold=args.iou_threshold,
                          max_age=args.max_age, min_hits=args.min_hits,
                          matcher="hungarian", motion=model,
                          appearance_threshold=args.appearance_threshold,
                          image_size=(info["im_width"], info["im_height"]),
                          miss_prefer_iou=args.miss_prefer_iou,
                          gate_growth=args.gate_growth,
                          gate_iou=args.gate_iou,
                          prefer_confirmed=args.prefer_confirmed)
        tracks = tracker.run(dets, embeddings=embeddings,
                             image_size=(info["im_width"], info["im_height"]))
        print(f"{scene}  {info['camera']:<7}  {len(tracks)} caixas  "
              f"{len({row[1] for row in tracks})} ids")
        for frame, identity, x, y, w, h in tracks:
            rows.append((scene, frame, identity, x, y, w, h))

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.name}_tracks.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "frame", "id", "x", "y", "w", "h"])
        writer.writerows(rows)
    print(f"gravado {path}")


if __name__ == "__main__":
    main()
