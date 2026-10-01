"""
Parte 2 — varredura das regras do estágio de aparência, só no treino.

O limiar do cosseno e o portão geométrico (quanto a última caixa cresce por
quadro perdido e o IoU mínimo com ela) são regras do tracker, não pesos do
modelo: valores diferentes dão números diferentes, então são escolhidos por
IDF1 médio nas cenas de TREINO (02, 04, 05, 10, 13) e congelados. O teste
(09, 11) não entra aqui.

A linha "cosseno desligado" é o controle: o mesmo tracker sem o estágio de
aparência. Se nenhuma configuração passa dela, a memória não está ajudando.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import itertools

import numpy as np
import torch

from src.appearance import CropEncoder, cache_sequence_embeddings
from src.dataset import TRAIN_SCENES, load_sequence
from src.detection import public_detections
from src.metrics import drop_distractor_matches, evaluate_sequence
from src.model import load_checkpoint
from src.tracker import Tracker

from run_tracker import embeddings_for_detections, resolve_checkpoint

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
TRACKER_KW = dict(iou_threshold=0.2, max_age=20, min_hits=2, matcher="hungarian")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="None usa experiments/results/best_model.json")
    parser.add_argument("--appearance-threshold", type=float, nargs="+",
                        default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--gate-growth", type=float, nargs="+",
                        default=[0.1, 0.3, 0.6])
    parser.add_argument("--gate-iou", type=float, nargs="+",
                        default=[0.01, 0.1])
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="tracker_sweep")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(resolve_checkpoint(args.checkpoint),
                                     device=device)
    encoder = CropEncoder(embed_dim=payload.get("kwargs", {}).get("embed_dim", 128),
                          freeze=True).to(device)
    if "encoder_proj" in payload:
        encoder.proj.load_state_dict(payload["encoder_proj"])

    scenes = []
    for scene in TRAIN_SCENES:
        info, gt, raw = load_sequence(scene, detector="SDP")
        dets = public_detections(raw)
        cache = cache_sequence_embeddings(scene, encoder, kind="det", device=device)
        scenes.append((scene, info, gt, dets, embeddings_for_detections(dets, cache)))

    def score(**rule):
        per_scene = []
        for scene, info, gt, dets, embeddings in scenes:
            tracker = Tracker(**TRACKER_KW, motion=model,
                              image_size=(info["im_width"], info["im_height"]),
                              **rule)
            tracks = tracker.run(dets, embeddings=embeddings)
            metrics = evaluate_sequence(
                gt, drop_distractor_matches(tracks, info["gt_all"]))
            per_scene.append((metrics["idf1"], metrics["id_switches"]))
        return (float(np.mean([idf1 for idf1, _ in per_scene])),
                int(sum(switches for _, switches in per_scene)),
                [round(idf1, 4) for idf1, _ in per_scene])

    rows = []
    control = score(appearance_threshold=-1.0)
    rows.append({"appearance_threshold": "off", "gate_growth": "", "gate_iou": "",
                 "idf1_train": control[0], "idsw_train": control[1],
                 **dict(zip(TRAIN_SCENES, control[2]))})
    print(f"cosseno desligado  IDF1 {control[0]:.4f}  IDSW {control[1]}", flush=True)

    for threshold, growth, gate_iou in itertools.product(
            args.appearance_threshold, args.gate_growth, args.gate_iou):
        idf1, switches, per_scene = score(appearance_threshold=threshold,
                                          gate_growth=growth, gate_iou=gate_iou)
        rows.append({"appearance_threshold": threshold, "gate_growth": growth,
                     "gate_iou": gate_iou, "idf1_train": idf1,
                     "idsw_train": switches, **dict(zip(TRAIN_SCENES, per_scene))})
        print(f"limiar {threshold:.2f}  crescimento {growth:.2f}  "
              f"IoU {gate_iou:.2f}  IDF1 {idf1:.4f}  IDSW {switches}"
              f"  {'(> controle)' if idf1 > control[0] else ''}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.name}.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows[1:], key=lambda row: row["idf1_train"])
    print(f"melhor no treino: limiar {best['appearance_threshold']}, "
          f"crescimento {best['gate_growth']}, IoU {best['gate_iou']} -> "
          f"IDF1 {best['idf1_train']:.4f} (controle {control[0]:.4f})")
    print(f"gravado {path}")


if __name__ == "__main__":
    main()
