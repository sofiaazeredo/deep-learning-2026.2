"""
Parte 5 — teste de estresse, sem retreinar, em cima do modelo final.

--mode detector   detecções SDP degradadas em 3 intensidades (descarte,
                  ruído nas caixas, falsos positivos). O modelo temporal
                  absorve ou amplifica a falha? mAP e IDF1 juntos.
--mode framerate  não é o eixo desta entrega.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.appearance import CropEncoder, cache_sequence_embeddings, embed_detections
from src.dataset import SCENES, TEST_SCENES, load_sequence
from src.detection import public_detections
from src.detector_sim import INTENSITIES, degrade_detections
from src.metrics import average_precision, drop_distractor_matches, evaluate_sequence
from src.model import load_checkpoint
from src.tracker import Tracker

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
FIGURES = Path(__file__).resolve().parents[1] / "experiments" / "figures"
ROOT = Path(__file__).resolve().parents[1]

LEVELS = (
    ("limpo", {}),
    ("leve", INTENSITIES["leve"]),
    ("media", INTENSITIES["media"]),
    ("forte", INTENSITIES["forte"]),
)
TRACKER_KW = dict(iou_threshold=0.2, max_age=20, min_hits=2,
                  matcher="hungarian", appearance_threshold=0.5)


def detections_ap(gt, dets, gt_all):
    rows = [(row[0], index, row[1], row[2], row[3], row[4], row[5])
            for index, row in enumerate(dets, start=1)]
    rows = drop_distractor_matches(rows, gt_all)
    cleaned = [(row[0], row[2], row[3], row[4], row[5], row[6]) for row in rows]
    return average_precision(gt, cleaned)


def run_one(dets, embeddings, info, model=None, miss_prefer_iou=False,
            prefer_confirmed=False):
    tracker = Tracker(**TRACKER_KW, motion=model,
                      image_size=(info["im_width"], info["im_height"]),
                      miss_prefer_iou=miss_prefer_iou,
                      prefer_confirmed=prefer_confirmed)
    tracks = tracker.run(dets, embeddings=embeddings,
                         image_size=(info["im_width"], info["im_height"]))
    return drop_distractor_matches(tracks, info["gt_all"])


def mean_split(rows, field, scenes=None):
    picked = [row for row in rows if scenes is None or row["sequence"] in scenes]
    if not picked:
        return float("nan")
    return float(np.mean([row[field] for row in picked]))


def detector_stress(checkpoint, name, device, seed):
    model, payload = load_checkpoint(checkpoint, device=device)
    encoder = CropEncoder(
        embed_dim=payload.get("kwargs", {}).get("embed_dim", 128),
        freeze=True).to(device)
    if "encoder_proj" in payload:
        encoder.proj.load_state_dict(payload["encoder_proj"])

    rows = []
    for scene in SCENES:
        info, gt, raw = load_sequence(scene)
        clean = public_detections(raw)
        size = (info["im_width"], info["im_height"])
        cache = cache_sequence_embeddings(scene, encoder, kind="det",
                                          device=device)
        for level, knobs in LEVELS:
            if level == "limpo":
                dets = clean
                embeddings = [cache[(int(row[0]), index)]
                              for index, row in enumerate(dets)]
            else:
                dets = degrade_detections(clean, image_size=size,
                                          seed=seed + int(scene), **knobs)
                print(f"embed {scene} {level}  {len(dets)} caixas...",
                      flush=True)
                embeddings = embed_detections(scene, dets, encoder,
                                              device=device)
            ap, details = detections_ap(gt, dets, info["gt_all"])
            baseline = evaluate_sequence(
                gt, run_one(dets, None, info))
            # O modelo final: Trilha B + a correção da Parte 4.
            temporal = evaluate_sequence(
                gt, run_one(dets, embeddings, info, model=model,
                            miss_prefer_iou=True, prefer_confirmed=True))
            for tracker_name, score in (("baseline", baseline),
                                        ("temporal", temporal)):
                row = {
                    "sequence": scene,
                    "split": "test" if scene in TEST_SCENES else "train",
                    "level": level,
                    "tracker": tracker_name,
                    "ap": ap,
                    "det_recall": details["recall"],
                    "n_dets": len(dets),
                    "idf1": score["idf1"],
                    "id_switches": score["id_switches"],
                    "id_count_error": score["id_count_error"],
                    "mota": score["mota"],
                }
                rows.append(row)
            print(f"{scene} {level:<5}  AP={ap:.3f}  "
                  f"base={baseline['idf1']:.3f}  "
                  f"temp={temporal['idf1']:.3f}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / f"{name}.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for split_name, scenes in (("test", set(TEST_SCENES)), ("all", None)):
        for level, _ in LEVELS:
            chunk = [row for row in rows if row["level"] == level
                     and (scenes is None or row["sequence"] in scenes)]
            base = [row for row in chunk if row["tracker"] == "baseline"]
            temp = [row for row in chunk if row["tracker"] == "temporal"]
            summary_rows.append({
                "split": split_name,
                "level": level,
                "ap": mean_split(base, "ap"),
                "idf1_baseline": mean_split(base, "idf1"),
                "idf1_temporal": mean_split(temp, "idf1"),
                "idsw_baseline": int(sum(row["id_switches"] for row in base)),
                "idsw_temporal": int(sum(row["id_switches"] for row in temp)),
            })

    with open(RESULTS / f"{name}_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    # O quanto cada curva mantém do nível limpo, POR SPLIT: as quedas do
    # teste e das 7 cenas não se misturam numa mesma frase.
    verdict = {}
    for split_name in ("test", "all"):
        clean = next(row for row in summary_rows
                     if row["split"] == split_name and row["level"] == "limpo")
        verdict[split_name] = []
        for row in summary_rows:
            if row["split"] != split_name or row["level"] == "limpo":
                continue
            ap_keep = row["ap"] / clean["ap"] if clean["ap"] else float("nan")
            base_keep = (row["idf1_baseline"] / clean["idf1_baseline"]
                         if clean["idf1_baseline"] else float("nan"))
            temp_keep = (row["idf1_temporal"] / clean["idf1_temporal"]
                         if clean["idf1_temporal"] else float("nan"))
            verdict[split_name].append({
                "level": row["level"],
                "ap_kept": ap_keep,
                "idf1_baseline_kept": base_keep,
                "idf1_temporal_kept": temp_keep,
                "temporal_vs_ap": temp_keep - ap_keep,
                "temporal_vs_baseline": temp_keep - base_keep,
            })

    absorbs = all(item["temporal_vs_baseline"] >= -0.02
                  for item in verdict["test"])
    summary = {
        "question": (
            "O modelo temporal absorve ou amplifica a falha do detector?"
        ),
        "answer_split": "test",
        "answer": (
            "absorve (cai menos que o baseline, e não mais que o AP)"
            if absorbs else
            "amplifica (o IDF1 temporal cai mais que o do baseline)"
        ),
        "levels": verdict,
        "test": [row for row in summary_rows if row["split"] == "test"],
        "all": [row for row in summary_rows if row["split"] == "all"],
    }
    with open(RESULTS / f"{name}.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    order = [level for level, _ in LEVELS]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for split_name, ax in (("test", axes[0]), ("all", axes[1])):
        block = [row for row in summary_rows if row["split"] == split_name]
        block = sorted(block, key=lambda row: order.index(row["level"]))
        xs = range(len(block))
        ax.plot(xs, [row["ap"] for row in block], marker="s", label="AP@0,5")
        ax.plot(xs, [row["idf1_baseline"] for row in block], marker="o",
                label="IDF1 baseline")
        ax.plot(xs, [row["idf1_temporal"] for row in block], marker="^",
                label="IDF1 temporal")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([row["level"] for row in block])
        ax.set_ylim(0, 1)
        ax.set_title("teste 09+11" if split_name == "test" else "as 7 cenas")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("AP / IDF1")
    fig.tight_layout()
    fig_path = FIGURES / f"{name}.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    print(f"resposta (teste)  {summary['answer']}")
    for split_name in ("test", "all"):
        for item in verdict[split_name]:
            print(f"  {split_name:<4} {item['level']:<5}  AP×{item['ap_kept']:.2f}  "
                  f"base×{item['idf1_baseline_kept']:.2f}  "
                  f"temp×{item['idf1_temporal_kept']:.2f}")
    print(f"gravado {csv_path}")
    print(f"gravado {fig_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["framerate", "detector"],
                        required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "framerate":
        raise SystemExit("esta entrega fez o eixo de qualidade do detector")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from src.inference import resolve_checkpoint

    checkpoint = resolve_checkpoint(args.checkpoint or None)
    name = args.name or "stress_detector"
    detector_stress(checkpoint, name, device, args.seed)


if __name__ == "__main__":
    main()
