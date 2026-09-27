"""
Avaliação de trajetórias (Partes 1, 2, 3 e 5).

IDF1, ID switches, fragmentações e erro de contagem de identidades únicas, com
as métricas de src/metrics.py. MOTA é opcional e sai em coluna separada.

Predições casadas a um distractor (IoU >= 0.5) saem antes da métrica, como
no protocolo oficial do MOT17.

Grava um CSV por execução em experiments/results/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
from collections import defaultdict

from src.dataset import load_sequence, resolve_root
from src.detection import public_detections
from src.metrics import (
    average_precision,
    drop_distractor_matches,
    evaluate_sequence,
)

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def read_tracks(path):
    by_sequence = defaultdict(list)

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            by_sequence[row["sequence"]].append((
                int(float(row["frame"])),
                int(float(row["id"])),
                float(row["x"]),
                float(row["y"]),
                float(row["w"]),
                float(row["h"]),
            ))

    return by_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True,
                        help="trajetórias previstas gravadas por run_baseline/run_tracker")
    parser.add_argument("--split", default="all",
                        choices=["all", "train", "test"])
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--detector", default="SDP")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--root", default="data/MOT17")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    root = resolve_root(args.root)
    by_sequence = read_tracks(args.tracks)
    rows = []

    for scene, pred in sorted(by_sequence.items()):
        info, gt, raw = load_sequence(scene, detector=args.detector, root=root)
        pred = drop_distractor_matches(pred, info["distractors"],
                                       threshold=args.iou_threshold)
        metrics = evaluate_sequence(gt, pred, threshold=args.iou_threshold)

        dets = public_detections(raw, score_threshold=args.score_threshold,
                                 nms_threshold=args.nms_threshold)
        det_tracks = [(row[0], i, row[1], row[2], row[3], row[4], row[5])
                      for i, row in enumerate(dets, start=1)]
        det_tracks = drop_distractor_matches(det_tracks, info["distractors"],
                                             threshold=args.iou_threshold)
        detections = [(row[0], row[2], row[3], row[4], row[5], row[6])
                      for row in det_tracks]
        ap, det_details = average_precision(gt, detections,
                                            threshold=args.iou_threshold)

        row = {
            "sequence": scene,
            "split": ("test" if scene in ("09", "11") else "train"),
            "camera": info["camera"],
            "density": info["density"],
            "occlusion": info["occlusion"],
            "n_frames": info["n_frames"],
            "ap": ap,
            "det_recall": det_details["recall"],
            **metrics,
        }
        rows.append(row)
        print(f"{scene}  AP={ap:.3f}  IDF1={metrics['idf1']:.3f}  "
              f"IDsw={metrics['id_switches']}  "
              f"ratio={metrics['id_ratio']:.2f}")

    if args.split in {"train", "test"}:
        rows = [row for row in rows if row["split"] == args.split]

    out = RESULTS / f"{args.name}_per_sequence.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        mean_idf1 = sum(row["idf1"] for row in rows) / len(rows)
        mean_ap = sum(row["ap"] for row in rows) / len(rows)
        switches = sum(row["id_switches"] for row in rows)
        count_err = sum(row["id_count_error"] for row in rows) / len(rows)
        print(f"média  AP={mean_ap:.3f}  IDF1={mean_idf1:.3f}  "
              f"IDsw={switches}  count_err={count_err:.2f}")
        print(f"gravado {out}")


if __name__ == "__main__":
    main()
