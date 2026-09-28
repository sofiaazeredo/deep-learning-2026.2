"""
Parte 1 — baseline por quadro.

IoU entre a última caixa observada da track e as detecções do quadro,
matching guloso ou Hungarian, limiar fixo, ID novo quando nada casa,
track morta depois de k quadros sem observação. Grava as trajetórias.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import itertools

from src.dataset import (
    SCENES,
    TRAIN_SCENES,
    create_splits,
    list_scenes,
    load_sequence,
    resolve_root,
)
from src.detection import public_detections
from src.metrics import drop_distractor_matches, evaluate_sequence
from src.tracker import Tracker

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def track_sequence(scene, detector, matcher, iou_threshold, max_age, min_hits,
                   score_threshold, nms_threshold, root):
    info, _, raw = load_sequence(scene, detector=detector, root=root)
    detections = public_detections(raw, score_threshold=score_threshold,
                                   nms_threshold=nms_threshold)
    tracker = Tracker(iou_threshold=iou_threshold, max_age=max_age,
                      min_hits=min_hits, matcher=matcher)
    tracks = tracker.run(detections)
    return info, tracks


def write_tracks(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "frame", "id", "x", "y", "w", "h"])
        writer.writerows(rows)


def mean_idf1(scenes, detector, matcher, iou_threshold, max_age, min_hits,
              score_threshold, nms_threshold, root):
    scores = []

    for scene in scenes:
        info, gt, raw = load_sequence(scene, detector=detector, root=root)
        detections = public_detections(raw, score_threshold=score_threshold,
                                       nms_threshold=nms_threshold)
        tracks = Tracker(iou_threshold=iou_threshold, max_age=max_age,
                         min_hits=min_hits, matcher=matcher).run(detections)
        tracks = drop_distractor_matches(tracks, info["gt_all"])
        scores.append(evaluate_sequence(gt, tracks)["idf1"])

    return sum(scores) / len(scores) if scores else float("nan")


def sweep(args, root):
    scenes = list(TRAIN_SCENES)
    grid = list(itertools.product(
        args.sweep_iou,
        args.sweep_min_hits,
        args.sweep_max_age,
    ))
    rows = []
    best = None

    print(f"sweep no treino {scenes}  detector={args.detector}  "
          f"matcher={args.matcher}  {len(grid)} configs")

    for iou_threshold, min_hits, max_age in grid:
        score = mean_idf1(scenes, args.detector, args.matcher, iou_threshold,
                          max_age, min_hits, args.score_threshold,
                          args.nms_threshold, root)
        row = {"iou_threshold": iou_threshold, "min_hits": min_hits,
               "max_age": max_age, "idf1": score}
        rows.append(row)
        print(f"  iou={iou_threshold:.2f}  min_hits={min_hits}  "
              f"max_age={max_age:<3}  IDF1={score:.4f}")
        if best is None or score > best["idf1"]:
            best = row

    path = RESULTS / f"{args.name}_sweep.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"melhor no treino: {best}")
    print(f"gravado {path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, col, title in zip(
        axes,
        ["iou_threshold", "min_hits", "max_age"],
        ["IoU threshold", "min_hits", "max_age"],
    ):
        grouped = frame.groupby(col)["idf1"].max()
        ax.plot(grouped.index, grouped.values, marker="o")
        ax.set_xlabel(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("melhor IDF1 no treino")
    fig.suptitle("Sweep do baseline nas sequências de treino")
    fig.tight_layout()
    figure = Path(__file__).resolve().parents[1] / "experiments" / "figures" / f"{args.name}_sweep.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    print(f"gravado {figure}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="SDP",
                        choices=["DPM", "FRCNN", "SDP", "torchvision"])
    parser.add_argument("--matcher", default="hungarian",
                        choices=["greedy", "hungarian"])
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--max-age", type=int, default=20)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--split", default="all",
                        choices=["all", "train", "test"])
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-iou", type=float, nargs="+",
                        default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--sweep-min-hits", type=int, nargs="+",
                        default=[1, 2, 3])
    parser.add_argument("--sweep-max-age", type=int, nargs="+",
                        default=[1, 5, 10, 20, 30])
    parser.add_argument("--root", default="data/MOT17")
    args = parser.parse_args()

    root = resolve_root(args.root)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        sweep(args, root)
        return

    available = list_scenes(root, detector=args.detector)
    splits = create_splits(available)
    if args.split == "all":
        scenes = [scene for scene in SCENES if scene in available]
    else:
        scenes = splits[args.split]

    rows = []

    for scene in scenes:
        info, tracks = track_sequence(
            scene, args.detector, args.matcher, args.iou_threshold,
            args.max_age, args.min_hits, args.score_threshold,
            args.nms_threshold, root)
        print(f"{scene}  {info['camera']:<7}  {len(tracks)} caixas  "
              f"{len({row[1] for row in tracks})} ids")
        for frame, identity, x, y, w, h in tracks:
            rows.append((scene, frame, identity, x, y, w, h))

    path = RESULTS / f"{args.name}_tracks.csv"
    write_tracks(path, rows)
    print(f"gravado {path}")


if __name__ == "__main__":
    main()
