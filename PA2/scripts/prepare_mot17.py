"""
Prepara o MOT17: confere o layout de data/MOT17, lista as sequências, imprime
o split por sequência e a estatística que justifica o critério (densidade,
câmera parada vs. móvel, duração de oclusão).

Roda só com o pacote de anotações (~10 MB) — não precisa dos quadros.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv

from src.dataset import (
    DETECTION_SOURCES,
    SCENES,
    create_splits,
    list_scenes,
    load_sequence,
    resolve_root,
    split_report,
)
from src.detection import public_detections
from src.metrics import average_precision, drop_distractor_matches

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def measure_detectors(root, score_threshold=None, nms_threshold=0.5):
    """
    AP@0.5 e recall de cada fonte, no ranking inteiro (sem limiar de score:
    o AP é quem decide até onde o ranking vale). As três públicas e o Faster
    R-CNN do torchvision, se o cache existir.
    """

    rows = []
    sources = [source for source in DETECTION_SOURCES
               if source != "torchvision" or list_scenes(root, detector=source)]

    print("\nAP@0.5 e recall das fontes de detecção (depois do nosso NMS)")
    print(f"{'scene':<8} {'det':<11} {'AP@0.5':>8} {'recall':>8} {'n_pred':>8}")

    for scene in SCENES:
        for detector in sources:
            info, gt, raw = load_sequence(scene, detector=detector, root=root)
            dets = public_detections(raw, score_threshold=score_threshold,
                                     nms_threshold=nms_threshold)
            # AP ignora id. Distratores não entram no GT; detecção que casa
            # com eles some (não é FP), como no protocolo MOT17.
            det_as_tracks = [(row[0], i, row[1], row[2], row[3], row[4], row[5])
                             for i, row in enumerate(dets, start=1)]
            scored = drop_distractor_matches(det_as_tracks, info["gt_all"])
            detections = [(row[0], row[2], row[3], row[4], row[5], row[6])
                          for row in scored]
            ap, details = average_precision(gt, detections)
            rows.append({
                "scene": scene,
                "detector": detector,
                "camera": info["camera"],
                "ap": ap,
                "recall": details["recall"],
                "n_gt": details["n_gt"],
                "n_pred": details["n_pred"],
                "density": info["density"],
                "occlusion": info["occlusion"],
            })
            print(f"{scene:<8} {detector:<11} {ap:>8.3f} {details['recall']:>8.3f} "
                  f"{details['n_pred']:>8}")

    path = RESULTS / "detector_measurement.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMédias por detector:")
    for detector in sources:
        subset = [row for row in rows if row["detector"] == detector]
        mean_ap = sum(row["ap"] for row in subset) / len(subset)
        mean_rec = sum(row["recall"] for row in subset) / len(subset)
        print(f"  {detector:<11} AP@0.5={mean_ap:.3f}  recall={mean_rec:.3f}")

    print(f"gravado {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/MOT17")
    parser.add_argument("--detector", default="SDP")
    parser.add_argument("--measure", action="store_true",
                        help="AP@0.5 e recall de DPM, FRCNN, SDP e torchvision nas 7 cenas")
    args = parser.parse_args()

    root = resolve_root(args.root)
    train = root / "train"

    if not train.exists():
        raise SystemExit(f"não achei {train}. Extraia MOT17Labels.zip em {root}.")

    scenes = list_scenes(root, detector=args.detector)
    print(f"root={root}")
    print(f"cenas ({args.detector}): {', '.join(scenes)}")

    splits = create_splits(scenes)
    split_report(splits, detector=args.detector, root=root)

    if args.measure:
        measure_detectors(root)


if __name__ == "__main__":
    main()
