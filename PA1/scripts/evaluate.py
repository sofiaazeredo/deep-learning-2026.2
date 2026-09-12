import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import DSB2018Dataset, create_splits
from src.model import UNet
from src.metrics import (
    iou_score,
    dice_score,
    instance_iou_matrix,
    match_instances,
    instance_precision,
    instance_map,
    counting_error,
    IOU_THRESHOLDS,
)
from src.postprocessing import semantic_prediction_to_instances


DATA_ROOT = "data/raw"
CHECKPOINT_PATH = "checkpoints/baseline_best.pt"
RESULTS_DIR = Path("experiments/results")

SEED = 42
BATCH_SIZE = 1


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    dataset = DSB2018Dataset(DATA_ROOT)

    _, _, test_dataset = create_splits(
        dataset,
        seed=SEED,
        train_ratio=0.8,
        val_ratio=0.1
    )

    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = UNet(
        in_channels=3,
        out_channels=2
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = RESULTS_DIR / "baseline_test_metrics.csv"

    rows = []

    semantic_ious = []
    semantic_dices = []

    map_scores = []
    count_errors = []

    threshold_scores = {
        float(t): []
        for t in IOU_THRESHOLDS
    }

    with torch.no_grad():

        for idx, batch in enumerate(test_loader):

            images = batch["image"].to(device)
            true_instances = batch[
                "instance_mask"
            ].to(device)

            output = model(images)

            # ----------------------------------------
            # Semantic metrics
            # ----------------------------------------

            sem_iou = iou_score(
                output,
                true_instances
            )

            sem_dice = dice_score(
                output,
                true_instances
            )

            semantic_ious.append(sem_iou)
            semantic_dices.append(sem_dice)

            # ----------------------------------------
            # Convert output to foreground probability
            # ----------------------------------------

            probs = torch.softmax(
                output,
                dim=1
            )

            foreground_prob = probs[
                0, 1
            ]

            pred_instances = (
                semantic_prediction_to_instances(
                    foreground_prob,
                    threshold=0.5
                )
            )

            true_np = (
                true_instances[0]
                .cpu()
                .numpy()
            )

            # ----------------------------------------
            # Instance metrics
            # ----------------------------------------

            image_map = instance_map(
                true_np,
                pred_instances
            )

            count_error = counting_error(
                true_np,
                pred_instances
            )

            map_scores.append(image_map)
            count_errors.append(count_error)

            # ----------------------------------------
            # Per-threshold scores
            # ----------------------------------------

            per_threshold = {}

            for threshold in IOU_THRESHOLDS:

                score = instance_precision(
                    true_np,
                    pred_instances,
                    threshold
                )

                threshold_scores[
                    float(threshold)
                ].append(score)

                per_threshold[
                    f"ap_{threshold:.2f}"
                ] = score

            # ----------------------------------------
            # Counts
            # ----------------------------------------

            true_count = int(
                len(np.unique(true_np)) - 1
            )

            pred_count = int(
                len(np.unique(pred_instances)) - 1
            )

            density = true_count / (
                true_np.shape[0]
                * true_np.shape[1]
            )

            row = {
                "image_index": idx,
                "image_path": batch[
                    "image_path"
                ][0],
                "semantic_iou": sem_iou,
                "semantic_dice": sem_dice,
                "true_count": true_count,
                "pred_count": pred_count,
                "count_error": count_error,
                "object_density": density,
                "map_50_95": image_map,
            }

            row.update(per_threshold)

            rows.append(row)

            print(
                f"[{idx + 1:02d}/{len(test_dataset)}] "
                f"IoU={sem_iou:.3f} "
                f"Dice={sem_dice:.3f} "
                f"mAP={image_map:.3f} "
                f"count_error={count_error}"
            )

    # --------------------------------------------
    # Save CSV
    # --------------------------------------------

    fieldnames = list(rows[0].keys())

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------
    # Aggregate results
    # --------------------------------------------

    print()
    print("=" * 50)
    print("BASELINE TEST RESULTS")
    print("=" * 50)

    print(
        f"Semantic IoU: "
        f"{np.mean(semantic_ious):.4f}"
    )

    print(
        f"Semantic Dice: "
        f"{np.mean(semantic_dices):.4f}"
    )

    print(
        f"mAP@0.50:0.95: "
        f"{np.mean(map_scores):.4f}"
    )

    print(
        f"Mean counting error: "
        f"{np.mean(count_errors):.4f}"
    )

    print()
    print("Per-IoU-threshold scores:")

    for threshold in IOU_THRESHOLDS:

        values = threshold_scores[
            float(threshold)
        ]

        print(
            f"AP@{threshold:.2f}: "
            f"{np.mean(values):.4f}"
        )

    print()
    print(
        f"Saved results to: {output_path}"
    )


if __name__ == "__main__":
    main()
