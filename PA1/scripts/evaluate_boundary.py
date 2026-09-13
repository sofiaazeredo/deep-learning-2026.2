import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    DSB2018Dataset,
    create_splits
)
from src.model import load_model_from_checkpoint
from src.metrics import (
    instance_scores,
    counting_error,
    IOU_THRESHOLDS,
)
from src.postprocessing import (
    boundary_prediction_to_instances
)


DATA_ROOT = "data/raw"

CHECKPOINT_PATH = (
    "checkpoints/boundary_best.pt"
)

RESULTS_DIR = Path(
    "experiments/results"
)

SEED = 42
BATCH_SIZE = 1

SPLIT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True
    )

    return parser.parse_args()

def main():

    args = parse_args()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = DSB2018Dataset(
        DATA_ROOT
    )

    _, _, test_dataset = create_splits(
        dataset,
        seed=SPLIT_SEED,
        train_ratio=0.8,
        val_ratio=0.1
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, checkpoint = load_model_from_checkpoint(
        args.checkpoint,
        device,
        in_channels=3,
        out_channels=3
    )

    architecture = checkpoint.get(
        "architecture",
        "unet"
    )

    print(
        f"Architecture: {architecture}"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint['val_loss']}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULTS_DIR
        / f"{args.name}_test_metrics.csv"
    )

    rows = []

    map_scores = []
    count_errors = []

    threshold_scores = {
        float(t): []
        for t in IOU_THRESHOLDS
    }

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.no_grad():

        for idx, batch in enumerate(
            test_loader
        ):

            images = batch[
                "image"
            ].to(device)

            true_instances = batch[
                "instance_mask"
            ]

            output = model(
                images
            )

            # Softmax não é estritamente necessário
            # para argmax, mas deixa a interface clara
            probs = torch.softmax(
                output,
                dim=1
            )

            pred_instances = (
                boundary_prediction_to_instances(
                    probs[0]
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

            # Uma única matriz de IoU por imagem, reaproveitada em
            # todos os limiares (antes: 20 matrizes por imagem).
            image_map, ap_by_threshold = instance_scores(
                true_np,
                pred_instances
            )

            count_error = counting_error(
                true_np,
                pred_instances
            )

            map_scores.append(
                image_map
            )

            count_errors.append(
                count_error
            )

            per_threshold = {}

            for threshold in IOU_THRESHOLDS:

                score = ap_by_threshold[
                    float(threshold)
                ]

                threshold_scores[
                    float(threshold)
                ].append(score)

                per_threshold[
                    f"ap_{threshold:.2f}"
                ] = score

            true_count = (
                len(
                    np.unique(
                        true_np
                    )
                )
                - 1
            )

            pred_count = (
                len(
                    np.unique(
                        pred_instances
                    )
                )
                - 1
            )

            row = {
                "image_index": idx,
                "image_path":
                    batch["image_path"][0],
                "architecture":
                    architecture,
                "true_count":
                    true_count,
                "pred_count":
                    pred_count,
                "count_error":
                    count_error,
                "count_bias":
                    pred_count - true_count,
                "object_density":
                    true_count,
                "map_50_95":
                    image_map,
            }

            row.update(
                per_threshold
            )

            rows.append(
                row
            )

            print(
                f"[{idx + 1:02d}/"
                f"{len(test_dataset)}] "
                f"mAP={image_map:.3f} "
                f"GT={true_count} "
                f"Pred={pred_count} "
                f"error={count_error}"
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    # --------------------------------------------------------
    # Aggregate results
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print(
        "BOUNDARY + WATERSHED RESULTS"
    )
    print("=" * 50)

    print(
        f"mAP@0.50:0.95: "
        f"{np.mean(map_scores):.4f}"
    )

    print(
        f"Mean counting error: "
        f"{np.mean(count_errors):.4f}"
    )

    count_biases = [
        row["count_bias"]
        for row in rows
    ]

    print(
        f"Mean count bias: "
        f"{np.mean(count_biases):.4f}"
    )

    print()
    print(
        "Per-IoU-threshold scores:"
    )

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
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
