from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import DSB2018Dataset, create_splits
from src.model import UNet
from src.metrics import instance_map, counting_error
from src.postprocessing import (
    boundary_prediction_to_instances
)


DATA_ROOT = "data/raw"
CHECKPOINT = "checkpoints/boundary_best.pt"

SEED = 42


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # -------------------------------
    # Validation data
    # -------------------------------

    dataset = DSB2018Dataset(DATA_ROOT)

    _, val_dataset, _ = create_splits(
        dataset,
        seed=SEED,
        train_ratio=0.8,
        val_ratio=0.1
    )

    loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )

    # -------------------------------
    # Model
    # -------------------------------

    model = UNet(
        in_channels=3,
        out_channels=3
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # -------------------------------
    # Cache predictions
    # -------------------------------

    cached = []

    print("Caching validation predictions...")

    with torch.no_grad():

        for batch in loader:

            image = batch["image"].to(device)

            logits = model(image)

            probs = torch.softmax(
                logits,
                dim=1
            )[0].cpu()

            gt = (
                batch["instance_mask"][0]
                .numpy()
            )

            cached.append(
                (probs, gt)
            )

    print(
        f"Cached {len(cached)} images."
    )

    # -------------------------------
    # Parameter grid
    # -------------------------------

    interior_thresholds = [
        0.30,
        0.40,
        0.50,
        0.60
    ]

    foreground_thresholds = [
        0.40,
        0.50,
        0.60
    ]

    min_sizes = [
        0,
        3,
        5,
        10
    ]

    results = []

    for (
        interior_threshold,
        foreground_threshold,
        min_size
    ) in product(
        interior_thresholds,
        foreground_thresholds,
        min_sizes
    ):

        maps = []
        errors = []
        biases = []

        for probs, gt in cached:

            pred = (
                boundary_prediction_to_instances(
                    probs,
                    interior_threshold=
                        interior_threshold,
                    foreground_threshold=
                        foreground_threshold,
                    min_size=min_size
                )
            )

            maps.append(
                instance_map(gt, pred)
            )

            errors.append(
                counting_error(gt, pred)
            )

            true_count = (
                len(np.unique(gt)) - 1
            )

            pred_count = (
                len(np.unique(pred)) - 1
            )

            biases.append(
                pred_count - true_count
            )

        mean_map = np.mean(maps)
        mean_error = np.mean(errors)
        mean_bias = np.mean(biases)

        results.append({
            "interior_threshold":
                interior_threshold,
            "foreground_threshold":
                foreground_threshold,
            "min_size":
                min_size,
            "map":
                mean_map,
            "count_error":
                mean_error,
            "count_bias":
                mean_bias
        })

        print(
            f"interior={interior_threshold:.2f} "
            f"foreground={foreground_threshold:.2f} "
            f"min_size={min_size:2d} | "
            f"mAP={mean_map:.4f} "
            f"error={mean_error:.2f} "
            f"bias={mean_bias:.2f}"
        )

    # -------------------------------
    # Best by mAP
    # -------------------------------

    results.sort(
        key=lambda x: x["map"],
        reverse=True
    )

    print()
    print("=" * 60)
    print("TOP 10 BY VALIDATION mAP")
    print("=" * 60)

    for result in results[:10]:
        print(result)


if __name__ == "__main__":
    main()
