import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import DSB2018Dataset, create_splits
from src.model import (
    UNet,
    UNetNoSkips,
    UNetASPP,
)
from src.losses import (
    BoundaryCrossEntropyLoss,
    FocalLoss,
)


DATA_ROOT = "data/raw"

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("experiments/results")

DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-3

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# Fixed so every ablation uses exactly the same
# train / validation / test partition.
SPLIT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--architecture",
        type=str,
        default="unet",
        choices=[
            "unet",
            "no_skips",
            "aspp",
        ],
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=[
            "ce",
            "balanced_ce",
            "focal",
        ],
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Gamma used by focal loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed. Does NOT change the dataset split.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="boundary_experiment",
        help="Name used for checkpoint and history files.",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    architecture,
    device,
):
    if architecture == "unet":
        model = UNet(
            in_channels=3,
            out_channels=3,
        )

    elif architecture == "no_skips":
        model = UNetNoSkips(
            in_channels=3,
            out_channels=3,
        )

    elif architecture == "aspp":
        model = UNetASPP(
            in_channels=3,
            out_channels=3,
        )

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}"
        )

    return model.to(device)


def build_loss(
    loss_name,
    gamma,
    device,
):
    if loss_name == "ce":
        return BoundaryCrossEntropyLoss()

    elif loss_name == "balanced_ce":
        # Hand-defined class weights:
        # 0 = background
        # 1 = interior
        # 2 = boundary
        class_weights = torch.tensor(
            [
                1.0,
                1.0,
                2.0,
            ],
            dtype=torch.float32,
            device=device,
        )

        return BoundaryCrossEntropyLoss(
            class_weights=class_weights
        )

    elif loss_name == "focal":
        return FocalLoss(
            gamma=gamma
        )

    else:
        raise ValueError(
            f"Unknown loss: {loss_name}"
        )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0

    for batch in loader:
        images = batch[
            "image"
        ].to(device)

        targets = batch[
            "boundary_target"
        ].to(device)

        optimizer.zero_grad()

        predictions = model(
            images
        )

        loss = criterion(
            predictions,
            targets
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch[
                "image"
            ].to(device)

            targets = batch[
                "boundary_target"
            ].to(device)

            predictions = model(
                images
            )

            loss = criterion(
                predictions,
                targets
            )

            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    args = parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Print experiment configuration
    # --------------------------------------------------------

    print("=" * 60)
    print("BOUNDARY TRAINING")
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    print(
        f"Experiment: {args.name}"
    )

    print(
        f"Architecture: {args.architecture}"
    )

    print(
        f"Loss: {args.loss}"
    )

    if args.loss == "focal":
        print(
            f"Gamma: {args.gamma}"
        )

    print(
        f"Training seed: {args.seed}"
    )

    print(
        f"Split seed: {SPLIT_SEED}"
    )

    print(
        f"Epochs: {args.epochs}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"Learning rate: {args.lr}"
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = DSB2018Dataset(
        DATA_ROOT
    )

    train_dataset, val_dataset, _ = create_splits(
        dataset,
        seed=SPLIT_SEED,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    print()

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model(
        args.architecture,
        device,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = build_loss(
        args.loss,
        args.gamma,
        device,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{args.name}_best.pt"
    )

    history_path = (
        RESULTS_DIR
        / f"{args.name}_training_history.csv"
    )

    # --------------------------------------------------------
    # Initialize training history
    # --------------------------------------------------------

    with open(
        history_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
            ]
        )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    best_epoch = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{args.epochs} | "
            f"train_loss="
            f"{train_loss:.4f} | "
            f"val_loss="
            f"{val_loss:.4f}"
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        with open(
            history_path,
            "a",
            newline="",
        ) as file:
            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_loss,
                ]
            )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:
            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_loss":
                        val_loss,

                    "architecture":
                        args.architecture,

                    "seed":
                        args.seed,

                    "split_seed":
                        SPLIT_SEED,

                    "loss":
                        args.loss,

                    "gamma":
                        args.gamma,

                    "learning_rate":
                        args.lr,

                    "batch_size":
                        args.batch_size,

                    "experiment_name":
                        args.name,
                },
                checkpoint_path,
            )

            print(
                f"  -> saved "
                f"{checkpoint_path}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "TRAINING FINISHED"
    )

    print(
        "=" * 60
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"History: "
        f"{history_path}"
    )


if __name__ == "__main__":
    main()
