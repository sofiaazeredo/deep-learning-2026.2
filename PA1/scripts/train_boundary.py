import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import DSB2018Dataset, create_splits
from src.model import UNet
from src.losses import BoundaryCrossEntropyLoss


DATA_ROOT = "data/raw"

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("experiments/results")

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0

    for batch in loader:

        images = batch["image"].to(device)

        targets = batch[
            "boundary_target"
        ].to(device)

        optimizer.zero_grad()

        predictions = model(images)

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
    device
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

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    dataset = DSB2018Dataset(
        DATA_ROOT
    )

    train_dataset, val_dataset, _ = create_splits(
        dataset,
        seed=SEED,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO
    )

    print(
        f"Train: {len(train_dataset)}"
    )

    print(
        f"Validation: {len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    model = UNet(
        in_channels=3,
        out_channels=3
    ).to(device)

    criterion = BoundaryCrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    history_path = (
        RESULTS_DIR
        / "boundary_training_history.csv"
    )

    with open(
        history_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss"
        ])

    best_val_loss = float("inf")

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        with open(
            history_path,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                epoch,
                train_loss,
                val_loss
            ])

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_path = (
                CHECKPOINT_DIR
                / "boundary_best.pt"
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "seed": SEED,
                },
                checkpoint_path
            )

            print(
                f"  -> saved "
                f"{checkpoint_path}"
            )

    print()
    print(
        f"Training history: "
        f"{history_path}"
    )

    print(
        f"Best checkpoint: "
        f"{CHECKPOINT_DIR / 'boundary_best.pt'}"
    )


if __name__ == "__main__":
    main()
