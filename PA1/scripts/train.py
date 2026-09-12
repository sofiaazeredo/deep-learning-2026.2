import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.dataset import DSB2018Dataset, create_splits
from src.model import UNet
from src.losses import CrossEntropyLoss


# ============================================================
# Configuração
# ============================================================

DATA_ROOT = "data/raw"
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("experiments/results")

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42


# ============================================================
# Reprodutibilidade
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Treinamento
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0

    for batch in loader:

        images = batch["image"].to(device)
        masks = batch["instance_mask"].to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(predictions, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# Validação
# ============================================================

def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)
            masks = batch["instance_mask"].to(device)

            predictions = model(images)

            loss = criterion(predictions, masks)

            total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = DSB2018Dataset(DATA_ROOT)

    print(f"Total samples: {len(dataset)}")

    # --------------------------------------------------------
    # Train / validation / test split
    # --------------------------------------------------------

    train_dataset, val_dataset, test_dataset = create_splits(
                                                        dataset,
                                                        seed=SEED,
                                                        train_ratio=TRAIN_RATIO,
                                                        val_ratio=VAL_RATIO
                                                        )

    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=2
    ).to(device)

    criterion = CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Diretórios
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Arquivo de histórico
    # --------------------------------------------------------

    history_path = RESULTS_DIR / "training_history.csv"

    with open(history_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss"
        ])

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):

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

        # ----------------------------------------------------
        # Salva histórico
        # ----------------------------------------------------

        with open(history_path, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                epoch,
                train_loss,
                val_loss
            ])

        # ----------------------------------------------------
        # Salva melhor checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_path = (
                CHECKPOINT_DIR / "baseline_best.pt"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "seed": SEED,
                },
                checkpoint_path
            )

            print(
                f"  -> saved {checkpoint_path}"
            )

    print()
    print(f"Training history: {history_path}")
    print(f"Best checkpoint: {CHECKPOINT_DIR / 'baseline_best.pt'}")


if __name__ == "__main__":
    main()