import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.dataset import DSB2018Dataset
from src.model import UNet
from src.losses import CrossEntropyLoss


# ============================================================
# Configuração
# ============================================================

DATA_ROOT = "data/raw"
CHECKPOINT_DIR = Path("checkpoints")

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

    n_total = len(dataset)

    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator
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
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
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
    # Checkpoint directory
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

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

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_path = CHECKPOINT_DIR / "baseline_best.pt"

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


if __name__ == "__main__":
    main()