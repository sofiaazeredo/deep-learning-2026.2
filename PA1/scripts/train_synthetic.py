"""
Parte 0 — teste unitário sintético.

"Antes de tocar em dados reais, gerem um dataset sintético próprio (...)
É o teste unitário de vocês: mostrem que treina em menos de 5 minutos e
reportem a métrica."

Treina o MESMO encoder-decoder e a MESMA perda dos dados reais sobre
elipses sintéticas, decodifica com o MESMO watershed e mede com a MESMA
métrica de instância. Se qualquer peça do pipeline estiver quebrada,
quebra aqui — em minutos, e não depois de uma hora de treino no DSB2018.

A rede usa `--base-channels 16` por padrão (1,9M de parâmetros, contra
31M da rede do trabalho) justamente para caber no orçamento de 5
minutos. Passe `--base-channels 64` para rodar a rede cheia.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.synthetic import SyntheticEllipses, touching_pairs
from src.model import UNet
from src.losses import BoundaryCrossEntropyLoss
from src.metrics import instance_scores
from src.postprocessing import boundary_prediction_to_instances


RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("experiments/figures/synthetic")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-val", type=int, default=64)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="synthetic")

    return parser.parse_args()


def describe_dataset(dataset, n_check=64):
    """
    Confere que o dataset cumpre o enunciado: 5 a 20 objetos por imagem
    e "muitas delas se tocando".
    """

    counts = []
    touching = []

    for index in range(min(n_check, len(dataset))):

        labels = dataset[index]["instance_mask"].numpy().astype(np.int32)

        n_objects = int(labels.max())

        pairs = touching_pairs(labels)

        counts.append(n_objects)
        touching.append(len({i for pair in pairs for i in pair}))

    counts = np.array(counts)
    touching = np.array(touching)

    print(f"  objetos por imagem: média {counts.mean():.1f} "
          f"(min {counts.min()}, max {counts.max()})")

    print(f"  objetos encostados em pelo menos um vizinho: "
          f"{100 * touching.sum() / counts.sum():.1f}%")

    return counts


@torch.no_grad()
def evaluate(model, dataset, device):
    """
    mAP de instância e erro de contagem no split sintético de validação.
    """

    model.eval()

    maps = []
    count_errors = []

    for index in range(len(dataset)):

        sample = dataset[index]

        image = sample["image"].unsqueeze(0).to(device)

        true_labels = sample["instance_mask"].numpy().astype(np.int32)

        probs = torch.softmax(model(image), dim=1)[0].cpu().numpy()

        pred_labels = boundary_prediction_to_instances(probs)

        image_map, _ = instance_scores(true_labels, pred_labels)

        maps.append(image_map)

        count_errors.append(
            abs(
                (len(np.unique(pred_labels)) - 1)
                - (len(np.unique(true_labels)) - 1)
            )
        )

    return float(np.mean(maps)), float(np.mean(count_errors))


def main():

    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Sem isto o resultado varia muito entre execuções: as convoluções do
    # cuDNN escolhem algoritmos não determinísticos, e num treino curto
    # isso chega a virar um colapso (o modelo para de prever a classe
    # interior, o watershed fica sem marcador e o mAP vai a zero).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 66)
    print("PARTE 0 — TESTE UNITÁRIO SINTÉTICO")
    print("=" * 66)
    print(f"Device: {device}")
    print(f"Rede: UNet base_channels={args.base_channels}")
    print()

    train_dataset = SyntheticEllipses(
        n_samples=args.n_train,
        size=args.size,
        seed=args.seed,
    )

    val_dataset = SyntheticEllipses(
        n_samples=args.n_val,
        size=args.size,
        seed=args.seed + 1000,      # imagens diferentes das de treino
    )

    print(f"Treino: {len(train_dataset)} imagens {args.size}x{args.size}")
    describe_dataset(train_dataset)
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    print(f"Parâmetros: {n_params / 1e6:.2f}M")
    print()

    criterion = BoundaryCrossEntropyLoss(
        class_weights=torch.tensor(
            [1.0, 1.0, 2.0],
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ----------------------------------------------------------------
    # Treino cronometrado
    # ----------------------------------------------------------------

    started = time.time()

    history = []

    for epoch in range(1, args.epochs + 1):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            images = batch["image"].to(device)
            targets = batch["boundary_target"].to(device)

            optimizer.zero_grad()

            loss = criterion(model(images), targets)

            loss.backward()

            # Sem clipping este treino diverge: a perda vira NaN por volta
            # da epoca 10 e o modelo passa a prever 100% fundo, zerando o
            # mAP. Foi o teste unitario que pegou isso.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        if not np.isfinite(train_loss):
            raise SystemExit(
                f"FALHA NO TESTE UNITARIO: a perda virou {train_loss} na "
                f"epoca {epoch}. O treino divergiu, e qualquer metrica "
                f"reportada depois disso seria lixo."
            )

        elapsed = time.time() - started

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | {elapsed:6.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "elapsed_s": elapsed,
        })

    training_time = time.time() - started

    # ----------------------------------------------------------------
    # Métrica
    # ----------------------------------------------------------------

    print()
    print("Avaliando no split sintético de validação...")

    mean_ap, count_error = evaluate(model, val_dataset, device)

    print()
    print("=" * 66)
    print(f"TEMPO DE TREINO: {training_time:.1f}s "
          f"({training_time / 60:.2f} min) "
          f"{'DENTRO' if training_time < 300 else 'ACIMA'} do limite de 5 min")
    print(f"mAP@0,50:0,95 (sintético): {mean_ap:.4f}")
    print(f"Erro absoluto de contagem: {count_error:.2f}")
    print("=" * 66)

    # ----------------------------------------------------------------
    # Figura
    # ----------------------------------------------------------------

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model.eval()

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    with torch.no_grad():

        for column in range(4):

            sample = val_dataset[column]

            image = sample["image"].unsqueeze(0).to(device)

            true_labels = sample["instance_mask"].numpy().astype(np.int32)

            probs = torch.softmax(model(image), dim=1)[0].cpu().numpy()

            pred_labels = boundary_prediction_to_instances(probs)

            image_map, _ = instance_scores(true_labels, pred_labels)

            axes[0, column].imshow(
                sample["image"][0].numpy(), cmap="gray"
            )
            axes[0, column].set_title(f"imagem #{column}")

            axes[1, column].imshow(true_labels, cmap="nipy_spectral")
            axes[1, column].set_title(
                f"ground truth ({int(true_labels.max())})"
            )

            axes[2, column].imshow(pred_labels, cmap="nipy_spectral")
            axes[2, column].set_title(
                f"previsão ({int(pred_labels.max())}) mAP={image_map:.2f}"
            )

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        f"Parte 0 — teste unitário sintético · treino em "
        f"{training_time:.0f}s · mAP {mean_ap:.3f}"
    )

    fig.tight_layout()

    figure_path = FIGURES_DIR / f"{args.name}.png"
    fig.savefig(figure_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------
    # CSV
    # ----------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    history_path = RESULTS_DIR / f"{args.name}_training_history.csv"

    with open(history_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["epoch", "train_loss", "elapsed_s"]
        )
        writer.writeheader()
        writer.writerows(history)

    summary_path = RESULTS_DIR / f"{args.name}_summary.csv"

    with open(summary_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["training_time_s", round(training_time, 1)])
        writer.writerow(["map_50_95", mean_ap])
        writer.writerow(["count_abs_error", count_error])
        writer.writerow(["n_train", args.n_train])
        writer.writerow(["n_val", args.n_val])
        writer.writerow(["epochs", args.epochs])
        writer.writerow(["base_channels", args.base_channels])
        writer.writerow(["params_millions", round(n_params / 1e6, 2)])

    print()
    print(f"Salvo: {figure_path}")
    print(f"Salvo: {history_path}")
    print(f"Salvo: {summary_path}")


if __name__ == "__main__":
    main()
