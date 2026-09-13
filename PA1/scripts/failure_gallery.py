"""
Parte 5 — galeria de falhas.

Seleciona as N imagens de teste em que o modelo final erra mais feio e
monta, para cada uma: imagem, ground truth, previsão e o MAPA
INTERMEDIÁRIO relevante (a probabilidade de fronteira, que é a
representação da Trilha A).

Além das figuras, imprime e salva os números que sustentam o
diagnóstico de cada falha — o PDF pede um diagnóstico concreto, do tipo
"o objeto tem X px e o campo receptivo é Y px", e para escrever isso é
preciso ter as medidas da imagem em mãos:

  - quantos núcleos existem e quantos foram previstos
  - o diâmetro mediano dos núcleos daquela imagem
  - a fração de pixels de núcleo que encostam em outro núcleo
    (o quanto a imagem é "grudada")
  - contraste e brilho médios
  - quantos núcleos do GT foram fundidos num único previsto
    (sub-segmentação) e quantos previstos partiram um GT
    (super-segmentação)

O diagnóstico em si é de vocês; o script entrega a evidência.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataset import DSB2018Dataset, create_splits
from src.model import load_model_from_checkpoint
from src.metrics import instance_scores, instance_iou_matrix
from src.postprocessing import boundary_prediction_to_instances


DATA_ROOT = "data/raw"
RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("experiments/figures/failures")

SPLIT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n-failures", type=int, default=5)
    parser.add_argument("--name", type=str, default="failures")

    return parser.parse_args()


def resolve_checkpoint(explicit):

    if explicit is not None:
        return explicit

    decision_path = RESULTS_DIR / "best_architecture.json"

    if not decision_path.exists():
        raise SystemExit(
            "Sem --checkpoint e sem experiments/results/best_architecture.json. "
            "Rode scripts/summarize_resolution_ablation.py antes, ou passe "
            "--checkpoint explicitamente."
        )

    with open(decision_path) as file:
        return json.load(file)["checkpoint"]


def contact_fraction(labels):
    """
    Fração dos pixels de núcleo que têm, na vizinhança-4, um pixel de
    OUTRO núcleo. É a medida de "o quanto esta imagem está grudada".
    """

    foreground = labels > 0

    if not foreground.any():
        return 0.0

    touching = np.zeros_like(foreground)

    for shift_axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):

        neighbour = np.roll(labels, shift, axis=shift_axis)

        touching |= foreground & (neighbour > 0) & (neighbour != labels)

    return float(touching.sum() / foreground.sum())


def median_diameter(labels):

    ids, areas = np.unique(labels, return_counts=True)

    areas = areas[ids != 0]

    if len(areas) == 0:
        return 0.0

    return float(np.median(2.0 * np.sqrt(areas / np.pi)))


def merge_split_counts(true_labels, pred_labels, overlap=0.5):
    """
    Quantifica os dois modos de falha de instância.

    merged  - núcleos do GT que dividem o MESMO ID previsto
              (o modelo colou núcleos encostados)
    split   - núcleos do GT partidos em mais de um ID previsto

    Um núcleo do GT conta para um ID previsto quando pelo menos
    `overlap` da sua área cai dentro dele.
    """

    true_ids = np.unique(true_labels)
    true_ids = true_ids[true_ids != 0]

    owner = {}
    split = 0

    for true_id in true_ids:

        mask = true_labels == true_id
        area = mask.sum()

        pred_ids, counts = np.unique(pred_labels[mask], return_counts=True)

        keep = pred_ids != 0

        pred_ids = pred_ids[keep]
        counts = counts[keep]

        if len(pred_ids) == 0:
            continue

        # Quantos IDs previstos cobrem uma fatia relevante deste núcleo?
        significant = (counts / area) >= 0.2

        if significant.sum() > 1:
            split += 1

        dominant = pred_ids[np.argmax(counts)]

        if counts.max() / area >= overlap:
            owner.setdefault(dominant, []).append(true_id)

    merged = sum(
        len(group)
        for group in owner.values()
        if len(group) > 1
    )

    return merged, split


def main():

    args = parse_args()

    checkpoint_path = resolve_checkpoint(args.checkpoint)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        device,
        in_channels=3,
        out_channels=3,
    )

    architecture = checkpoint.get("architecture", "unet")

    print(f"Architecture: {architecture}")

    dataset = DSB2018Dataset(DATA_ROOT)

    _, _, test_dataset = create_splits(
        dataset,
        seed=SPLIT_SEED,
        train_ratio=0.8,
        val_ratio=0.1,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    records = []

    print()
    print("Avaliando o conjunto de teste...")

    with torch.no_grad():

        for index, batch in enumerate(loader):

            images = batch["image"].to(device)

            true_labels = batch["instance_mask"][0].numpy().astype(np.int32)

            probs = torch.softmax(model(images), dim=1)[0].cpu().numpy()

            pred_labels = boundary_prediction_to_instances(probs)

            image_map, _ = instance_scores(true_labels, pred_labels)

            n_true = int(len(np.unique(true_labels)) - 1)
            n_pred = int(len(np.unique(pred_labels)) - 1)

            merged, split = merge_split_counts(true_labels, pred_labels)

            image_np = images[0].cpu().numpy().transpose(1, 2, 0)

            records.append({
                "index": index,
                "image_path": batch["image_path"][0],
                "map_50_95": image_map,
                "n_true": n_true,
                "n_pred": n_pred,
                "count_error": abs(n_pred - n_true),
                "median_diameter_px": median_diameter(true_labels),
                "contact_fraction": contact_fraction(true_labels),
                "mean_intensity": float(image_np.mean()),
                "contrast_std": float(image_np.std()),
                "gt_merged": merged,
                "gt_split": split,
                "image": image_np,
                "true_labels": true_labels,
                "pred_labels": pred_labels,
                "boundary_prob": probs[2],
            })

    records.sort(key=lambda r: r["map_50_95"])

    worst = records[:args.n_failures]

    # ----------------------------------------------------------------
    # Tabela de evidências
    # ----------------------------------------------------------------

    print()
    print("=" * 78)
    print(f"PARTE 5 — AS {args.n_failures} PIORES IMAGENS DO TESTE")
    print("=" * 78)

    header = (
        f"{'#':<3}{'mAP':>7}{'GT':>5}{'pred':>6}{'diam':>7}"
        f"{'grudado':>9}{'brilho':>8}{'fundidos':>10}{'partidos':>10}"
    )

    print(header)

    for position, record in enumerate(worst, start=1):
        print(
            f"{position:<3}{record['map_50_95']:>7.3f}"
            f"{record['n_true']:>5}{record['n_pred']:>6}"
            f"{record['median_diameter_px']:>7.1f}"
            f"{record['contact_fraction']:>8.0%}"
            f"{record['mean_intensity']:>8.2f}"
            f"{record['gt_merged']:>10}{record['gt_split']:>10}"
        )

    median_all = np.median([r["map_50_95"] for r in records])

    print()
    print(f"mAP mediano do teste inteiro: {median_all:.3f}")
    print(f"'grudado' = % dos pixels de núcleo que encostam em outro núcleo")
    print(f"'fundidos' = núcleos do GT colados num mesmo ID previsto")
    print(f"'partidos' = núcleos do GT quebrados em vários IDs previstos")

    # ----------------------------------------------------------------
    # Figuras
    # ----------------------------------------------------------------

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for position, record in enumerate(worst, start=1):

        fig, axes = plt.subplots(1, 4, figsize=(19, 5))

        axes[0].imshow(record["image"])
        axes[0].set_title("imagem")

        axes[1].imshow(record["true_labels"], cmap="nipy_spectral")
        axes[1].set_title(f"ground truth ({record['n_true']} núcleos)")

        axes[2].imshow(record["pred_labels"], cmap="nipy_spectral")
        axes[2].set_title(f"previsão ({record['n_pred']} núcleos)")

        boundary = axes[3].imshow(record["boundary_prob"], cmap="magma")
        axes[3].set_title("mapa intermediário: P(fronteira)")

        fig.colorbar(boundary, ax=axes[3], fraction=0.046)

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"Falha {position} — mAP={record['map_50_95']:.3f} · "
            f"{record['n_true']} núcleos, {record['contact_fraction']:.0%} "
            f"de pixels encostados · {record['gt_merged']} fundidos, "
            f"{record['gt_split']} partidos"
        )

        fig.tight_layout()

        path = FIGURES_DIR / f"{args.name}_{position}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"Salvo: {path}")

    # ----------------------------------------------------------------
    # CSV com TODAS as imagens (para escolher outras falhas se quiser)
    # ----------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / f"{args.name}_diagnostics.csv"

    fields = [
        "index", "image_path", "map_50_95", "n_true", "n_pred",
        "count_error", "median_diameter_px", "contact_fraction",
        "mean_intensity", "contrast_std", "gt_merged", "gt_split",
    ]

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for record in records:
            writer.writerow({key: record[key] for key in fields})

    print(f"Salvo: {csv_path}")

    # ----------------------------------------------------------------
    # Correlações que apontam a causa dominante
    # ----------------------------------------------------------------

    maps = np.array([r["map_50_95"] for r in records])

    print()
    print("Correlação (Pearson) do mAP por imagem com:")

    for key, label in [
        ("contact_fraction", "fração de pixels encostados"),
        ("n_true", "número de núcleos"),
        ("median_diameter_px", "diâmetro mediano"),
        ("mean_intensity", "brilho médio"),
        ("contrast_std", "contraste"),
    ]:
        values = np.array([r[key] for r in records], dtype=float)

        if values.std() == 0:
            continue

        correlation = float(np.corrcoef(maps, values)[0, 1])

        print(f"  {label:<32} {correlation:+.3f}")


if __name__ == "__main__":
    main()
