"""
Parte 5 — a correção, com antes/depois.

Diagnóstico escolhido (o da falha 1 da galeria):

    O alvo de 3 classes é gerado em src/dataset.py com
    binary_erosion(mask, iterations=2). Um núcleo de 4 px de diâmetro
    perde 2 px de cada lado e fica SEM NENHUM pixel de interior: 100%
    dos núcleos com diâmetro <= 4 px e 29.8% dos de 4-8 px caem nesse
    caso. Sem interior no alvo, a rede aprende a marcar esses núcleos
    inteiros como fronteira; na decodificação o watershed não recebe
    marcador e o objeto desaparece da saída — ainda que o mapa de
    fronteira acerte a posição dele. Na pior imagem do teste isso zera
    a previsão: 78 núcleos no ground truth, 0 previstos, mAP = 0.

Correção implementada (sem retreinar):

    Todo componente conexo de foreground que não contém marcador de
    interior recebe um marcador no seu ponto mais interno (máximo da
    transformada de distância). É exatamente o que o diagnóstico
    sugere: o problema não é a rede não enxergar o núcleo, é a
    decodificação não ter semente para ele.

Este script roda o teste inteiro com as duas decodificações e reporta
mAP antes e depois — no conjunto todo e no subconjunto de imagens de
núcleos pequenos, que é onde o diagnóstico prevê o ganho.
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
from src.metrics import instance_scores
from src.postprocessing import (
    boundary_prediction_to_instances,
    boundary_prediction_to_instances_seeded,
)


DATA_ROOT = "data/raw"
RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("experiments/figures/failures")

SPLIT_SEED = 42

# Abaixo deste diâmetro mediano a erosão de 2 px começa a apagar
# interiores (ver a tabela por faixa no relatório da Parte 5).
SMALL_NUCLEI_PX = 8.0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--name", type=str, default="correcao")

    return parser.parse_args()


def resolve_checkpoint(explicit):

    if explicit is not None:
        return explicit

    decision_path = RESULTS_DIR / "best_architecture.json"

    if not decision_path.exists():
        raise SystemExit(
            "Sem --checkpoint e sem best_architecture.json. "
            "Rode scripts/summarize_resolution_ablation.py antes."
        )

    with open(decision_path) as file:
        return json.load(file)["checkpoint"]


def median_diameter(labels):

    ids, areas = np.unique(labels, return_counts=True)
    areas = areas[ids != 0]

    if len(areas) == 0:
        return 0.0

    return float(np.median(2.0 * np.sqrt(areas / np.pi)))


def main():

    args = parse_args()

    checkpoint_path = resolve_checkpoint(args.checkpoint)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path, device, in_channels=3, out_channels=3
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Architecture: {checkpoint.get('architecture', 'unet')}")

    dataset = DSB2018Dataset(DATA_ROOT)

    _, _, test_dataset = create_splits(
        dataset, seed=SPLIT_SEED, train_ratio=0.8, val_ratio=0.1
    )

    loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    rows = []
    examples = []

    print()
    print("Rodando o teste com as duas decodificações...")

    with torch.no_grad():

        for index, batch in enumerate(loader):

            images = batch["image"].to(device)

            true_labels = (
                batch["instance_mask"][0].numpy().astype(np.int32)
            )

            probs = torch.softmax(model(images), dim=1)[0].cpu().numpy()

            before = boundary_prediction_to_instances(probs)

            after, rescued = boundary_prediction_to_instances_seeded(probs)

            map_before, _ = instance_scores(true_labels, before)
            map_after, _ = instance_scores(true_labels, after)

            n_true = int(len(np.unique(true_labels)) - 1)

            rows.append({
                "index": index,
                "image_path": batch["image_path"][0],
                "n_true": n_true,
                "median_diameter_px": median_diameter(true_labels),
                "n_pred_before": int(len(np.unique(before)) - 1),
                "n_pred_after": int(len(np.unique(after)) - 1),
                "rescued_markers": rescued,
                "map_before": map_before,
                "map_after": map_after,
                "delta": map_after - map_before,
            })

            examples.append(
                (images[0].cpu().numpy().transpose(1, 2, 0),
                 true_labels, before, after)
            )

    # ----------------------------------------------------------------
    # Agregados
    # ----------------------------------------------------------------

    before_all = np.array([r["map_before"] for r in rows])
    after_all = np.array([r["map_after"] for r in rows])

    diameters = np.array([r["median_diameter_px"] for r in rows])

    small = diameters < SMALL_NUCLEI_PX
    large = ~small

    print()
    print("=" * 72)
    print("PARTE 5 — CORREÇÃO: MARCADOR DE RESGATE NO WATERSHED")
    print("=" * 72)

    print(f"{'subconjunto':<34}{'antes':>10}{'depois':>10}{'delta':>10}")

    def line(label, mask):
        if mask.sum() == 0:
            return
        print(
            f"{label:<34}{before_all[mask].mean():>10.4f}"
            f"{after_all[mask].mean():>10.4f}"
            f"{after_all[mask].mean() - before_all[mask].mean():>+10.4f}"
        )

    line(f"teste inteiro (n={len(rows)})", np.ones(len(rows), bool))
    line(f"núcleos pequenos <{SMALL_NUCLEI_PX:.0f}px (n={small.sum()})", small)
    line(f"núcleos maiores (n={large.sum()})", large)

    improved = sum(1 for r in rows if r["delta"] > 1e-9)
    worsened = sum(1 for r in rows if r["delta"] < -1e-9)

    print()
    print(f"imagens que melhoraram: {improved}")
    print(f"imagens que pioraram:   {worsened}")
    print(f"marcadores resgatados:  "
          f"{sum(r['rescued_markers'] for r in rows)}")

    best = max(rows, key=lambda r: r["delta"])

    print()
    print(
        f"maior ganho: imagem {best['index']} "
        f"({best['n_true']} núcleos, diâmetro mediano "
        f"{best['median_diameter_px']:.1f} px) "
        f"{best['map_before']:.3f} -> {best['map_after']:.3f}, "
        f"previstos {best['n_pred_before']} -> {best['n_pred_after']}"
    )

    if after_all.mean() <= before_all.mean():
        print()
        print(
            "ATENÇÃO: a correção NÃO melhorou o mAP médio. Isso não "
            "invalida o diagnóstico do marcador ausente (os núcleos "
            "pequenos de fato somem), mas mostra que recuperá-los "
            "introduz mais falsos positivos do que verdadeiros — "
            "discuta isso na apresentação."
        )

    # ----------------------------------------------------------------
    # Figura antes/depois na imagem de maior ganho
    # ----------------------------------------------------------------

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    image, true_labels, before, after = examples[best["index"]]

    fig, axes = plt.subplots(1, 4, figsize=(19, 5))

    panels = [
        (image, "imagem", None),
        (true_labels, f"ground truth ({best['n_true']})", "nipy_spectral"),
        (before, f"antes ({best['n_pred_before']})", "nipy_spectral"),
        (after, f"depois ({best['n_pred_after']})", "nipy_spectral"),
    ]

    for ax, (data, title, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"Correção do marcador ausente — mAP "
        f"{best['map_before']:.3f} -> {best['map_after']:.3f}"
    )

    fig.tight_layout()

    figure_path = FIGURES_DIR / f"{args.name}_antes_depois.png"
    fig.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------
    # CSV
    # ----------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / f"{args.name}_por_imagem.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Salvo: {figure_path}")
    print(f"Salvo: {csv_path}")


if __name__ == "__main__":
    main()
