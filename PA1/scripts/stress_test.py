"""
Parte 6 — teste de estresse.

O PDF dá TRÊS opções e manda escolher UMA:

    1. mudança de modalidade / cidade (exige retreinar sem uma modalidade)
    2. corrupções: blur, ruído, brilho/contraste, em 3 intensidades
    3. mudança de escala: avaliar em 0,5x e 2x

Este script implementa as opções 2 e 3, que rodam só com inferência —
escolham UMA para a apresentação. A opção 1 exigiria um treino a mais.

    --mode corruptions   curva de degradação do mAP por tipo e intensidade
    --mode scale         mAP em 0,5x, 1x e 2x

Sobre a opção 3, o PDF pede a discussão explícita: "por que uma rede
totalmente convolucional não é invariante a escala, e o que o ASPP faz
(ou não faz) a respeito?". Rode o mesmo comando com o checkpoint da
U-Net e com o da U-Net+ASPP para responder com número, não com opinião.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

from src.dataset import DSB2018Dataset, create_splits
from src.model import load_model_from_checkpoint
from src.metrics import instance_scores
from src.postprocessing import boundary_prediction_to_instances


DATA_ROOT = "data/raw"
RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("experiments/figures/stress")

SPLIT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument(
        "--mode",
        choices=["corruptions", "scale"],
        default="corruptions",
    )

    parser.add_argument("--name", type=str, default=None)

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


# ----------------------------------------------------------------------
# Corrupções (opção 2)
# ----------------------------------------------------------------------

def apply_blur(image, sigma):
    """
    image: [3, H, W] em [0, 1]. Desfoque gaussiano por canal.
    """

    out = np.stack([
        ndimage.gaussian_filter(channel, sigma=sigma)
        for channel in image
    ])

    return np.clip(out, 0.0, 1.0)


def apply_noise(image, std, seed=0):

    rng = np.random.default_rng(seed)

    return np.clip(
        image + rng.normal(0.0, std, image.shape),
        0.0,
        1.0,
    )


def apply_brightness(image, delta):
    """
    Desloca o brilho e comprime o contraste na mesma proporção — é a
    degradação típica de uma aquisição mal exposta.
    """

    return np.clip(
        (image - 0.5) * (1.0 - abs(delta)) + 0.5 + delta,
        0.0,
        1.0,
    )


CORRUPTIONS = {
    "blur": (apply_blur, [1.0, 2.0, 4.0]),
    "ruido": (apply_noise, [0.05, 0.10, 0.20]),
    "brilho": (apply_brightness, [0.15, 0.30, 0.45]),
}


# ----------------------------------------------------------------------
# Escala (opção 3)
# ----------------------------------------------------------------------

def rescale(image, labels, factor):
    """
    Reescala imagem (bilinear) e rótulos (nearest, obrigatório) juntos.
    """

    if factor == 1.0:
        return image, labels

    size = int(round(image.shape[-1] * factor))

    tensor = torch.from_numpy(image).unsqueeze(0)

    resized = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0].numpy()

    resized_labels = np.array(
        Image.fromarray(labels.astype(np.int32)).resize(
            (size, size),
            Image.Resampling.NEAREST,
        )
    ).astype(np.int32)

    return resized, resized_labels


def evaluate(model, device, samples, transform):
    """
    Roda o modelo sobre todas as amostras já transformadas e devolve o
    mAP médio e o erro de contagem médio.
    """

    maps = []
    count_errors = []

    with torch.no_grad():

        for image, labels in samples:

            image, labels = transform(image, labels)

            tensor = torch.from_numpy(
                image.astype(np.float32)
            ).unsqueeze(0).to(device)

            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

            pred = boundary_prediction_to_instances(probs)

            image_map, _ = instance_scores(labels, pred)

            maps.append(image_map)

            count_errors.append(
                abs(
                    (len(np.unique(pred)) - 1)
                    - (len(np.unique(labels)) - 1)
                )
            )

    return float(np.mean(maps)), float(np.mean(count_errors))


def main():

    args = parse_args()

    name = args.name or args.mode

    checkpoint_path = resolve_checkpoint(args.checkpoint)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path, device, in_channels=3, out_channels=3
    )

    architecture = checkpoint.get("architecture", "unet")

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Architecture: {architecture}")

    dataset = DSB2018Dataset(DATA_ROOT)

    _, _, test_dataset = create_splits(
        dataset, seed=SPLIT_SEED, train_ratio=0.8, val_ratio=0.1
    )

    loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    samples = [
        (
            batch["image"][0].numpy(),
            batch["instance_mask"][0].numpy().astype(np.int32),
        )
        for batch in loader
    ]

    print(f"Amostras de teste: {len(samples)}")
    print()

    rows = []

    baseline_map, baseline_error = evaluate(
        model, device, samples, lambda i, l: (i, l)
    )

    print(f"referência (sem estresse): mAP={baseline_map:.4f} "
          f"erro de contagem={baseline_error:.2f}")
    print()

    if args.mode == "corruptions":

        rows.append({
            "eixo": "nenhum", "intensidade": 0.0,
            "map": baseline_map, "count_error": baseline_error,
        })

        for corruption, (function, levels) in CORRUPTIONS.items():

            for level in levels:

                score, error = evaluate(
                    model, device, samples,
                    lambda i, l, f=function, v=level: (f(i, v), l),
                )

                rows.append({
                    "eixo": corruption, "intensidade": level,
                    "map": score, "count_error": error,
                })

                change = 100 * (score / baseline_map - 1) if baseline_map else 0

                print(f"{corruption:<8} {level:<6} mAP={score:.4f} "
                      f"({change:+6.1f}% vs referência)  "
                      f"erro={error:.2f}")

    else:

        for factor in (0.5, 1.0, 2.0):

            score, error = evaluate(
                model, device, samples,
                lambda i, l, f=factor: rescale(i, l, f),
            )

            rows.append({
                "eixo": "escala", "intensidade": factor,
                "map": score, "count_error": error,
            })

            side = int(round(256 * factor))

            change = 100 * (score / baseline_map - 1) if baseline_map else 0

            print(f"escala {factor}x ({side}x{side} px)  mAP={score:.4f} "
                  f"({change:+6.1f}% vs 1x)  erro={error:.2f}")

    # ----------------------------------------------------------------
    # Figura
    # ----------------------------------------------------------------

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    if args.mode == "corruptions":

        for corruption in CORRUPTIONS:

            points = [r for r in rows if r["eixo"] == corruption]

            ax.plot(
                [0] + [r["intensidade"] for r in points],
                [baseline_map] + [r["map"] for r in points],
                marker="o",
                label=corruption,
            )

        ax.set_xlabel("intensidade da corrupção")
        ax.set_title(
            f"Parte 6 — degradação do mAP sob corrupção ({architecture})"
        )

    else:

        points = [r for r in rows if r["eixo"] == "escala"]

        ax.plot(
            [r["intensidade"] for r in points],
            [r["map"] for r in points],
            marker="o",
        )

        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1.0, 2.0])
        ax.set_xticklabels(["0,5x", "1x", "2x"])
        ax.set_xlabel("escala da imagem")
        ax.set_title(
            f"Parte 6 — mAP x escala ({architecture})"
        )

    ax.set_ylabel("mAP@0,50:0,95")
    ax.axhline(
        baseline_map, color="gray", linestyle="--", linewidth=1,
        label="referência",
    )
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    figure_path = FIGURES_DIR / f"{name}_{architecture}.png"
    fig.savefig(figure_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------
    # CSV
    # ----------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / f"stress_{name}_{architecture}.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["eixo", "intensidade", "map", "count_error"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Salvo: {figure_path}")
    print(f"Salvo: {csv_path}")


if __name__ == "__main__":
    main()
