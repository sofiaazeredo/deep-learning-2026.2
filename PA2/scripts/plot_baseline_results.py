"""
Parte 1.5 — o gráfico obrigatório do descolamento.

Dois painéis sobre as mesmas sequências:
  em cima,   AP@0.5 de detecção e IDF1;
  embaixo,   identidades previstas / verdadeiras e ID switches por identidade
             verdadeira.

Sequências ordenadas pelo eixo de dificuldade --order-by (default: oclusão).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.dataset import OCCLUSION_ORDER

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
FIGURES = Path(__file__).resolve().parents[1] / "experiments" / "figures"


def order_sequences(df, order_by):
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.zfill(2)
    if order_by == "occlusion":
        rank = {scene: i for i, scene in enumerate(OCCLUSION_ORDER)}
        return df.assign(_rank=df["sequence"].map(rank)).sort_values("_rank")
    if order_by == "density":
        return df.sort_values("density")
    if order_by == "camera_motion":
        rank = {"static": 0, "moving": 1}
        return df.assign(_rank=df["camera"].map(rank)).sort_values(
            ["_rank", "sequence"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",
                        default=str(RESULTS / "baseline_per_sequence.csv"))
    parser.add_argument("--order-by", default="occlusion",
                        choices=["density", "camera_motion", "occlusion"])
    parser.add_argument("--name", default="baseline_decoupling")
    args = parser.parse_args()

    df = order_sequences(pd.read_csv(args.results), args.order_by)
    labels = [f"{scene}\n(occ {occ:.0f})"
              for scene, occ in zip(df["sequence"], df["occlusion"])]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(range(len(df)), df["ap"], marker="o", label="AP@0.5")
    axes[0].plot(range(len(df)), df["idf1"], marker="s", label="IDF1")
    axes[0].set_ylabel("score")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower left")
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Detecção vs. identidade (ordenado por oclusão)")

    axes[1].plot(range(len(df)), df["id_ratio"], marker="o",
                 label="ids pred / gt")
    axes[1].plot(range(len(df)), df["id_switches_per_gt"], marker="s",
                 label="ID switches / id gt")
    axes[1].axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("razão")
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(labels)
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{args.name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"gravado {out}")


if __name__ == "__main__":
    main()
