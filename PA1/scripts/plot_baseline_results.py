from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_PATH = Path(
    "experiments/results/baseline_test_metrics.csv"
)

FIGURES_DIR = Path(
    "experiments/figures/baseline"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def plot_map_vs_density(df):

    plt.figure(figsize=(7, 5))

    plt.scatter(
        df["true_count"],
        df["map_50_95"],
        alpha=0.7
    )

    plt.xlabel("Number of ground-truth instances")
    plt.ylabel("mAP@0.50:0.95")
    plt.title(
        "Instance segmentation performance vs. object density"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = (
        FIGURES_DIR / "map_vs_density.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_count_error_vs_density(df):

    plt.figure(figsize=(7, 5))

    plt.scatter(
        df["true_count"],
        df["count_error"],
        alpha=0.7
    )

    plt.xlabel("Number of ground-truth instances")
    plt.ylabel("Absolute counting error")
    plt.title(
        "Counting error vs. object density"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = (
        FIGURES_DIR / "count_error_vs_density.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


def write_summary(df):
    """
    Agrega o CSV por imagem no baseline_summary.csv.

    Antes este arquivo estava versionado sem que nenhum script o
    gerasse: o repositorio tinha um resultado que nao se reproduzia.
    """

    summary = {
        "semantic_iou": df["semantic_iou"].mean(),
        "semantic_dice": df["semantic_dice"].mean(),
        "map_50_95": df["map_50_95"].mean(),
        "mean_count_error": df["count_error"].mean(),
        "count_bias": (df["pred_count"] - df["true_count"]).mean(),
        "density_map_corr": df["true_count"].corr(df["map_50_95"]),
        "density_count_error_corr": df["true_count"].corr(df["count_error"]),
    }

    output_path = RESULTS_PATH.parent / "baseline_summary.csv"

    pd.DataFrame([summary]).to_csv(output_path, index=False)

    print()
    for key, value in summary.items():
        print(f"  {key:<26} {value:.6f}")

    print(f"Saved: {output_path}")


def main():

    df = pd.read_csv(RESULTS_PATH)

    print(f"Loaded {len(df)} test images")

    plot_map_vs_density(df)
    plot_count_error_vs_density(df)
    write_summary(df)


if __name__ == "__main__":
    main()
