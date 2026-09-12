from pathlib import Path

import matplotlib.pyplot as plt
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


def main():

    df = pd.read_csv(RESULTS_PATH)

    print(f"Loaded {len(df)} test images")

    plot_map_vs_density(df)
    plot_count_error_vs_density(df)


if __name__ == "__main__":
    main()
