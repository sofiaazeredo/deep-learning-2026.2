from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("experiments/results")


EXPERIMENTS = {
    "CE": [
        "loss_ce_seed42_test_metrics.csv",
        "loss_ce_seed123_test_metrics.csv",
    ],

    "Weighted CE": [
        "loss_balanced_seed42_test_metrics.csv",
        "loss_balanced_seed123_test_metrics.csv",
    ],

    "Focal γ=1": [
        "loss_focal1_seed42_test_metrics.csv",
        "loss_focal1_seed123_test_metrics.csv",
    ],

    "Focal γ=2": [
        "loss_focal2_seed42_test_metrics.csv",
        "loss_focal2_seed123_test_metrics.csv",
    ],

    "Focal γ=5": [
        "loss_focal5_seed42_test_metrics.csv",
        "loss_focal5_seed123_test_metrics.csv",
    ],
}


def summarize_file(path):

    df = pd.read_csv(path)

    return {
        "map": df["map_50_95"].mean(),
        "count_error": df["count_error"].mean(),
        "count_bias": df["count_bias"].mean(),
    }


def main():

    rows = []

    for config, files in EXPERIMENTS.items():

        run_results = []

        for filename in files:

            path = (
                RESULTS_DIR
                / filename
            )

            result = summarize_file(
                path
            )

            run_results.append(
                result
            )

        maps = np.array(
            [
                r["map"]
                for r in run_results
            ]
        )

        errors = np.array(
            [
                r["count_error"]
                for r in run_results
            ]
        )

        biases = np.array(
            [
                r["count_bias"]
                for r in run_results
            ]
        )

        row = {
            "config": config,

            "map_mean":
                maps.mean(),

            "map_std":
                maps.std(ddof=1),

            "count_error_mean":
                errors.mean(),

            "count_error_std":
                errors.std(ddof=1),

            "count_bias_mean":
                biases.mean(),

            "count_bias_std":
                biases.std(ddof=1),
        }

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    summary = summary.sort_values(
        "map_mean",
        ascending=False
    )

    output_path = (
        RESULTS_DIR
        / "loss_ablation_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False
    )

    print()
    print(summary.to_string(index=False))

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
