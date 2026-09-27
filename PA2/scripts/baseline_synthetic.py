"""
Parte 0.4 e o ensaio da Parte 1.

Roda a associação ingênua no piso fácil (poucas elipses, lentas, sem oclusão),
onde o IDF1 tem que ficar muito perto de 1, e depois gira os botões do gerador
(mais objetos, mais rápidos, oclusão mais longa) para mostrar onde o baseline
quebra. O gráfico dessa varredura é o ensaio da Parte 1.

O tracker é o mesmo da Parte 1 (última caixa observada, Hungarian). As
detecções vêm do simulador sem degradação: o que quebra é a associação, não
o detector.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.detector_sim import degrade
from src.metrics import evaluate_sequence
from src.synthetic import make_sequence
from src.tracker import Tracker

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
FIGURES = Path(__file__).resolve().parents[1] / "experiments" / "figures"

# Mesma regra congelada na Parte 1.
TRACKER = dict(iou_threshold=0.2, max_age=20, min_hits=2, matcher="hungarian")

EASY = dict(n_objects=5, speed=0.5, occlusion_frames=0, n_frames=45)

SWEEPS = {
    "n_objects": {
        "key": "n_objects",
        "values": [5, 8, 12, 15],
        "base": dict(n_objects=5, speed=0.5, occlusion_frames=0, n_frames=45),
        "xlabel": "número de objetos",
    },
    "speed": {
        "key": "speed",
        "values": [0.5, 1.0, 2.0, 4.0, 8.0],
        "base": dict(n_objects=5, speed=0.5, occlusion_frames=0, n_frames=45),
        "xlabel": "velocidade (px / quadro)",
    },
    "occlusion": {
        "key": "occlusion_frames",
        "values": [0, 4, 8, 12, 16],
        # Poucos objetos: com 8 o IDF1 mal sente um alvo ocluído.
        "base": dict(n_objects=3, speed=2.0, occlusion_frames=0, n_frames=45),
        "xlabel": "duração da oclusão (quadros)",
    },
}


def run_once(params, seed):
    _, tracks = make_sequence(seed=seed, **params)
    detections = degrade(tracks, seed=seed)
    pred = Tracker(**TRACKER).run(detections)
    metrics = evaluate_sequence(tracks, pred)
    return metrics


def run_easy(seeds):
    rows = []

    for seed in seeds:
        metrics = run_once(EASY, seed)
        rows.append({"sweep": "easy", "value": 0, "seed": seed, **metrics})
        print(f"piso fácil  seed={seed}  IDF1={metrics['idf1']:.3f}  "
              f"IDsw={metrics['id_switches']}  "
              f"count_err={metrics['id_count_error']}")

    mean = float(np.mean([row["idf1"] for row in rows]))
    print(f"piso fácil  média IDF1={mean:.3f}")
    return rows, mean


def run_sweep(name, seeds):
    spec = SWEEPS[name]
    rows = []

    for value in spec["values"]:
        params = dict(spec["base"])
        params[spec["key"]] = value
        scores = []

        for seed in seeds:
            metrics = run_once(params, seed)
            rows.append({"sweep": name, "value": value, "seed": seed, **metrics})
            scores.append(metrics["idf1"])

        print(f"{name}={value}  IDF1={np.mean(scores):.3f} ± {np.std(scores):.3f}")

    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_sweeps(rows, names, path):
    by_name = {name: [row for row in rows if row["sweep"] == name]
               for name in names}
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4),
                             sharey=True)

    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        spec = SWEEPS[name]
        subset = by_name[name]
        values = spec["values"]
        means, stds = [], []

        for value in values:
            scores = [row["idf1"] for row in subset if row["value"] == value]
            means.append(float(np.mean(scores)))
            stds.append(float(np.std(scores)))

        ax.errorbar(values, means, yerr=stds, marker="o", capsize=3)
        ax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("IDF1")
    fig.suptitle("Onde o baseline ingênuo quebra (média ± desvio, 3 seeds)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"gravado {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["n_objects", "speed", "occlusion", "all"],
                        default="occlusion")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--name", default="synthetic_sweep")
    args = parser.parse_args()

    names = list(SWEEPS) if args.sweep == "all" else [args.sweep]
    easy_rows, easy_idf1 = run_easy(args.seeds)

    if easy_idf1 < 0.95:
        raise SystemExit(f"piso fácil IDF1={easy_idf1:.3f} < 0.95 — o tracker "
                         "não está associando o caso trivial")

    sweep_rows = []
    for name in names:
        sweep_rows.extend(run_sweep(name, args.seeds))

    rows = easy_rows + sweep_rows
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / f"{args.name}.csv", rows)
    plot_sweeps(sweep_rows, names, FIGURES / f"{args.name}.png")
    print(f"gravado {RESULTS / f'{args.name}.csv'}")


if __name__ == "__main__":
    main()
