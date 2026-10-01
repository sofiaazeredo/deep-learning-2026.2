"""
Parte 2 — a comparação que o enunciado pede: as mesmas métricas lado a lado
com o baseline da Parte 1, nas mesmas sequências.

Duas figuras:
  experiments/figures/<name>_vs_baseline.png   IDF1 e ID switches por cena,
      na ordem de oclusão (eixo da Parte 1.5), baseline vs Trilha B e o
      controle sem estágio de aparência;
  experiments/figures/<name>_<cena>_stills.png  8 quadros de uma cena com as
      trajetórias da Trilha B coloridas por identidade.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset import OCCLUSION_ORDER, load_frame, load_sequence
from src.inference import color_for

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
FIGURES = Path(__file__).resolve().parents[1] / "experiments" / "figures"


def per_sequence(name):
    df = pd.read_csv(RESULTS / f"{name}_per_sequence.csv")
    df["sequence"] = df["sequence"].astype(str).str.zfill(2)
    return df.set_index("sequence").reindex(list(OCCLUSION_ORDER))


def plot_comparison(runs, path):
    """
    runs: [(rótulo, DataFrame por sequência, estilo)].
    """

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True)
    x = np.arange(len(OCCLUSION_ORDER))
    width = 0.8 / len(runs)
    for k, (label, df, alpha) in enumerate(runs):
        offset = k * width - 0.4 + width / 2
        axes[0].bar(x + offset, df["idf1"], width, label=label, alpha=alpha)
        axes[1].bar(x + offset, df["id_switches"], width, label=label, alpha=alpha)
    occlusion = runs[0][1]["occlusion"]
    axes[1].set_xticks(x, [f"{s}\n(occ {o:g})" for s, o in zip(OCCLUSION_ORDER, occlusion)])
    axes[0].set_ylabel("IDF1")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("ID switches")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Trilha B vs baseline, por cena (ordem de oclusão)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_stills(tracks_name, scene, path, n=8):
    info, _, _ = load_sequence(scene)
    tracks = pd.read_csv(RESULTS / f"{tracks_name}_tracks.csv")
    tracks["sequence"] = tracks["sequence"].astype(str).str.zfill(2)
    tracks = tracks[tracks["sequence"] == scene]

    frames = np.linspace(1, info["n_frames"], n).astype(int)
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 6.2))
    for ax, t in zip(axes.ravel(), frames):
        ax.imshow(load_frame(info["path"], int(t), info["im_dir"], info["im_ext"]))
        for row in tracks[tracks["frame"] == t].itertuples():
            color = np.array(color_for(row.id)) / 255
            ax.add_patch(plt.Rectangle((row.x, row.y), row.w, row.h, fill=False,
                                       edgecolor=color, linewidth=1.4))
            ax.text(row.x, row.y - 6, str(row.id), color="white", fontsize=7,
                    bbox=dict(facecolor=color, edgecolor="none", pad=1))
        ax.set_title(f"quadro {t}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Trilha B em MOT17-{scene} ({info['camera']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="temporal")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--control", default="null_after_miss",
                        help="mesmo tracker sem o estágio de aparência")
    parser.add_argument("--scene", default="09", help="cena das imagens")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = [("baseline (Parte 1)", per_sequence(args.baseline), 1.0),
            ("Trilha B", per_sequence(args.name), 1.0)]
    if (RESULTS / f"{args.control}_per_sequence.csv").exists():
        runs.insert(1, ("sem estágio de aparência", per_sequence(args.control), 0.45))
    comparison = FIGURES / f"{args.name}_vs_baseline.png"
    plot_comparison(runs, comparison)
    print(f"gravado {comparison}")

    stills = FIGURES / f"{args.name}_{args.scene}_stills.png"
    plot_stills(args.name, args.scene, stills)
    print(f"gravado {stills}")


if __name__ == "__main__":
    main()
