"""
Parte 0.1 — gerador sintético.

Gera os vídeos de elipses com oclusão real (ordem de profundidade) e salva a
figura obrigatória: uma trajetória que some por N quadros e volta.

Número de objetos, velocidade típica e duração da oclusão vêm da linha de
comando; duração do vídeo (30 a 60 quadros), ruído e contraste variam por
sequência, sorteados a partir de --seed.

Saídas:
  data/synthetic/<name>/seq_XXX.npz        frames (uint8, T x H x W) e tracks
                                           (frame, id, x, y, w, h, visibility)
  experiments/results/<name>_sequences.csv parâmetros e buracos por sequência
  experiments/figures/<name>_occlusion.png a trajetória que some e volta
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.synthetic import make_sequence, occlusion_durations

DATA = "data/synthetic"
RESULTS = "experiments/results"
FIGURES = "experiments/figures"


def sample_sequence_params(rng, args):
    """
    O que varia entre sequências. O vídeo nunca é curto demais para a
    oclusão pedida (sobram pelo menos 10 quadros visíveis de cada lado).
    """

    min_frames = max(30, args.occlusion_frames + 20)
    max_frames = max(60, min_frames)

    return {
        "n_frames": int(rng.integers(min_frames, max_frames + 1)),
        "n_objects": args.n_objects,
        "speed": args.speed,
        "occlusion_frames": args.occlusion_frames,
        "noise": float(rng.uniform(0.02, 0.10)),
        "contrast": float(rng.uniform(0.4, 1.0)),
        "seed": int(rng.integers(2**31)),
    }


def plot_occlusion(frames, tracks, path):
    """
    A identidade com o maior buraco: tira de quadros antes, durante e depois,
    com a caixa dela destacada, e a visibilidade quadro a quadro embaixo.
    """

    holes = occlusion_durations(tracks)
    target = max(holes, key=lambda i: max(holes[i], default=0))

    rows = {row[0]: row for row in tracks if row[1] == target}
    present = sorted(rows)

    gaps = [(a, b) for a, b in zip(present, present[1:]) if b - a > 1]
    last_seen, back = max(gaps, key=lambda g: g[1] - g[0])
    first_hidden, last_hidden = last_seen + 1, back - 1

    picks = [last_seen - 4, last_seen - 2, last_seen, first_hidden,
             (first_hidden + last_hidden) // 2, last_hidden, back, back + 3]
    picks = sorted({t for t in picks if 0 <= t < len(frames)})

    fig = plt.figure(figsize=(2.0 * len(picks), 5.2))
    grid = fig.add_gridspec(2, len(picks), height_ratios=[1.0, 0.7])

    for k, t in enumerate(picks):
        ax = fig.add_subplot(grid[0, k])
        ax.imshow(frames[t], cmap="gray", vmin=0.0, vmax=1.0)

        for frame, track_id, x, y, w, h, _ in tracks:
            if frame == t and track_id != target:
                ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                       edgecolor="white", linewidth=0.5,
                                       alpha=0.5))

        if t in rows:
            _, _, x, y, w, h, visibility = rows[t]
            ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                   edgecolor="tab:red", linewidth=1.5))
            ax.set_title(f"t={t}  vis={visibility:.2f}", fontsize=8)
        else:
            ax.set_title(f"t={t}  oculto", fontsize=8, color="tab:red")

        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(grid[1, :])
    visibility = [rows[t][6] if t in rows else 0.0 for t in range(len(frames))]
    ax.step(range(len(frames)), visibility, where="mid", color="tab:red")
    ax.axvspan(first_hidden - 0.5, last_hidden + 0.5, color="tab:red",
               alpha=0.15, label=f"fora do GT: {last_hidden - first_hidden + 1}"
                                 " quadros")
    ax.set_xlim(-0.5, len(frames) - 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("quadro")
    ax.set_ylabel(f"visibilidade do id {target}")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-objects", type=int, default=8)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--occlusion-frames", type=int, default=10)
    parser.add_argument("--n-sequences", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="synthetic")
    args = parser.parse_args()

    if args.occlusion_frames <= 0:
        # Sem oclusão não há trajetória que some e volta para a figura.
        print("aviso: --occlusion-frames 0, a figura de oclusão não é gerada")

    out_dir = Path(DATA) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    Path(FIGURES).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    summary = []
    first = None

    for index in range(args.n_sequences):
        params = sample_sequence_params(rng, args)
        frames, tracks = make_sequence(**params)

        np.savez_compressed(
            out_dir / f"seq_{index:03d}.npz",
            frames=np.round(frames * 255).astype(np.uint8),
            tracks=np.array(tracks, dtype=np.float64).reshape(-1, 7),
        )

        holes = [h for per_id in occlusion_durations(tracks).values()
                 for h in per_id]

        summary.append({
            "sequence": f"seq_{index:03d}",
            **params,
            "n_boxes": len(tracks),
            "n_holes": len(holes),
            "longest_hole": max(holes, default=0),
        })

        if first is None:
            first = (frames, tracks)

    csv_path = Path(RESULTS) / f"{args.name}_sequences.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"{args.n_sequences} sequências em {out_dir}")
    print(f"resumo em {csv_path}")

    if args.occlusion_frames > 0:
        fig_path = Path(FIGURES) / f"{args.name}_occlusion.png"
        plot_occlusion(*first, fig_path)
        print(f"figura em {fig_path}")


if __name__ == "__main__":
    main()
