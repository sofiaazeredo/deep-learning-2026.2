"""
Parte 3 — sumário da ablação, 3 seeds, média ± desvio.

Um eixo só, escolhido em --axis:
  cell    RNN simples vs. LSTM vs. GRU, mesmo orçamento de parâmetros,
          variando T em {4, 8, 16, 32}
  regime  teacher forcing -> scheduled sampling -> free-running, com e sem
          gradient clipping
  input   só geometria, só aparência, ou os dois
  context causal (online) vs. bidirecional (offline), reportando ganho de IDF1
          e latência juntos

Grava o CSV do sumário e a tabela que vai para a apresentação.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd

from src.dataset import OCCLUSION_ORDER

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
FIGURES = Path(__file__).resolve().parents[1] / "experiments" / "figures"


def _input_rows(results):
    pattern = re.compile(r"input_(appearance|geometry|both)_seed(\d+)_per_sequence\.csv")
    rows = []
    for path in sorted(results.glob("input_*_seed*_per_sequence.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        kind, seed = match.group(1), int(match.group(2))
        df = pd.read_csv(path)
        df["sequence"] = df["sequence"].astype(str).str.zfill(2)
        df["input"] = kind
        df["seed"] = seed
        rows.append(df)
    if not rows:
        raise SystemExit(f"nenhum CSV input_*_seed*_per_sequence.csv em {results}")
    return pd.concat(rows, ignore_index=True)


def summarize_input(frame):
    summary = []
    for kind, group in frame.groupby("input"):
        for split_name, subset in (("test", group[group["split"] == "test"]),
                                   ("all", group)):
            per_seed = subset.groupby("seed").agg(
                idf1=("idf1", "mean"),
                id_switches=("id_switches", "sum"),
                id_count_error=("id_count_error", "mean"),
            )
            summary.append({
                "input": kind,
                "split": split_name,
                "n_seeds": len(per_seed),
                "idf1_mean": per_seed["idf1"].mean(),
                "idf1_std": per_seed["idf1"].std(ddof=0),
                "id_switches_mean": per_seed["id_switches"].mean(),
                "id_switches_std": per_seed["id_switches"].std(ddof=0),
                "id_count_error_mean": per_seed["id_count_error"].mean(),
                "id_count_error_std": per_seed["id_count_error"].std(ddof=0),
            })
    return pd.DataFrame(summary)


def per_scene(frame):
    scene = (frame.groupby(["input", "sequence", "split", "camera", "occlusion"])
             .agg(idf1_mean=("idf1", "mean"),
                  idf1_std=("idf1", "std"),
                  id_switches_mean=("id_switches", "mean"))
             .reset_index())
    scene["occlusion"] = scene["occlusion"].astype(float)
    order = {scene_id: i for i, scene_id in enumerate(OCCLUSION_ORDER)}
    scene["order"] = scene["sequence"].map(order)
    return scene.sort_values(["order", "input"])


def plot_input(scene, path):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for kind, group in scene.groupby("input"):
        group = group.sort_values("order")
        ax.errorbar(group["order"], group["idf1_mean"], yerr=group["idf1_std"].fillna(0),
                    marker="o", label=kind)
    ax.set_xticks(range(len(OCCLUSION_ORDER)))
    ax.set_xticklabels([f"{s}\n({scene.loc[scene['sequence']==s, 'occlusion'].iloc[0]:g})"
                        if (scene["sequence"] == s).any() else s
                        for s in OCCLUSION_ORDER])
    ax.set_ylabel("IDF1")
    ax.set_xlabel("cena (mediana de oclusão)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", required=True,
                        choices=["cell", "regime", "input", "context"])
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    if args.axis != "input":
        raise SystemExit(f"Eixo {args.axis} não implementado (Parte 3 é o Eixo 3)")

    name = args.name or "ablation_input"
    frame = _input_rows(RESULTS)
    summary = summarize_input(frame)
    scenes = per_scene(frame)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / f"{name}.csv", index=False)
    scenes.to_csv(RESULTS / f"{name}_per_sequence.csv", index=False)
    plot_input(scenes, FIGURES / f"{name}.png")

    print(summary.round(3).to_string(index=False))
    print()
    print(scenes[["input", "sequence", "occlusion", "idf1_mean", "idf1_std"]]
          .round(3).to_string(index=False))
    print(f"gravado {RESULTS / f'{name}.csv'}")
    print(f"gravado {FIGURES / f'{name}.png'}")


if __name__ == "__main__":
    main()
