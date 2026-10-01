"""
Parte 3 — sumário da ablação, 3 seeds, média ± desvio.

Um eixo só, escolhido em --axis:
  cell    RNN simples vs. LSTM vs. GRU, mesmo orçamento de parâmetros,
          variando T em {4, 8, 16, 32}
  regime  teacher forcing -> scheduled sampling -> free-running, com e sem
          gradient clipping
  input   só geometria, só aparência, ou os dois. Além do IDF1, responde a
          pergunta do enunciado ("qual dos dois sustenta a identidade através
          de uma oclusão longa, e isso muda com a densidade?") medindo a
          sobrevivência da identidade em cada oclusão (src.metrics.
          occlusion_survival), por duração e por cena ordenada por densidade,
          contra o baseline da Parte 1 e o controle sem estágio de aparência
  context causal (online) vs. bidirecional (offline), reportando ganho de IDF1
          e latência juntos

Grava o CSV do sumário e a tabela que vai para a apresentação.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset import OCCLUSION_ORDER, SCENES, TEST_SCENES, TRAIN_SCENES, load_sequence
from src.metrics import drop_distractor_matches, occlusion_survival

from evaluate_tracking import read_tracks

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
                "idf1_std": per_seed["idf1"].std(ddof=1),
                "id_switches_mean": per_seed["id_switches"].mean(),
                "id_switches_std": per_seed["id_switches"].std(ddof=1),
                "id_count_error_mean": per_seed["id_count_error"].mean(),
                "id_count_error_std": per_seed["id_count_error"].std(ddof=1),
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


BUCKETS = ((1, 5, "1-5"), (6, 20, "6-20"), (21, 50, "21-50"), (51, 10**9, ">50"))
# As tabelas saem nas três visões: o agregado das 7 cenas mistura treino e
# teste, então treino e teste também saem separados.
SPLITS = (("todas", SCENES), ("treino", TRAIN_SCENES), ("teste", TEST_SCENES))
LONG = 5          # "oclusão longa" no painel por cena: mais de 5 quadros


def survival_events(tracks_path, sequences):
    by_sequence = read_tracks(tracks_path)
    rows = []
    for scene in SCENES:
        info, gt = sequences[scene]
        pred = drop_distractor_matches(by_sequence.get(scene, []), info["gt_all"])
        for event in occlusion_survival(gt, pred):
            rows.append({"sequence": scene, "density": info["density"], **event})
    return pd.DataFrame(rows)


def survival_tables(runs, sequences):
    """
    runs: [(rótulo, seed, caminho das trajetórias)]. Devolve (por duração,
    por cena): média e desvio (amostral, entre seeds) da taxa de
    sobrevivência e o número médio de oclusões avaliadas (n_mean), só nas
    oclusões scorable. A tabela por duração sai por split (SPLITS).
    """

    by_length, by_scene = [], []
    for label, seed, path in runs:
        events = survival_events(path, sequences)
        scorable = events[events["scorable"]] if len(events) else events
        for split, scenes in SPLITS:
            in_split = scorable[scorable["sequence"].isin(scenes)]
            for lo, hi, name in BUCKETS:
                bucket = in_split[(in_split["length"] >= lo) & (in_split["length"] <= hi)]
                by_length.append({"split": split, "run": label, "seed": seed,
                                  "bucket": name, "n": len(bucket),
                                  "survival": bucket["survived"].mean() if len(bucket) else np.nan})
        for scene, group in scorable[scorable["length"] > LONG].groupby("sequence"):
            by_scene.append({"run": label, "seed": seed, "sequence": scene,
                             "density": group["density"].iloc[0], "n": len(group),
                             "survival": group["survived"].mean()})

    def aggregate(frame, keys):
        grouped = frame.groupby(keys)
        out = (grouped["survival"]
               .agg(survival_mean="mean", survival_std=lambda v: v.std(ddof=1))
               .reset_index())
        out["n_mean"] = grouped["n"].mean().values
        out["n_seeds"] = grouped["seed"].nunique().values
        return out

    return (aggregate(pd.DataFrame(by_length), ["split", "run", "bucket"]),
            aggregate(pd.DataFrame(by_scene), ["run", "sequence", "density"]))


def plot_survival(by_length, by_scene, order, path):
    """
    Esquerda: sobrevivência por duração da oclusão (7 cenas; n médio de
    oclusões em cada barra). Direita: oclusões > LONG quadros por cena,
    da menos à mais densa — a metade "muda com a densidade?" da pergunta.
    Barra de erro só onde há seeds (baseline e controle são 1 execução).
    """

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 4.2),
                                      gridspec_kw={"width_ratios": [1, 1.2]})
    names = [name for _, _, name in BUCKETS]
    width = 0.8 / len(order)
    table = by_length[by_length["split"] == "todas"]
    for k, label in enumerate(order):
        rows = table[table["run"] == label].set_index("bucket").reindex(names)
        errors = 100 * rows["survival_std"].where(rows["n_seeds"] > 1)
        left.bar(np.arange(len(names)) + k * width - 0.4 + width / 2,
                 100 * rows["survival_mean"], width,
                 yerr=errors if (rows["n_seeds"] > 1).any() else None,
                 capsize=2, label=label)
    counts = table[table["run"] == order[0]].set_index("bucket").reindex(names)["n_mean"]
    left.set_xticks(range(len(names)),
                    [f"{n} quadros\n(n≈{c:.0f})" for n, c in zip(names, counts)])
    left.set_ylabel("% das oclusões em que o id sobrevive")
    left.set_title("Por duração (7 cenas; eixos: média ± desvio de 3 seeds)")
    left.legend(fontsize=7)

    scenes = (by_scene[["sequence", "density"]].drop_duplicates()
              .sort_values("density")["sequence"].tolist())
    width = 0.8 / len(order)
    for k, label in enumerate(order):
        rows = by_scene[by_scene["run"] == label].set_index("sequence").reindex(scenes)
        right.bar(np.arange(len(scenes)) + k * width - 0.4 + width / 2,
                  100 * rows["survival_mean"], width, label=label)
    dens = by_scene.drop_duplicates("sequence").set_index("sequence")["density"]
    right.set_xticks(range(len(scenes)),
                     [f"{s}\n({dens[s]:.0f}/q)" for s in scenes])
    right.set_title(f"Oclusões > {LONG} quadros, por cena (da menos à mais densa)")
    right.set_xlabel("cena (pessoas por quadro)")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", required=True,
                        choices=["cell", "regime", "input", "context"])
    parser.add_argument("--name", default=None)
    parser.add_argument("--baseline", default=str(RESULTS / "baseline_tracks.csv"),
                        help="trajetórias do baseline da Parte 1 (referência)")
    parser.add_argument("--control", default=str(RESULTS / "null_after_miss_tracks.csv"),
                        help="mesmo tracker com o estágio de aparência desligado")
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

    # --- a pergunta do eixo: sobrevivência através da oclusão -------------
    sequences = {}
    for scene in SCENES:
        info, gt, _ = load_sequence(scene)
        sequences[scene] = (info, gt)
    runs = []
    for label, path in (("baseline (Parte 1)", args.baseline),
                        ("sem estágio de aparência", args.control)):
        if Path(path).exists():
            runs.append((label, 0, path))
        else:
            print(f"aviso: {path} não existe, referência fora do gráfico")
    for kind in ("appearance", "geometry", "both"):
        for path in sorted(RESULTS.glob(f"input_{kind}_seed*_tracks.csv")):
            seed = int(re.search(r"seed(\d+)", path.name).group(1))
            runs.append((kind, seed, str(path)))

    by_length, by_scene = survival_tables(runs, sequences)
    by_length.to_csv(RESULTS / f"{name}_survival.csv", index=False)
    by_scene.to_csv(RESULTS / f"{name}_survival_per_sequence.csv", index=False)
    order = [label for label in ("baseline (Parte 1)", "sem estágio de aparência",
                                 "appearance", "geometry", "both")
             if label in set(by_length["run"])]
    plot_survival(by_length, by_scene, order, FIGURES / f"{name}_survival.png")

    for split, _ in SPLITS:
        part = by_length[by_length["split"] == split]
        table = part.pivot(index="run", columns="bucket", values="survival_mean")
        counts = part[part["run"] == order[0]].set_index("bucket")["n_mean"]
        print(f"\n% sobrevivência por duração da oclusão ({split}; média das seeds); "
              f"n = {', '.join(f'{b}: {counts[b]:.0f}' for _, _, b in BUCKETS)}")
        print((100 * table.reindex(order)[[n for _, _, n in BUCKETS]]).round(1).to_string())
    dens = by_scene.pivot(index="run", columns="sequence", values="survival_mean")
    cols = sorted(dens.columns, key=lambda s: sequences[s][0]["density"])
    print(f"\n% sobrevivência (oclusões > {LONG} quadros) por cena, da menos à mais densa:")
    print((100 * dens.reindex(order)[cols]).round(0).to_string())
    print(f"gravado {RESULTS / f'{name}_survival.csv'}")
    print(f"gravado {FIGURES / f'{name}_survival.png'}")


if __name__ == "__main__":
    main()
