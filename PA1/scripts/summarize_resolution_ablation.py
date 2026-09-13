"""
Eixo 1 da Parte 3 — como recuperar resolução.

Agrega as execuções das três arquiteturas (2 seeds cada) e reporta
média ± desvio. Todas compartilham a mesma perda (balanced_ce), o
mesmo split (SPLIT_SEED = 42) e o mesmo número de épocas, de modo que
a única variável é o mecanismo de recuperação de resolução.

O braço "U-Net + skips" reaproveita as execuções Weighted CE da
ablação de perdas: são exatamente a mesma configuração.

Também escolhe a melhor arquitetura por mAP de instância (Passo 3) e
grava experiments/results/best_architecture.json, que os scripts das
Partes 4, 5, 6 e 7 leem para saber qual checkpoint usar.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("experiments/results")


# config -> (arquitetura, [csv por seed], [checkpoint por seed])
EXPERIMENTS = {
    "U-Net + skips": (
        "unet",
        [
            "loss_balanced_seed42_test_metrics.csv",
            "loss_balanced_seed123_test_metrics.csv",
        ],
        [
            "checkpoints/loss_balanced_seed42_best.pt",
            "checkpoints/loss_balanced_seed123_best.pt",
        ],
    ),

    "U-Net sem skips": (
        "no_skips",
        [
            "resolution_no_skips_seed42_test_metrics.csv",
            "resolution_no_skips_seed123_test_metrics.csv",
        ],
        [
            "checkpoints/resolution_no_skips_seed42_best.pt",
            "checkpoints/resolution_no_skips_seed123_best.pt",
        ],
    ),

    "U-Net + ASPP": (
        "aspp",
        [
            "resolution_aspp_seed42_test_metrics.csv",
            "resolution_aspp_seed123_test_metrics.csv",
        ],
        [
            "checkpoints/resolution_aspp_seed42_best.pt",
            "checkpoints/resolution_aspp_seed123_best.pt",
        ],
    ),
}


def summarize_file(path):
    """
    Médias por imagem de uma execução.
    """

    df = pd.read_csv(path)

    return {
        "map": df["map_50_95"].mean(),
        "count_error": df["count_error"].mean(),
        "count_bias": df["count_bias"].mean(),
    }


def main():

    rows = []
    per_config_seed_maps = {}
    missing = []

    for config, (architecture, files, _) in EXPERIMENTS.items():

        run_results = []

        for filename in files:

            path = RESULTS_DIR / filename

            if not path.exists():
                missing.append(str(path))
                continue

            run_results.append(
                summarize_file(path)
            )

        if len(run_results) < 2:
            # Sem as duas seeds não há desvio padrão para reportar.
            continue

        maps = np.array([r["map"] for r in run_results])
        errors = np.array([r["count_error"] for r in run_results])
        biases = np.array([r["count_bias"] for r in run_results])

        per_config_seed_maps[config] = maps

        rows.append({
            "config": config,
            "architecture": architecture,
            "n_seeds": len(run_results),
            "map_mean": maps.mean(),
            "map_std": maps.std(ddof=1),
            "count_error_mean": errors.mean(),
            "count_error_std": errors.std(ddof=1),
            "count_bias_mean": biases.mean(),
            "count_bias_std": biases.std(ddof=1),
        })

    if missing:
        print("CSVs ausentes (execute a avaliação antes):")
        for path in missing:
            print(f"  - {path}")
        print()

    if not rows:
        raise SystemExit(
            "Nenhuma configuração completa (2 seeds). "
            "Rode scripts/evaluate_boundary.py para cada checkpoint."
        )

    summary = pd.DataFrame(rows).sort_values(
        "map_mean",
        ascending=False,
    )

    output_path = RESULTS_DIR / "resolution_ablation_summary.csv"
    summary.to_csv(output_path, index=False)

    print()
    print("=" * 70)
    print("EIXO 1 — RECUPERAÇÃO DE RESOLUÇÃO (média ± desvio, 2 seeds)")
    print("=" * 70)
    print(summary.to_string(index=False))

    # ----------------------------------------------------------------
    # Passo 3 — escolher a arquitetura final
    # ----------------------------------------------------------------

    best = summary.iloc[0]

    print()
    print(f"Melhor por mAP: {best['config']} "
          f"({best['map_mean']:.4f} ± {best['map_std']:.4f})")

    # Duas seeds dão um desvio muito frágil. Se o segundo colocado cai
    # dentro de um desvio do primeiro, a diferença não é conclusiva e a
    # apresentação precisa dizer isso em vez de cravar um vencedor.
    if len(summary) > 1:

        runner_up = summary.iloc[1]

        gap = best["map_mean"] - runner_up["map_mean"]
        tolerance = best["map_std"] + runner_up["map_std"]

        if gap < tolerance:
            print(
                f"  ATENÇÃO: {runner_up['config']} "
                f"({runner_up['map_mean']:.4f} ± {runner_up['map_std']:.4f}) "
                f"está a {gap:.4f} de distância, dentro da soma dos desvios "
                f"({tolerance:.4f}). Com 2 seeds a diferença NÃO é conclusiva "
                f"— reporte as duas e justifique a escolha."
            )
        else:
            print(
                f"  Diferença de {gap:.4f} para o segundo colocado, acima da "
                f"soma dos desvios ({tolerance:.4f})."
            )

    # Só escolhe a arquitetura final quando as TRÊS configurações
    # estão completas — senão o "vencedor" seria só a única que rodou.
    if len(summary) < len(EXPERIMENTS):
        print()
        print(
            f"Apenas {len(summary)} de {len(EXPERIMENTS)} configurações "
            f"completas: best_architecture.json NÃO foi gravado. "
            f"Termine as avaliações e rode este script de novo."
        )
        return

    # Checkpoint da melhor seed dentro da melhor configuração.
    best_config = best["config"]
    _, files, checkpoints = EXPERIMENTS[best_config]

    seed_maps = per_config_seed_maps[best_config]
    best_seed_index = int(np.argmax(seed_maps))

    decision = {
        "config": best_config,
        "architecture": best["architecture"],
        "map_mean": float(best["map_mean"]),
        "map_std": float(best["map_std"]),
        "checkpoint": checkpoints[best_seed_index],
        "checkpoints_all_seeds": checkpoints,
        "metrics_csv": files[best_seed_index],
        "selected_by": "maior mAP médio; dentro dele, a seed de maior mAP",
    }

    decision_path = RESULTS_DIR / "best_architecture.json"

    with open(decision_path, "w") as file:
        json.dump(decision, file, indent=2, ensure_ascii=False)

    print()
    print(f"Checkpoint final: {decision['checkpoint']}")
    print(f"Salvo: {output_path}")
    print(f"Salvo: {decision_path}")


if __name__ == "__main__":
    main()
