"""
Parte 4 — a correção.

Diagnóstico (scripts/failure_gallery.py, tipos medidos em TODOS os switches
do modelo final, no treino): 34% dos switches são "não recasou" com buraco
mediano de 2 quadros. A track confirmada perde a detecção por um quadro, a
detecção sobrante vira uma track nova, e no quadro seguinte essa tentativa —
"vista no quadro anterior" — casa antes da confirmada, que só concorre como
perdida. A duplicata ganha da track verdadeira.

A mudança que o diagnóstico sugere (Tracker(prefer_confirmed=True)): as
tentativas só casam depois das confirmadas, vivas e perdidas. Antes/depois
calculados aqui mesmo, no mesmo checkpoint; a decisão é pelo treino.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.appearance import CropEncoder, cache_sequence_embeddings
from src.dataset import SCENES, TEST_SCENES, load_sequence
from src.detection import public_detections
from src.metrics import drop_distractor_matches, evaluate_sequence
from src.model import load_checkpoint
from src.tracker import Tracker

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
FIGURES = Path(__file__).resolve().parents[1] / "experiments" / "figures"
ROOT = Path(__file__).resolve().parents[1]


def embeddings_for_detections(dets, cache):
    return [cache[(int(row[0]), index)] for index, row in enumerate(dets)]


def run_all(model, encoder, device, prefer_confirmed):
    rows = []
    metrics = []
    for scene in SCENES:
        cache = cache_sequence_embeddings(scene, encoder, kind="det",
                                          device=device)
        info, gt, raw = load_sequence(scene)
        dets = public_detections(raw)
        embeddings = embeddings_for_detections(dets, cache)
        tracker = Tracker(iou_threshold=0.2, max_age=20, min_hits=2,
                          matcher="hungarian", motion=model,
                          appearance_threshold=0.5,
                          image_size=(info["im_width"], info["im_height"]),
                          prefer_confirmed=prefer_confirmed)
        tracks = tracker.run(dets, embeddings=embeddings,
                             image_size=(info["im_width"], info["im_height"]))
        # A métrica vê as predições sem os distractores; o CSV guarda as
        # trajetórias cruas, como os outros scripts (quem reavaliar o arquivo
        # tira os distractores uma vez só).
        score = evaluate_sequence(gt, drop_distractor_matches(tracks, info["gt_all"]))
        score.update({"sequence": scene, "split": info.get("camera"),
                      "camera": info["camera"], "occlusion": info["occlusion"],
                      "density": info["density"]})
        score["split"] = "test" if scene in TEST_SCENES else "train"
        metrics.append(score)
        print(f"{scene}  IDF1={score['idf1']:.3f}  IDsw={score['id_switches']}  "
              f"prefer_confirmed={prefer_confirmed}")
        for frame, identity, x, y, w, h in tracks:
            rows.append((scene, frame, identity, x, y, w, h))
    return rows, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--name", default="correcao")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from src.inference import resolve_checkpoint

    checkpoint = resolve_checkpoint(args.checkpoint or None)
    model, payload = load_checkpoint(checkpoint, device=device)
    embed_dim = payload.get("kwargs", {}).get("embed_dim", 128)
    encoder = CropEncoder(embed_dim=embed_dim, freeze=True).to(device)
    if "encoder_proj" in payload:
        encoder.proj.load_state_dict(payload["encoder_proj"])

    # Os dois lados rodam aqui: ler um CSV antigo como "antes" misturaria
    # versões do tracker.
    _, before_metrics = run_all(model, encoder, device, False)
    before = pd.DataFrame(before_metrics)
    before["sequence"] = before["sequence"].astype(str).str.zfill(2)
    after_rows, after_metrics = run_all(model, encoder, device, True)
    after = pd.DataFrame(after_metrics)
    after["sequence"] = after["sequence"].astype(str).str.zfill(2)

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    after.to_csv(RESULTS / f"{args.name}_per_sequence.csv", index=False)
    with open(RESULTS / f"{args.name}_tracks.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "frame", "id", "x", "y", "w", "h"])
        writer.writerows(after_rows)

    merged = before[["sequence", "idf1", "id_switches", "id_count_error"]].merge(
        after[["sequence", "idf1", "id_switches", "id_count_error"]],
        on="sequence", suffixes=("_antes", "_depois"))
    merged.to_csv(RESULTS / f"{args.name}_before_after.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = range(len(merged))
    axes[0].bar([i - 0.18 for i in x], merged["idf1_antes"], width=0.36,
                label="antes")
    axes[0].bar([i + 0.18 for i in x], merged["idf1_depois"], width=0.36,
                label="depois")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(merged["sequence"])
    axes[0].set_ylabel("IDF1")
    axes[0].legend()
    axes[1].bar([i - 0.18 for i in x], merged["id_switches_antes"], width=0.36,
                label="antes")
    axes[1].bar([i + 0.18 for i in x], merged["id_switches_depois"], width=0.36,
                label="depois")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(merged["sequence"])
    axes[1].set_ylabel("ID switches")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{args.name}_before_after.png", dpi=140)
    plt.close(fig)

    summary = {
        "diagnosis": (
            "34% dos ID switches do modelo final (treino) são 'não recasou' "
            "com buraco mediano de 2 quadros: a duplicata nascida no quadro "
            "do miss casa antes da track confirmada. Correção: tentativas só "
            "casam depois das confirmadas (prefer_confirmed)."
        ),
        "decided_on": "train",
    }
    for split in ("train", "test"):
        b = before[before["split"] == split]
        a = after[after["split"] == split]
        summary[f"{split}_idf1_before"] = float(b["idf1"].mean())
        summary[f"{split}_idf1_after"] = float(a["idf1"].mean())
        summary[f"{split}_switches_before"] = int(b["id_switches"].sum())
        summary[f"{split}_switches_after"] = int(a["id_switches"].sum())
    summary.update({
        "all_idf1_before": float(before["idf1"].mean()),
        "all_idf1_after": float(after["idf1"].mean()),
        "all_switches_before": int(before["id_switches"].sum()),
        "all_switches_after": int(after["id_switches"].sum()),
    })
    with open(RESULTS / f"{args.name}.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"treino IDF1 {summary['train_idf1_before']:.3f} → "
          f"{summary['train_idf1_after']:.3f}  IDsw "
          f"{summary['train_switches_before']} → {summary['train_switches_after']}")
    print(f"teste  IDF1 {summary['test_idf1_before']:.3f} → "
          f"{summary['test_idf1_after']:.3f}  IDsw "
          f"{summary['test_switches_before']} → {summary['test_switches_after']}")
    print(f"todas  IDF1 {summary['all_idf1_before']:.3f} → "
          f"{summary['all_idf1_after']:.3f}  "
          f"IDsw {summary['all_switches_before']} → "
          f"{summary['all_switches_after']}")
    print(f"gravado {RESULTS / f'{args.name}.json'}")


if __name__ == "__main__":
    main()
