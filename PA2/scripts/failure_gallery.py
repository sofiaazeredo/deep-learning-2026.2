"""
Parte 4 — galeria de falhas.

Três trechos em que o modelo final erra feio. Cada um com a tira de quadros
(ground truth e predição coloridos por identidade) mais o mapa intermediário
relevante — a matriz de similaridade dos embeddings — e o diagnóstico.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.appearance import CropEncoder, cache_sequence_embeddings
from src.association import cosine_cost
from src.dataset import SCENES, TEST_SCENES, load_frame, load_sequence
from src.detection import public_detections
from src.metrics import drop_distractor_matches, switch_events
from src.model import load_checkpoint

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
FIGURES = Path(__file__).resolve().parents[1] / "experiments" / "figures"


def read_tracks(path):
    by_sequence = defaultdict(list)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            by_sequence[str(row["sequence"]).zfill(2)].append((
                int(float(row["frame"])), int(float(row["id"])),
                float(row["x"]), float(row["y"]),
                float(row["w"]), float(row["h"]),
            ))
    return by_sequence


def collect_events(tracks_path):
    by_sequence = read_tracks(tracks_path)
    events = []
    for scene in SCENES:
        info, gt, _ = load_sequence(scene)
        pred = drop_distractor_matches(by_sequence.get(scene, []),
                                       info["gt_all"])
        spans = {}
        for frame, identity, *_ in pred:
            first, last = spans.get(identity, (frame, frame))
            spans[identity] = (min(first, frame), max(last, frame))
        for event in switch_events(gt, pred):
            event["scene"] = scene
            event["split"] = "test" if scene in TEST_SCENES else "train"
            event["camera"] = info["camera"]
            event["occlusion"] = info["occlusion"]
            event["density"] = info["density"]
            event["kind"] = classify(event, spans)
            event["pred_from_last"] = spans[event["pred_from"]][1]
            events.append(event)
    events.sort(key=lambda item: (-item["gap"], -item["density"]))
    return events


MAX_AGE = 20      # regra congelada do tracker (README)
WINDOW = 16       # janela de BPTT do treino


def classify(event, spans):
    """
    O que aconteceu, medido nas trajetórias:
      morreu        o buraco passou de max_age: a track antiga já tinha morrido
                    e nasceu um id novo;
      nao_recasou   a track antiga ainda vivia (buraco <= max_age), mas na volta
                    nem o IoU nem o cosseno com portão a aceitaram: id novo;
      troca         o id novo já existia antes (era de outra pessoa).
    spans[id] = (primeiro, último) quadro emitido do id previsto.
    """

    born = spans[event["pred_to"]][0] >= event["prev_frame"]
    if not born:
        return "troca"
    # Vista pela última vez no quadro s, a track ainda casa no quadro
    # s + max_age + 1 (ela só é removida depois do miss desse quadro). E se o
    # id antigo continuou saindo depois do sumiço, ele estava vivo — casou
    # com outra pessoa — por mais longo que seja o buraco.
    still_emitting = spans[event["pred_from"]][1] > event["prev_frame"]
    if event["gap"] > MAX_AGE + 1 and not still_emitting:
        return "morreu"
    return "nao_recasou"


def pick_failures(events, n=3):
    """
    Uma falha de cada tipo medido (morreu / não recasou / troca), em cenas
    diferentes, preferindo as de TREINO — a correção sai daqui, então o teste
    não entra na escolha. Dentro do tipo: o buraco mais longo que ainda cabe
    numa figura (<= 60 quadros), cena mais densa no desempate.
    """

    used_scenes, chosen = set(), []
    for kind in ("morreu", "nao_recasou", "troca"):
        pool = [e for e in events if e["kind"] == kind and e["gap"] <= 60
                and e["split"] == "train" and e["scene"] not in used_scenes]
        pool.sort(key=lambda e: (-e["gap"], -e["density"]))
        if pool:
            chosen.append(pool[0])
            used_scenes.add(pool[0]["scene"])
    return chosen[:n]


def category_table(events):
    """
    Quanto cada tipo pesa entre TODOS os switches, no treino: é o diagnóstico
    quantitativo que escolhe a correção.
    """

    rows = []
    for split in ("train", "test"):
        subset = [e for e in events if e["split"] == split]
        for kind in ("morreu", "nao_recasou", "troca"):
            k = [e for e in subset if e["kind"] == kind]
            rows.append({"split": split, "kind": kind, "n": len(k),
                         "share": len(k) / len(subset) if subset else 0.0,
                         "median_gap": float(np.median([e["gap"] for e in k])) if k else 0.0})
    return rows


def _color(track_id):
    cmap = plt.get_cmap("tab20")
    return cmap(int(track_id) % 20)


def _focus_box(rows, identity):
    for row in rows:
        if int(row[1]) == int(identity):
            return row[2], row[3], row[4], row[5]
    return None


def _crop_view(image, boxes, pad=90):
    height, width = image.shape[:2]
    valid = [box for box in boxes if box is not None]
    if not valid:
        return image, (0, 0)
    x0 = max(0, int(min(box[0] for box in valid) - pad))
    y0 = max(0, int(min(box[1] for box in valid) - pad))
    x1 = min(width, int(max(box[0] + box[2] for box in valid) + pad))
    y1 = min(height, int(max(box[1] + box[3] for box in valid) + pad))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return image, (0, 0)
    return image[y0:y1, x0:x1], (x0, y0)


# pad amplo o bastante para caber o vizinho que roubou o id
_CROP_PAD = 160


def _draw_boxes(ax, image, rows, title, highlight=None, origin=(0, 0)):
    ax.imshow(image)
    ox, oy = origin
    for frame, identity, x, y, w, h in rows:
        thick = 2.4 if highlight is not None and int(identity) == int(highlight) else 1.2
        ax.add_patch(Rectangle((x - ox, y - oy), w, h, fill=False,
                               edgecolor=_color(identity), linewidth=thick))
        ax.text(x - ox, y - oy - 4, str(identity), color="white", fontsize=7,
                bbox=dict(facecolor=_color(identity), edgecolor="none", pad=1))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def render_failure(event, pred_rows, encoder, device, path):
    info, gt, raw = load_sequence(event["scene"])
    dets = public_detections(raw)
    cache = cache_sequence_embeddings(event["scene"], encoder, kind="det",
                                      device=device)
    center = event["frame"]
    frames = [t for t in range(center - 3, center + 4)
              if 1 <= t <= info["n_frames"]]
    fig, axes = plt.subplots(3, len(frames), figsize=(2.6 * len(frames), 7.6),
                             constrained_layout=True)
    if len(frames) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    for col, t in enumerate(frames):
        image = load_frame(info["path"], t, info["im_dir"], info["im_ext"])
        gt_rows = [(r[0], r[1], r[2], r[3], r[4], r[5])
                   for r in gt if r[0] == t and int(r[1]) == int(event["gt_id"])]
        pred_t = [row for row in pred_rows if row[0] == t
                  and int(row[1]) in {int(event["pred_from"]),
                                      int(event["pred_to"])}]
        focus = [
            _focus_box(gt_rows, event["gt_id"]),
            _focus_box(pred_t, event["pred_from"]),
            _focus_box(pred_t, event["pred_to"]),
        ]
        crop, origin = _crop_view(image, focus, pad=_CROP_PAD)
        _draw_boxes(axes[0, col], crop, gt_rows, f"gt {t}",
                    highlight=event["gt_id"], origin=origin)
        _draw_boxes(axes[1, col], crop, pred_t, f"pred {t}",
                    highlight=event["pred_to"], origin=origin)

        frame_dets = [row for row in dets if row[0] == t]
        if len(frame_dets) >= 2:
            start = next(i for i, row in enumerate(dets) if row[0] == t)
            vectors = [cache[(t, start + i)] for i in range(len(frame_dets))]
            sim = 1.0 - cosine_cost(vectors, vectors)
            axes[2, col].imshow(sim, vmin=0, vmax=1, cmap="magma")
        else:
            axes[2, col].imshow(np.zeros((1, 1)), vmin=0, vmax=1, cmap="magma")
        axes[2, col].set_title("cosseno dets", fontsize=8)
        axes[2, col].axis("off")

    fig.suptitle(
        f"MOT17-{event['scene']}  gt={event['gt_id']}  "
        f"id {event['pred_from']}→{event['pred_to']}  "
        f"gap={event['gap']}  quadro {event['frame']}",
        fontsize=11)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def diagnose(event):
    """
    Diagnóstico com os números da própria falha.
    """

    head = (f"Cena {event['scene']} ({event['camera']}, densidade "
            f"{event['density']:.1f}/quadro): o gt {event['gt_id']} ficou "
            f"{event['gap']} quadros sem casamento e voltou como id "
            f"{event['pred_to']} (era {event['pred_from']}).")
    # A ligação com o treino que o enunciado pede: buraco maior que a janela
    # de BPTT nunca teve gradiente atravessando.
    window = ""
    if event["gap"] > WINDOW:
        window = (f" O buraco ({event['gap']}) é maior que a janela de BPTT "
                  f"(T={WINDOW}): no treino a memória nunca recebeu sinal de "
                  f"supervisão atravessando um buraco desse tamanho.")
    if event["kind"] == "morreu":
        return (f"{head} O buraco de {event['gap']} passa de max_age={MAX_AGE}: "
                f"a track {event['pred_from']} morreu antes de a pessoa voltar, "
                f"então nenhuma memória — nem a da GRU — tinha como casar."
                + window)
    if event["kind"] == "nao_recasou":
        last = event.get("pred_from_last", event["prev_frame"])
        where = (f" O id {event['pred_from']} continuou saindo até o quadro "
                 f"{last}, em outra pessoa: a track derivou durante o buraco."
                 if last > event["prev_frame"] else "")
        return (f"{head} A track {event['pred_from']} ainda existia, mas na "
                f"volta nem o IoU da última caixa nem o cosseno dentro do "
                f"portão a aceitaram; nasceu um id novo." + where + window)
    return (f"{head} O id {event['pred_to']} já existia: a detecção da volta foi "
            f"para a track de outra pessoa (troca), não para uma track nova."
            + window)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tracks", default=None)
    parser.add_argument("--n-failures", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="failures")
    args = parser.parse_args()

    import torch

    tracks_path = Path(args.tracks) if args.tracks else RESULTS / "temporal_tracks.csv"
    events = collect_events(tracks_path)
    categories = category_table(events)
    print("tipos de ID switch do modelo final:")
    for row in categories:
        print(f"  {row['split']:<5} {row['kind']:<12} {row['n']:>5}  "
              f"({100 * row['share']:.0f}%)  gap mediano {row['median_gap']:.0f}")
    chosen = pick_failures(events, args.n_failures)
    if not chosen:
        raise SystemExit("nenhum ID switch nas trajetórias")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from src.inference import resolve_checkpoint

    checkpoint = str(resolve_checkpoint(args.checkpoint or None))
    encoder = CropEncoder(embed_dim=128, freeze=True).to(device)
    if Path(checkpoint).exists():
        _, payload = load_checkpoint(checkpoint, device=device)
        if "encoder_proj" in payload:
            encoder.proj.load_state_dict(payload["encoder_proj"])

    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    by_sequence = read_tracks(tracks_path)
    report = []

    for index, event in enumerate(chosen, start=1):
        event["diagnosis"] = diagnose(event)
        fig_path = FIGURES / f"{args.name}_{index}.png"
        render_failure(event, by_sequence[event["scene"]], encoder, device,
                       fig_path)
        event["figure"] = str(fig_path)
        report.append(event)
        print(f"[{index}] {event['scene']} gt={event['gt_id']} "
              f"gap={event['gap']}  {fig_path}")
        print(f"    {event['diagnosis']}")

    with open(RESULTS / f"{args.name}.json", "w") as handle:
        json.dump({"categories": categories, "failures": report}, handle, indent=2)
    print(f"gravado {RESULTS / f'{args.name}.json'}")


if __name__ == "__main__":
    main()
