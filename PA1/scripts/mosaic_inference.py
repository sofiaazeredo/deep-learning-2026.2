"""
Parte 4 — Inferência em mosaico.

O slide 83 descreve a prática padrão: imagens grandes são processadas
em tiles com sobreposição, considerando a parte interna e fazendo a
média dos resultados. Isso resolve segmentação semântica. Para
instâncias não resolve, porque os IDs de instância são LOCAIS a cada
tile: o objeto que cai na costura vira dois objetos.

Este script mede três decodificações sobre o mesmo mosaico:

  full          - o mosaico inteiro numa passada só (referência)
  tiles_naive   - decodifica cada tile e cola com IDs novos (a falha)
  tiles_merged  - decodifica cada tile e funde instâncias entre tiles
                  por IoU na faixa de sobreposição (a correção)

e reporta mAP e contagem para as três, antes e depois da correção.
"""

import sys
from pathlib import Path

# Rodar "python scripts/x.py" coloca scripts/ no sys.path, não a raiz do
# projeto, então "import src" falha. Isso resolve sem exigir PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import DSB2018Dataset, create_splits
from src.model import load_model_from_checkpoint
from src.metrics import instance_map, counting_error
from src.postprocessing import boundary_prediction_to_instances
from src.mosaic import generate_tiles, merge_tile_instances


DATA_ROOT = "data/raw"
RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("experiments/figures/mosaic")

SPLIT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Padrão: o vencedor em best_architecture.json.",
    )

    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--name", type=str, default="mosaic")

    return parser.parse_args()


def resolve_checkpoint(explicit):
    """
    Usa o checkpoint pedido, ou o escolhido pela ablação de resolução.
    """

    if explicit is not None:
        return explicit

    decision_path = RESULTS_DIR / "best_architecture.json"

    if not decision_path.exists():
        raise SystemExit(
            "Sem --checkpoint e sem experiments/results/best_architecture.json. "
            "Rode scripts/summarize_resolution_ablation.py antes, ou passe "
            "--checkpoint explicitamente."
        )

    with open(decision_path) as file:
        return json.load(file)["checkpoint"]


def build_mosaic(dataset, indices, grid):
    """
    Monta um mosaico grid x grid a partir de imagens do teste.

    Os rótulos de cada imagem são deslocados para que cada núcleo do
    mosaico tenha um ID global único.

    Returns
    -------
    image_hwc : np.ndarray [H, W, 3] float32 em [0, 1]
    labels    : np.ndarray [H, W] int32
    tile_px   : int
    """

    samples = [dataset[i] for i in indices[:grid * grid]]

    tile_px = samples[0]["image"].shape[-1]

    canvas = np.zeros(
        (grid * tile_px, grid * tile_px, 3),
        dtype=np.float32,
    )

    labels = np.zeros(
        (grid * tile_px, grid * tile_px),
        dtype=np.int32,
    )

    offset = 0

    for position, sample in enumerate(samples):

        row = position // grid
        col = position % grid

        y = row * tile_px
        x = col * tile_px

        image = sample["image"].numpy().transpose(1, 2, 0)

        canvas[y:y + tile_px, x:x + tile_px] = image

        local = sample["instance_mask"].numpy().astype(np.int32)

        foreground = local > 0

        labels[y:y + tile_px, x:x + tile_px][foreground] = (
            local[foreground] + offset
        )

        offset += int(local.max())

    return canvas, labels, tile_px


def predict_probabilities(model, image_hwc, device):
    """
    Uma passada do modelo sobre uma imagem [H, W, 3] -> probs [3, H, W].
    """

    tensor = torch.from_numpy(
        image_hwc.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    return probs[0].cpu().numpy()


def decode_full(model, image_hwc, device):
    """
    Referência: o mosaico inteiro numa passada só.
    """

    probs = predict_probabilities(model, image_hwc, device)

    return boundary_prediction_to_instances(probs)


def decode_tiles(
    model,
    image_hwc,
    device,
    tile_size,
    overlap,
    merge,
    iou_threshold,
):
    """
    Decodifica tile a tile e cola no mosaico.

    merge=False -> cada instância de cada tile recebe um ID novo.
                   É o que o slide 83 produz quando a decodificação é
                   por tile: o objeto da costura vira dois.
    merge=True  -> instâncias que já existem na faixa de sobreposição
                   são reaproveitadas (a correção).
    """

    height, width = image_hwc.shape[:2]

    global_mask = np.zeros((height, width), dtype=np.int32)

    for tile, x, y in generate_tiles(image_hwc, tile_size, overlap):

        probs = predict_probabilities(model, tile, device)

        tile_instances = boundary_prediction_to_instances(probs)

        if merge:
            global_mask = merge_tile_instances(
                global_mask,
                tile_instances,
                x,
                y,
                iou_threshold=iou_threshold,
            )

        else:
            region = global_mask[y:y + tile_size, x:x + tile_size]

            next_id = int(global_mask.max()) + 1

            for local_id in np.unique(tile_instances):

                if local_id == 0:
                    continue

                region[tile_instances == local_id] = next_id
                next_id += 1

    return global_mask


def relabel_consecutive(labels):
    """
    Remove buracos na sequência de IDs (o matching não deve ver gaps).
    """

    ids = np.unique(labels)
    ids = ids[ids != 0]

    out = np.zeros_like(labels)

    for new_id, old_id in enumerate(ids, start=1):
        out[labels == old_id] = new_id

    return out


def score(true_labels, pred_labels):

    pred_labels = relabel_consecutive(pred_labels)

    return {
        "map_50_95": instance_map(true_labels, pred_labels),
        "count_error": int(counting_error(true_labels, pred_labels)),
        "n_pred": int(len(np.unique(pred_labels)) - 1),
    }


def find_seam_instance(
    true_labels,
    naive_labels,
    tile_size,
    overlap,
    margin=12,
):
    """
    Encontra o núcleo do ground truth que melhor ilustra a falha da
    Parte 4: cruza uma fronteira de tile E foi QUEBRADO em vários IDs
    pela decodificação sem fusão.

    Critério de desempate: número de IDs previstos distintos sobre o
    núcleo (quanto mais fragmentado, mais didático) e, em seguida,
    o tamanho do objeto.
    """

    stride = tile_size - overlap

    height, width = true_labels.shape

    seams = list(range(stride, width, stride))

    best = None
    best_key = (0, 0)

    for instance_id in np.unique(true_labels):

        if instance_id == 0:
            continue

        mask = true_labels == instance_id

        ys, xs = np.where(mask)

        crosses = None

        for seam in seams:
            if xs.min() < seam - margin < seam + margin < xs.max():
                crosses = seam
                break

        if crosses is None:
            continue

        # Quantos IDs distintos a versão sem fusão colocou neste núcleo?
        overlapping = np.unique(naive_labels[mask])
        overlapping = overlapping[overlapping != 0]

        key = (len(overlapping), int(xs.max() - xs.min()))

        if key > best_key:
            best_key = key
            best = (instance_id, crosses, ys, xs, len(overlapping))

    return best


def main():

    args = parse_args()

    checkpoint_path = resolve_checkpoint(args.checkpoint)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        device,
        in_channels=3,
        out_channels=3,
    )

    print(f"Architecture: {checkpoint.get('architecture', 'unet')}")

    dataset = DSB2018Dataset(DATA_ROOT)

    _, _, test_dataset = create_splits(
        dataset,
        seed=SPLIT_SEED,
        train_ratio=0.8,
        val_ratio=0.1,
    )

    indices = list(range(len(test_dataset)))

    image_hwc, true_labels, tile_px = build_mosaic(
        test_dataset,
        indices,
        args.grid,
    )

    n_true = int(len(np.unique(true_labels)) - 1)

    print()
    print(f"Mosaico: {image_hwc.shape[0]}x{image_hwc.shape[1]} "
          f"({args.grid}x{args.grid} imagens de {tile_px} px), "
          f"{n_true} núcleos")

    print(f"Tiles de {args.tile_size} px, sobreposição {args.overlap} px")
    print()

    results = {}

    print("full ...")
    full = decode_full(model, image_hwc, device)
    results["full"] = score(true_labels, full)

    print("tiles_naive ...")
    naive = decode_tiles(
        model, image_hwc, device,
        args.tile_size, args.overlap,
        merge=False, iou_threshold=args.iou_threshold,
    )
    results["tiles_naive"] = score(true_labels, naive)

    print("tiles_merged ...")
    merged = decode_tiles(
        model, image_hwc, device,
        args.tile_size, args.overlap,
        merge=True, iou_threshold=args.iou_threshold,
    )
    results["tiles_merged"] = score(true_labels, merged)

    # ----------------------------------------------------------------
    # Tabela
    # ----------------------------------------------------------------

    print()
    print("=" * 66)
    print("PARTE 4 — INFERÊNCIA EM MOSAICO")
    print("=" * 66)
    print(f"{'decodificação':<16}{'mAP':>10}{'núcleos':>10}"
          f"{'erro cont.':>12}")

    for key in ("full", "tiles_naive", "tiles_merged"):
        row = results[key]
        print(f"{key:<16}{row['map_50_95']:>10.4f}{row['n_pred']:>10}"
              f"{row['count_error']:>12}")

    print(f"{'ground truth':<16}{'—':>10}{n_true:>10}{'—':>12}")

    delta = (
        results["tiles_merged"]["map_50_95"]
        - results["tiles_naive"]["map_50_95"]
    )

    print()
    print(f"Ganho da fusão de instâncias: {delta:+.4f} de mAP")

    # ----------------------------------------------------------------
    # Figura: o objeto na costura
    # ----------------------------------------------------------------

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    seam = find_seam_instance(
        true_labels, naive, args.tile_size, args.overlap
    )

    if seam is None:
        print("Nenhum núcleo do GT cruza uma costura neste mosaico.")

    else:
        instance_id, seam_x, ys, xs, n_fragments = seam

        pad = 40

        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad, true_labels.shape[0])
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad, true_labels.shape[1])

        panels = [
            (image_hwc[y0:y1, x0:x1], "imagem", None),
            (true_labels[y0:y1, x0:x1], "ground truth", "nipy_spectral"),
            (naive[y0:y1, x0:x1], "tiles, sem fusão", "nipy_spectral"),
            (merged[y0:y1, x0:x1], "tiles, com fusão", "nipy_spectral"),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        for ax, (data, title, cmap) in zip(axes, panels):

            ax.imshow(data, cmap=cmap)
            ax.axvline(seam_x - x0, color="red", linestyle="--", linewidth=1)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(
            f"Núcleo atravessado pela costura em x={seam_x} "
            f"(linha vermelha tracejada) — sem fusão ele vira "
            f"{n_fragments} instância(s)"
        )

        fig.tight_layout()

        seam_path = FIGURES_DIR / f"{args.name}_costura.png"
        fig.savefig(seam_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        print(f"Salvo: {seam_path}")

    # ----------------------------------------------------------------
    # Figura: mosaico inteiro
    # ----------------------------------------------------------------

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    for ax, (data, title, cmap) in zip(
        axes,
        [
            (image_hwc, "mosaico", None),
            (true_labels, f"ground truth ({n_true})", "nipy_spectral"),
            (naive, f"sem fusão ({results['tiles_naive']['n_pred']})",
             "nipy_spectral"),
            (merged, f"com fusão ({results['tiles_merged']['n_pred']})",
             "nipy_spectral"),
        ],
    ):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()

    overview_path = FIGURES_DIR / f"{args.name}_visao_geral.png"
    fig.savefig(overview_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------
    # CSV
    # ----------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    import csv

    csv_path = RESULTS_DIR / f"{args.name}_results.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(
            ["decoding", "map_50_95", "n_pred", "count_error", "n_true"]
        )

        for key in ("full", "tiles_naive", "tiles_merged"):
            row = results[key]
            writer.writerow(
                [key, row["map_50_95"], row["n_pred"],
                 row["count_error"], n_true]
            )

    print(f"Salvo: {overview_path}")
    print(f"Salvo: {csv_path}")


if __name__ == "__main__":
    main()
