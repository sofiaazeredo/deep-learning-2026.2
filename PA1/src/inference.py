"""
Parte 7 — pipeline de inferência.

Recebe o caminho de UMA imagem qualquer (qualquer tamanho, RGB, RGBA ou
tons de cinza) e devolve a máscara de instâncias colorida e a contagem
de objetos. Não treina nada: só carrega um checkpoint.

Funciona com os dois tipos de modelo do projeto e decide sozinho pelo
número de canais de saída do checkpoint:

    2 canais -> baseline semântico (Parte 1): limiar + componentes conexos
    3 canais -> fronteira + watershed (Parte 2, Trilha A)

Uso típico:

    from src.inference import segment_image

    result = segment_image("caminho/para/imagem.png")
    print(result["count"])
    result["overlay"]        # RGB com as instâncias coloridas
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.model import load_model_from_checkpoint
from src.postprocessing import (
    semantic_prediction_to_instances,
    boundary_prediction_to_instances,
    boundary_prediction_to_instances_seeded,
)


RESULTS_DIR = Path("experiments/results")

# O modelo foi treinado com todas as imagens redimensionadas para 256x256
# (ver src/dataset.py), então a inferência precisa usar a mesma escala.
MODEL_INPUT_SIZE = 256


def default_checkpoint():
    """
    O checkpoint escolhido pela ablação de resolução, se existir.
    """

    decision_path = RESULTS_DIR / "best_architecture.json"

    if decision_path.exists():
        with open(decision_path) as file:
            return json.load(file)["checkpoint"]

    # Ordem de preferência quando a ablação ainda não rodou.
    for candidate in (
        "checkpoints/boundary_best.pt",
        "checkpoints/baseline_best.pt",
    ):
        if Path(candidate).exists():
            return candidate

    raise FileNotFoundError(
        "Nenhum checkpoint encontrado. Passe checkpoint=... "
        "explicitamente ou rode scripts/summarize_resolution_ablation.py."
    )


def load_image(path):
    """
    Lê uma imagem de qualquer tipo e devolve RGB uint8 [H, W, 3].

    DSB2018 vem em RGBA; imagens de fora podem vir em tons de cinza ou
    com paleta. Image.convert("RGB") normaliza todos esses casos.
    """

    image = Image.open(path).convert("RGB")

    return np.array(image)


def preprocess(image_rgb):
    """
    RGB uint8 [H, W, 3] -> tensor [1, 3, 256, 256] em [0, 1].

    Devolve também o tamanho original, para remapear a máscara no fim.
    """

    original_size = image_rgb.shape[:2]

    resized = Image.fromarray(image_rgb).resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        Image.Resampling.BILINEAR,
    )

    tensor = torch.from_numpy(
        np.array(resized).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)

    return tensor, original_size


def restore_size(labels, original_size):
    """
    Volta a máscara de instâncias ao tamanho original da imagem.

    NEAREST é obrigatório: qualquer interpolação inventaria IDs que não
    existem (a média entre a instância 3 e a 5 seria a instância 4).
    """

    height, width = original_size

    restored = Image.fromarray(labels.astype(np.int32)).resize(
        (width, height),
        Image.Resampling.NEAREST,
    )

    return np.array(restored).astype(np.int32)


def colorize_instances(labels, seed=0):
    """
    Mapa de instâncias -> RGB float com uma cor por instância.
    """

    rng = np.random.default_rng(seed)

    n_instances = int(labels.max())

    colors = rng.uniform(0.35, 1.0, size=(n_instances + 1, 3))
    colors[0] = 0.0

    return colors[labels]


def overlay_instances(image_rgb, labels, alpha=0.55, seed=0):
    """
    Sobrepõe as instâncias coloridas à imagem original.
    """

    colored = colorize_instances(labels, seed=seed)

    base = image_rgb.astype(np.float32) / 255.0

    mask = (labels > 0)[..., None]

    return np.where(
        mask,
        (1 - alpha) * base + alpha * colored,
        base,
    )


def predict_instances(
    model,
    tensor,
    device,
    out_channels,
    threshold=0.5,
    rescue_markers=False,
):
    """
    Decodifica a saída da rede em instâncias, conforme o tipo de cabeça.
    """

    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    if out_channels == 2:
        # Baseline da Parte 1: canal 1 é a probabilidade de foreground.
        return semantic_prediction_to_instances(
            probs[1],
            threshold=threshold,
        ), probs

    if rescue_markers:
        labels, _ = boundary_prediction_to_instances_seeded(probs)
        return labels, probs

    return boundary_prediction_to_instances(probs), probs


def segment_image(
    image_path,
    checkpoint=None,
    device=None,
    rescue_markers=False,
    keep_original_size=True,
    seed=0,
):
    """
    Caminho de imagem -> instâncias, contagem e visualização.

    Returns
    -------
    dict com:
        image        imagem original RGB uint8
        labels       máscara de instâncias int32 (0 = fundo)
        colored      máscara colorida RGB float
        overlay      imagem + máscara sobreposta
        count        número de objetos
        probs        mapas de probabilidade da rede (para diagnóstico)
        checkpoint   checkpoint usado
        architecture arquitetura reconstruída
    """

    if checkpoint is None:
        checkpoint = default_checkpoint()

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model, meta = load_model_from_checkpoint(checkpoint, device)

    out_channels = model.final.out_channels if hasattr(
        model, "final"
    ) else model.output.out_channels

    image_rgb = load_image(image_path)

    tensor, original_size = preprocess(image_rgb)

    labels, probs = predict_instances(
        model,
        tensor,
        device,
        out_channels,
        rescue_markers=rescue_markers,
    )

    if keep_original_size:
        labels = restore_size(labels, original_size)

    return {
        "image": image_rgb,
        "labels": labels,
        "colored": colorize_instances(labels, seed=seed),
        "overlay": overlay_instances(image_rgb, labels, seed=seed),
        "count": int(len(np.unique(labels)) - 1),
        "probs": probs,
        "checkpoint": str(checkpoint),
        "architecture": meta.get("architecture", "unet"),
        "out_channels": out_channels,
    }
