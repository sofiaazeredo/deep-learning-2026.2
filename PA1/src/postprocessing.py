import numpy as np
import torch
from scipy import ndimage
from skimage.segmentation import watershed

def semantic_prediction_to_instances(
    prediction,
    threshold=0.5,
    connectivity=2
):
    """
    Converte uma previsão binária de segmentação em
    uma máscara de instâncias usando connected components.

    Parameters
    ----------
    prediction : torch.Tensor or np.ndarray
        Probabilidade da classe foreground.
        Pode ter shape [H, W] ou [1, H, W].

    threshold : float
        Threshold utilizado para transformar a probabilidade
        em máscara binária.

    connectivity : int
        Conectividade usada pelo connected components.
        1 = apenas vizinhos ortogonais.
        2 = inclui diagonais.

    Returns
    -------
    instance_mask : np.ndarray
        Máscara [H, W] com:
        0 = background
        1..N = instâncias individuais
    """

    # --------------------------------------------------------
    # Tensor -> NumPy
    # --------------------------------------------------------

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    prediction = np.asarray(prediction)

    # Remove dimensões extras
    prediction = np.squeeze(prediction)

    if prediction.ndim != 2:
        raise ValueError(
            f"Expected [H, W] prediction, got {prediction.shape}"
        )

    # --------------------------------------------------------
    # Probabilidade -> máscara binária
    # --------------------------------------------------------

    binary_mask = prediction >= threshold

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    structure = ndimage.generate_binary_structure(
        rank=2,
        connectivity=connectivity
    )

    instance_mask, num_instances = ndimage.label(
        binary_mask,
        structure=structure
    )

    return instance_mask.astype(np.int32)


def boundary_prediction_to_instances(
    prediction,
    interior_threshold=0.5,
    foreground_threshold=0.5,
    min_size=5
):
    """
    prediction: probabilities [3, H, W]

    classes:
        0 = background
        1 = interior
        2 = boundary
    """

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    prediction = np.asarray(prediction)

    if prediction.ndim != 3:
        raise ValueError(
            f"Expected [3, H, W], got {prediction.shape}"
        )

    # probabilities
    background_prob = prediction[0]
    interior_prob = prediction[1]

    # foreground = anything that is not background
    foreground_prob = 1.0 - background_prob

    foreground = (
        foreground_prob >= foreground_threshold
    )

    # markers for watershed
    interior = (
        interior_prob >= interior_threshold
    )

    interior &= foreground

    # connected components of seeds
    markers, num_markers = ndimage.label(
        interior
    )

    # remove tiny markers
    if min_size > 0:

        sizes = ndimage.sum(
            interior,
            markers,
            range(1, num_markers + 1)
        )

        clean = np.zeros_like(
            interior,
            dtype=bool
        )

        for marker_id, size in enumerate(
            sizes,
            start=1
        ):
            if size >= min_size:
                clean[
                    markers == marker_id
                ] = True

        markers, _ = ndimage.label(
            clean
        )

    distance = ndimage.distance_transform_edt(
        foreground
    )

    instances = watershed(
        -distance,
        markers=markers,
        mask=foreground
    )

    return instances.astype(np.int32)