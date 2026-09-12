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
    interior_class=1,
    boundary_class=2,
    min_size=5
):
    """
    Converte logits/probabilidades de 3 classes em instâncias
    usando interior + boundary + watershed.

    Classes esperadas:
        0 = background
        1 = interior
        2 = boundary

    Parameters
    ----------
    prediction : torch.Tensor or np.ndarray
        Tensor [3, H, W] contendo logits ou probabilidades.

    interior_class : int
        Índice da classe de interior.

    boundary_class : int
        Índice da classe de boundary.

    min_size : int
        Remove marcadores muito pequenos.

    Returns
    -------
    instance_mask : np.ndarray
        Máscara [H, W]:
        0 = background
        1..N = instâncias.
    """

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    prediction = np.asarray(prediction)

    if prediction.ndim != 3:
        raise ValueError(
            f"Expected [C, H, W], got {prediction.shape}"
        )

    # --------------------------------------------------------
    # Classe predita por pixel
    # --------------------------------------------------------

    class_mask = np.argmax(
        prediction,
        axis=0
    )

    interior = (
        class_mask == interior_class
    )

    boundary = (
        class_mask == boundary_class
    )

    # Tudo que não é background é considerado foreground
    foreground = interior | boundary

    # --------------------------------------------------------
    # Remove componentes minúsculos do interior
    # --------------------------------------------------------

    markers, num_markers = ndimage.label(
        interior
    )

    if min_size > 0:

        sizes = ndimage.sum(
            interior,
            markers,
            range(1, num_markers + 1)
        )

        cleaned_interior = np.zeros_like(
            interior,
            dtype=bool
        )

        for marker_id, size in enumerate(
            sizes,
            start=1
        ):
            if size >= min_size:
                cleaned_interior[
                    markers == marker_id
                ] = True

        markers, _ = ndimage.label(
            cleaned_interior
        )

    # --------------------------------------------------------
    # Distance transform
    # --------------------------------------------------------

    distance = ndimage.distance_transform_edt(
        foreground
    )

    # --------------------------------------------------------
    # Watershed
    # --------------------------------------------------------

    instances = watershed(
        -distance,
        markers=markers,
        mask=foreground
    )

    return instances.astype(np.int32)
