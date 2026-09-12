import numpy as np
import torch
from scipy import ndimage


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
