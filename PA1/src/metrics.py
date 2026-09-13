import torch
import numpy as np


def semantic_mask(instance_mask):
    """
    Convert instance IDs into a binary semantic mask.
    """

    return (instance_mask > 0).long()


def iou_score(prediction, instance_mask):

    target = semantic_mask(instance_mask)

    prediction = torch.argmax(
        prediction,
        dim=1
    )

    prediction = prediction.bool()
    target = target.bool()

    intersection = (prediction & target).sum().float()
    union = (prediction | target).sum().float()

    if union == 0:
        return 1.0

    return (intersection / union).item()


def dice_score(prediction, instance_mask):

    target = semantic_mask(instance_mask)

    prediction = torch.argmax(
        prediction,
        dim=1
    )

    prediction = prediction.bool()
    target = target.bool()

    intersection = (prediction & target).sum().float()

    denominator = (
        prediction.sum() +
        target.sum()
    ).float()

    if denominator == 0:
        return 1.0

    return (
        2 * intersection / denominator
    ).item()


IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)


def instance_iou_matrix(true_instances, pred_instances):
    """
    Computes IoU between every ground-truth instance
    and every predicted instance.

    Returns
    -------
    iou_matrix : np.ndarray
        Shape [num_true, num_pred].
    """

    true_instances = np.asarray(true_instances)
    pred_instances = np.asarray(pred_instances)

    true_ids = np.unique(true_instances)
    pred_ids = np.unique(pred_instances)

    # Remove background
    true_ids = true_ids[true_ids != 0]
    pred_ids = pred_ids[pred_ids != 0]

    num_true = len(true_ids)
    num_pred = len(pred_ids)

    if num_true == 0 or num_pred == 0:
        return np.zeros(
            (num_true, num_pred),
            dtype=np.float32
        )

    # Um par (instância verdadeira, instância prevista) por pixel, contado
    # de uma vez com np.bincount, em vez de um par de máscaras booleanas
    # de imagem inteira para cada uma das num_true * num_pred combinações.
    # O resultado é idêntico; a versão ingênua fica inviável no mosaico da
    # Parte 4, onde num_true * num_pred passa de 10^5.

    true_index = np.searchsorted(
        true_ids,
        true_instances.ravel()
    )

    pred_index = np.searchsorted(
        pred_ids,
        pred_instances.ravel()
    )

    true_flat = true_instances.ravel()
    pred_flat = pred_instances.ravel()

    # searchsorted devolve posição válida até para IDs ausentes (fundo),
    # então a máscara de foreground é obrigatória.
    both = (true_flat != 0) & (pred_flat != 0)

    intersections = np.bincount(
        true_index[both] * num_pred + pred_index[both],
        minlength=num_true * num_pred,
    ).reshape(num_true, num_pred)

    true_areas = np.bincount(
        true_index[true_flat != 0],
        minlength=num_true,
    )

    pred_areas = np.bincount(
        pred_index[pred_flat != 0],
        minlength=num_pred,
    )

    unions = (
        true_areas[:, None]
        + pred_areas[None, :]
        - intersections
    )

    iou_matrix = np.zeros(
        (num_true, num_pred),
        dtype=np.float32
    )

    valid = unions > 0

    iou_matrix[valid] = (
        intersections[valid] / unions[valid]
    )

    return iou_matrix


def match_instances(iou_matrix, threshold):
    """
    Performs one-to-one greedy matching between
    ground-truth and predicted instances.

    Matches with highest IoU are selected first.

    Returns
    -------
    tp, fp, fn
    """

    num_true, num_pred = iou_matrix.shape

    candidates = []

    for true_idx in range(num_true):
        for pred_idx in range(num_pred):

            iou = iou_matrix[true_idx, pred_idx]

            if iou >= threshold:
                candidates.append(
                    (iou, true_idx, pred_idx)
                )

    # Highest IoU first
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    matched_true = set()
    matched_pred = set()

    for iou, true_idx, pred_idx in candidates:

        if true_idx in matched_true:
            continue

        if pred_idx in matched_pred:
            continue

        matched_true.add(true_idx)
        matched_pred.add(pred_idx)

    tp = len(matched_true)
    fp = num_pred - tp
    fn = num_true - tp

    return tp, fp, fn


def instance_precision(
    true_instances,
    pred_instances,
    threshold
):
    """
    Instance-level precision used for the baseline:

        TP / (TP + FP + FN)

    This is the object-level score commonly used
    for DSB-style instance segmentation evaluation.
    """

    iou_matrix = instance_iou_matrix(
        true_instances,
        pred_instances
    )

    tp, fp, fn = match_instances(
        iou_matrix,
        threshold
    )

    denominator = tp + fp + fn

    if denominator == 0:
        return 1.0

    return tp / denominator


def precision_from_matrix(iou_matrix, threshold):
    """
    TP / (TP + FP + FN) a partir de uma matriz de IoU já calculada.
    """

    tp, fp, fn = match_instances(
        iou_matrix,
        threshold
    )

    denominator = tp + fp + fn

    if denominator == 0:
        return 1.0

    return tp / denominator


def instance_scores(
    true_instances,
    pred_instances,
    thresholds=IOU_THRESHOLDS
):
    """
    AP por limiar e mAP em UMA passada.

    A matriz de IoU é calculada uma única vez e reaproveitada em todos
    os limiares. Prefira esta função nos loops de avaliação: chamar
    instance_map e depois instance_precision por limiar recalcula a
    mesma matriz 20 vezes por imagem.

    Returns
    -------
    mean_ap : float
    per_threshold : dict {float(threshold): ap}
    """

    iou_matrix = instance_iou_matrix(
        true_instances,
        pred_instances
    )

    per_threshold = {
        float(threshold): precision_from_matrix(
            iou_matrix,
            threshold
        )
        for threshold in thresholds
    }

    return (
        float(np.mean(list(per_threshold.values()))),
        per_threshold,
    )


def instance_map(
    true_instances,
    pred_instances,
    thresholds=IOU_THRESHOLDS
):
    """
    Mean instance precision over IoU thresholds
    0.50, 0.55, ..., 0.95.
    """

    mean_ap, _ = instance_scores(
        true_instances,
        pred_instances,
        thresholds
    )

    return mean_ap


def counting_error(
    true_instances,
    pred_instances
):
    """
    Absolute difference between predicted and
    ground-truth instance counts.
    """

    true_count = len(
        np.unique(true_instances)
    ) - 1

    pred_count = len(
        np.unique(pred_instances)
    ) - 1

    return abs(pred_count - true_count)
