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

    iou_matrix = np.zeros(
        (len(true_ids), len(pred_ids)),
        dtype=np.float32
    )

    for i, true_id in enumerate(true_ids):

        true_mask = true_instances == true_id

        for j, pred_id in enumerate(pred_ids):

            pred_mask = pred_instances == pred_id

            intersection = np.logical_and(
                true_mask,
                pred_mask
            ).sum()

            if intersection == 0:
                continue

            union = np.logical_or(
                true_mask,
                pred_mask
            ).sum()

            iou_matrix[i, j] = intersection / union

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


def instance_map(
    true_instances,
    pred_instances,
    thresholds=IOU_THRESHOLDS
):
    """
    Mean instance precision over IoU thresholds
    0.50, 0.55, ..., 0.95.
    """

    scores = []

    for threshold in thresholds:

        score = instance_precision(
            true_instances,
            pred_instances,
            threshold
        )

        scores.append(score)

    return float(np.mean(scores))


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
