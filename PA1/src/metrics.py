import torch


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
