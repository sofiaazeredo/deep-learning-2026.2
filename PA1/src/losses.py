import torch
import torch.nn as nn


def semantic_mask(instance_mask):
    """
    Convert an instance mask into a binary semantic mask.

    0 = background
    1 = object
    """

    return (instance_mask > 0).long()


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, prediction, instance_mask):
        target = semantic_mask(instance_mask)

        return self.loss(prediction, target)


class BoundaryCrossEntropyLoss(nn.Module):

    def __init__(self, class_weights=None):
        super().__init__()

        if class_weights is not None:
            class_weights = torch.tensor(
                class_weights,
                dtype=torch.float32
            )

        self.loss = nn.CrossEntropyLoss(
            weight=class_weights
        )

    def forward(self, prediction, boundary_target):
        return self.loss(
            prediction,
            boundary_target
        )
