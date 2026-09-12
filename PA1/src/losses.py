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
