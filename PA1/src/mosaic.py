import numpy as np


def generate_tiles(
    image,
    tile_size=256,
    overlap=64
):
    """
    Yield overlapping tiles and their coordinates.
    """

    h, w = image.shape[:2]

    stride = tile_size - overlap

    y_positions = list(
        range(0, max(h - tile_size, 0) + 1, stride)
    )

    x_positions = list(
        range(0, max(w - tile_size, 0) + 1, stride)
    )

    if not y_positions or y_positions[-1] != h - tile_size:
        y_positions.append(
            max(h - tile_size, 0)
        )

    if not x_positions or x_positions[-1] != w - tile_size:
        x_positions.append(
            max(w - tile_size, 0)
        )

    for y in y_positions:
        for x in x_positions:

            tile = image[
                y:y + tile_size,
                x:x + tile_size
            ]

            yield tile, x, y

def mask_iou(mask_a, mask_b):
    """
    Compute IoU between two binary instance masks.
    """
    intersection = np.logical_and(
        mask_a,
        mask_b
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b
    ).sum()

    if union == 0:
        return 0.0

    return intersection / union
