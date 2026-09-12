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

def merge_tile_instances(
    global_mask,
    tile_instances,
    x,
    y,
    iou_threshold=0.5
):
    """
    Merge instance IDs from one tile into a global instance mask.

    Existing instances with sufficient overlap IoU are reused.
    Otherwise a new global instance ID is created.
    """

    h, w = tile_instances.shape

    region = global_mask[
        y:y + h,
        x:x + w
    ]

    local_ids = np.unique(
        tile_instances
    )

    local_ids = local_ids[
        local_ids != 0
    ]

    next_global_id = (
        int(global_mask.max()) + 1
    )

    for local_id in local_ids:

        local_mask = (
            tile_instances == local_id
        )

        overlapping_ids = np.unique(
            region[local_mask]
        )

        overlapping_ids = overlapping_ids[
            overlapping_ids != 0
        ]

        best_id = None
        best_iou = 0.0

        for global_id in overlapping_ids:

            global_instance = (
                region == global_id
            )

            score = mask_iou(
                local_mask,
                global_instance
            )

            if score > best_iou:
                best_iou = score
                best_id = global_id

        if (
            best_id is not None
            and best_iou >= iou_threshold
        ):
            region[
                local_mask
            ] = best_id

        else:
            region[
                local_mask
            ] = next_global_id

            next_global_id += 1

    global_mask[
        y:y + h,
        x:x + w
    ] = region

    return global_mask
