"""Box mAP: average precision over detection boxes (Faster R-CNN backend).

The training metric for object detection, in the PASCAL VOC convention:
per class, predicted boxes are matched to ground-truth boxes greedily by
detection score at an IoU threshold — **mAP@0.5** (AP50) by default, AP75
via ``ModelConfig.det_ap_iou``. Average precision is the all-point area
under the precision-recall curve (VOC-2010 style); mAP is the mean over
the classes present in the ground truth. The matching/AP core and the
empty-dataset conventions are shared with the Mask R-CNN mask metric
(:mod:`lidark.models._ap`); this module adds the box IoU.

Ground-truth boxes are the bounding boxes of the class-map instances
(:func:`lidark.models._ap.instances_from_label` + :func:`mask_box`: one box
per 4-connected component; touching same-class objects merge, because the
class maps carry no instance identity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from lidark.models._ap import (
    AP50,
    AP75,
    instances_from_label,
    mask_box,
    mean_average_precision,
)


@dataclass
class Box:
    """One axis-aligned detection box ``(x0, y0, x1, y1)`` (xyxy, exclusive
    max), its class id (>= 1), and the detection score (ground-truth boxes
    carry 1.0)."""

    xyxy: tuple[float, float, float, float]
    class_id: int
    score: float = 1.0


def box_iou_matrix(pred_boxes: list[Box], gt_boxes: list[Box]) -> npt.NDArray[np.float64]:
    """(P, G) pairwise IoU of axis-aligned boxes (0 for empty inputs)."""
    shape = (len(pred_boxes), len(gt_boxes))
    if not pred_boxes or not gt_boxes:
        return np.zeros(shape, dtype=np.float64)
    p = np.asarray([b.xyxy for b in pred_boxes], dtype=np.float64)
    g = np.asarray([b.xyxy for b in gt_boxes], dtype=np.float64)
    x0 = np.maximum(p[:, None, 0], g[None, :, 0])
    y0 = np.maximum(p[:, None, 1], g[None, :, 1])
    x1 = np.minimum(p[:, None, 2], g[None, :, 2])
    y1 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return cast("npt.NDArray[np.float64]", iou)


def boxes_from_label(label: npt.NDArray[np.uint8]) -> list[Box]:
    """Ground-truth boxes of a per-pixel class map (component boxes)."""
    return [
        Box(xyxy=mask_box(instance.mask), class_id=instance.class_id)
        for instance in instances_from_label(label)
    ]


def box_average_precision(
    gts: list[list[Box]],
    preds: list[list[Box]],
    *,
    num_classes: int,
    iou_threshold: float,
) -> float:
    """mAP of predicted boxes vs ground truth at one IoU threshold.

    Args:
        gts: Ground-truth boxes per image (``len(gts) == len(preds)``).
        preds: Predicted boxes (with scores) per image.
        num_classes: Number of foreground classes.
        iou_threshold: Box IoU a prediction must reach to match
            (0.5 = the PASCAL VOC mAP@0.5 default, 0.75 = AP75).

    Returns:
        Mean AP over the classes with ground truth (see
        :mod:`lidark.models._ap` for the empty-dataset conventions).
    """
    return mean_average_precision(
        gts,
        preds,
        num_classes=num_classes,
        iou_threshold=iou_threshold,
        iou_matrix_fn=box_iou_matrix,
    )


__all__ = [
    "AP50",
    "AP75",
    "Box",
    "box_average_precision",
    "box_iou_matrix",
    "boxes_from_label",
]
