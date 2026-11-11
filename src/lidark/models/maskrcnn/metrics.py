"""Mask mAP: average precision over instance masks (Mask R-CNN backend).

The training metric for instance segmentation: for each class, predicted
instance masks are matched to ground-truth instance masks greedily by
detection score at an IoU threshold — 0.5 for the default **AP50**, 0.75 for
the optional **AP75**. Average precision is the all-point area under the
precision-recall curve (VOC-2010 style); mAP is the mean over the classes
present in the ground truth. The matching/AP core and the empty-dataset
conventions are shared with the Faster R-CNN box metric
(:mod:`lidark.models._ap`); this module adds the mask IoU.

Ground-truth instances come from the per-pixel class maps via
:func:`lidark.models._ap.instances_from_label` (connected components; the
class maps carry no instance identity, so touching objects of the same
class merge — the documented limitation the plan notes for the dense path).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from lidark.models._ap import (
    AP50,
    AP75,
    Instance,
    instances_from_label,
    mask_box,
    mean_average_precision,
)


def mask_iou_matrix(
    pred_masks: list[npt.NDArray[np.bool_]], gt_masks: list[npt.NDArray[np.bool_]]
) -> npt.NDArray[np.float64]:
    """(P, G) pairwise IoU of binary masks (0 for empty inputs)."""
    shape = (len(pred_masks), len(gt_masks))
    if not pred_masks or not gt_masks:
        return np.zeros(shape, dtype=np.float64)
    p = np.stack([np.asarray(m, dtype=bool).ravel() for m in pred_masks]).astype(np.float64)
    g = np.stack([np.asarray(m, dtype=bool).ravel() for m in gt_masks]).astype(np.float64)
    inter = p @ g.T
    union = p.sum(axis=1, keepdims=True) + g.sum(axis=1, keepdims=True).T - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return cast("npt.NDArray[np.float64]", iou)


def mask_average_precision(
    gts: list[list[Instance]],
    preds: list[list[Instance]],
    *,
    num_classes: int,
    iou_threshold: float,
) -> float:
    """mAP of predicted instance masks vs ground truth at one IoU threshold.

    Args:
        gts: Ground-truth instances per image (``len(gts) == len(preds)``).
        preds: Predicted instances (with scores) per image.
        num_classes: Number of foreground classes.
        iou_threshold: Mask IoU a prediction must reach to match
            (0.5 = AP50, 0.75 = AP75).

    Returns:
        Mean AP over the classes with ground truth (see
        :mod:`lidark.models._ap` for the empty-dataset conventions).
    """
    return mean_average_precision(
        gts,
        preds,
        num_classes=num_classes,
        iou_threshold=iou_threshold,
        iou_matrix_fn=lambda ps, gs: mask_iou_matrix([p.mask for p in ps], [g.mask for g in gs]),
    )


__all__ = [
    "AP50",
    "AP75",
    "Instance",
    "instances_from_label",
    "mask_average_precision",
    "mask_box",
    "mask_iou_matrix",
]
