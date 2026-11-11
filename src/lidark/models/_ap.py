"""Shared instance-metric core for the detection backends.

Ground-truth instances derive from the dataset's per-pixel class maps via
:func:`instances_from_label` (4-connected components; the class maps carry
no instance identity, so touching objects of the same class merge - the
documented limitation the plan notes for the dense path). Detection quality
is mean average precision: per class, predictions are matched to ground
truth greedily by detection score at an IoU threshold, and average
precision is the all-point area under the precision-recall curve
(VOC-2010 style). The instance backends plug in their own IoU function -
masks for Mask R-CNN (:mod:`lidark.models.maskrcnn.metrics`), boxes for
Faster R-CNN (:mod:`lidark.models.fasterrcnn.metrics`).

Conventions (shared by every backend metric):
- A class with ground truth but no detections scores 0.
- Classes absent from both ground truth and predictions are excluded from
  the mean; when *nothing* has ground truth or predictions, the dataset
  scores 1.0 (nothing to detect - the same convention as the U-Net IoU
  metric's empty union).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

# IoU thresholds of the reported metrics (AP50 default, AP75 optional).
AP50 = 0.5
AP75 = 0.75

# Builds the (P, G) pairwise IoU matrix of one class's predictions and
# ground truth (mask IoU, box IoU, ...).
IoUFactory = Callable[[list[Any], list[Any]], npt.NDArray[np.float64]]


@dataclass
class Instance:
    """One instance: a (H, W) boolean mask, its class id (>= 1), and the
    detection score (ground-truth instances carry 1.0)."""

    mask: npt.NDArray[np.bool_]
    class_id: int
    score: float = 1.0


def mask_box(mask: npt.NDArray[np.bool_]) -> tuple[float, float, float, float]:
    """Bounding box ``(x0, y0, x1, y1)`` (xyxy, exclusive max) of a mask.

    Raises:
        ValueError: If the mask is empty.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("mask_box of an empty mask")
    return float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)


def instances_from_label(label: npt.NDArray[np.uint8]) -> list[Instance]:
    """Connected-component instances of a per-pixel class map.

    Each 4-connected component of a class becomes one instance. Returns the
    instances grouped by class id (ascending), components in scan order.
    """
    import scipy.ndimage

    out: list[Instance] = []
    for class_id in sorted(int(c) for c in np.unique(label) if c >= 1):
        components, count = scipy.ndimage.label(label == class_id)
        for i in range(1, count + 1):
            out.append(Instance(mask=components == i, class_id=class_id))
    return out


def class_detections(
    gts: list[list[Any]],
    preds: list[list[Any]],
    class_id: int,
    iou_threshold: float,
    iou_matrix_fn: IoUFactory,
) -> tuple[list[tuple[float, bool]], int]:
    """Greedy score-ordered matching of one class across all images.

    Returns ``(score, is_tp)`` per prediction (dataset-wide) and the total
    ground-truth count of the class.
    """
    scored: list[tuple[float, bool]] = []
    total_gt = 0
    for image_gts, image_preds in zip(gts, preds, strict=True):
        class_gts = [g for g in image_gts if g.class_id == class_id]
        class_preds = [p for p in image_preds if p.class_id == class_id]
        total_gt += len(class_gts)
        if not class_preds:
            continue
        order = sorted(range(len(class_preds)), key=lambda i: -class_preds[i].score)
        iou = iou_matrix_fn(class_preds, class_gts)
        used = np.zeros(len(class_gts), dtype=bool)
        for i in order:
            candidates = [
                j for j in range(len(class_gts)) if not used[j] and iou[i, j] >= iou_threshold
            ]
            if not candidates:
                scored.append((class_preds[i].score, False))
                continue
            best = max(candidates, key=lambda j: iou[i, j])
            used[best] = True
            scored.append((class_preds[i].score, True))
    return scored, total_gt


def average_precision(scored: list[tuple[float, bool]], total_gt: int) -> float:
    """All-point AP (VOC-2010 style) from (score, is_tp) pairs."""
    if not scored:
        return 0.0  # ground truth exists but nothing was detected
    ranked = sorted(scored, key=lambda pair: -pair[0])
    tp = np.array([1.0 if is_tp else 0.0 for _score, is_tp in ranked], dtype=np.float64)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(total_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # Precision envelope over the extended recall range, then the area
    # under the (stepped) PR curve.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(np.sum(np.diff(mrec) * mpre[1:]))


def mean_average_precision(
    gts: list[list[Any]],
    preds: list[list[Any]],
    *,
    num_classes: int,
    iou_threshold: float,
    iou_matrix_fn: IoUFactory,
) -> float:
    """mAP of predicted instances vs ground truth at one IoU threshold.

    Args:
        gts: Ground-truth instances per image (``len(gts) == len(preds)``).
        preds: Predicted instances (with scores) per image.
        num_classes: Number of foreground classes.
        iou_threshold: IoU a prediction must reach to match
            (0.5 = AP50, 0.75 = AP75).
        iou_matrix_fn: Pairwise IoU of one class's predictions vs ground
            truth (masks or boxes).

    Returns:
        Mean AP over the classes with ground truth (see module docstring
        for the empty-dataset conventions).
    """
    if len(gts) != len(preds):
        raise ValueError(f"gts ({len(gts)}) and preds ({len(preds)}) must align per image")
    aps: list[float] = []
    for class_id in range(1, num_classes + 1):
        scored, total_gt = class_detections(gts, preds, class_id, iou_threshold, iou_matrix_fn)
        if total_gt == 0 and not scored:
            continue  # class absent on both sides
        aps.append(average_precision(scored, total_gt))
    if not aps:
        return 1.0  # nothing to detect and nothing predicted: perfect
    return float(np.mean(aps))


__all__ = [
    "AP50",
    "AP75",
    "Instance",
    "average_precision",
    "class_detections",
    "instances_from_label",
    "mask_box",
    "mean_average_precision",
]
