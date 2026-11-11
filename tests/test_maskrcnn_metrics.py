"""Tests for the Mask R-CNN mask-mAP metric and instance targets (numpy only)."""

from __future__ import annotations

import numpy as np
import pytest

from lidark.models.maskrcnn.metrics import (
    AP50,
    AP75,
    Instance,
    instances_from_label,
    mask_average_precision,
    mask_box,
    mask_iou_matrix,
)


def _rect(shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def test_instances_from_label_components_and_classes() -> None:
    label = np.zeros((8, 8), dtype=np.uint8)
    label[0:2, 0:2] = 1
    label[5:7, 5:7] = 1
    label[3:5, 3:5] = 2
    instances = instances_from_label(label)
    assert [i.class_id for i in instances] == [1, 1, 2]
    assert all(i.score == 1.0 for i in instances)
    # Components are disjoint masks of their class cells.
    assert instances[0].mask.sum() == 4
    assert instances[2].mask.sum() == 4


def test_instances_from_label_touching_objects_merge() -> None:
    # Documented limitation: the class maps carry no instance identity, so
    # touching same-class components become one instance.
    label = np.zeros((4, 8), dtype=np.uint8)
    label[0:3, 0:3] = 1
    label[0:3, 3:6] = 1
    assert len(instances_from_label(label)) == 1


def test_instances_from_label_empty_map() -> None:
    assert instances_from_label(np.zeros((4, 4), dtype=np.uint8)) == []


def test_mask_box() -> None:
    assert mask_box(_rect((8, 8), 1, 2, 3, 5)) == (2.0, 1.0, 5.0, 3.0)
    with pytest.raises(ValueError, match="empty"):
        mask_box(np.zeros((4, 4), dtype=bool))


def test_mask_iou_matrix() -> None:
    a = _rect((4, 4), 0, 0, 2, 2)
    same = _rect((4, 4), 0, 0, 2, 2)
    shifted = _rect((4, 4), 0, 1, 2, 3)
    disjoint = _rect((4, 4), 2, 2, 4, 4)
    iou = mask_iou_matrix([a, same, disjoint], [a, shifted])
    assert iou.shape == (3, 2)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(2.0 / 6.0)  # inter 2, union 6
    assert iou[2, 0] == pytest.approx(0.0)
    # Empty inputs give empty-side-shaped zero matrices.
    assert mask_iou_matrix([], [a]).shape == (0, 1)
    assert mask_iou_matrix([a], []).shape == (1, 0)


def _one_image(
    gt: list[tuple], pred: list[tuple]
) -> tuple[list[list[Instance]], list[list[Instance]]]:
    """One-image dataset: gt entries ``(mask, class_id)``, predictions
    ``(mask, class_id, score)``."""
    gts = [[Instance(mask=m, class_id=c) for m, c in gt]]
    preds = [[Instance(mask=m, class_id=c, score=s) for m, c, s in pred]]
    return gts, preds


def test_perfect_predictions_score_one() -> None:
    gt_mask = _rect((8, 8), 1, 1, 4, 4)
    gts, preds = _one_image([(gt_mask, 1)], [(gt_mask, 1, 0.9)])
    for iou in (AP50, AP75):
        assert mask_average_precision(gts, preds, num_classes=1, iou_threshold=iou) == 1.0


def test_nothing_to_detect_is_perfect() -> None:
    gts: list[list[Instance]] = [[]]
    preds: list[list[Instance]] = [[]]
    assert mask_average_precision(gts, preds, num_classes=1, iou_threshold=AP50) == 1.0


def test_false_predictions_without_gt_score_zero() -> None:
    pred_mask = _rect((8, 8), 1, 1, 4, 4)
    gts: list[list[Instance]] = [[]]
    preds = [[Instance(mask=pred_mask, class_id=1, score=0.9)]]
    assert mask_average_precision(gts, preds, num_classes=1, iou_threshold=AP50) == 0.0


def test_ap50_vs_ap75_threshold_split() -> None:
    # IoU 0.6: a TP at AP50, an FP at AP75.
    gt_mask = _rect((8, 8), 1, 1, 5, 5)
    pred_mask = _rect((8, 8), 2, 1, 6, 5)  # inter 12, union 20 -> IoU 0.6
    assert mask_iou_matrix([pred_mask], [gt_mask])[0, 0] == pytest.approx(0.6)
    gts, preds = _one_image([(gt_mask, 1)], [(pred_mask, 1, 0.9)])
    assert mask_average_precision(gts, preds, num_classes=1, iou_threshold=AP50) == 1.0
    assert mask_average_precision(gts, preds, num_classes=1, iou_threshold=AP75) == 0.0


def test_missed_instance_lowers_ap() -> None:
    gt_a = _rect((8, 8), 1, 1, 3, 3)
    gt_b = _rect((8, 8), 5, 5, 7, 7)
    gts, preds = _one_image([(gt_a, 1), (gt_b, 1)], [(gt_a, 1, 0.9)])
    assert mask_average_precision(gts, preds, num_classes=1, iou_threshold=AP50) == 0.5


def test_fp_ranked_before_tp_halves_ap() -> None:
    gt = _rect((8, 8), 1, 1, 4, 4)
    fp = _rect((8, 8), 5, 5, 8, 8)  # disjoint from the GT
    gts, preds = _one_image([(gt, 1)], [(fp, 1, 0.9), (gt, 1, 0.8)])
    # Rank: FP (precision 0), TP (precision 0.5, recall 1) -> AP 0.5.
    assert mask_average_precision(gts, preds, num_classes=1, iou_threshold=AP50) == 0.5


def test_map_is_mean_over_classes() -> None:
    gt_a = _rect((8, 8), 1, 1, 3, 3)
    gt_b = _rect((8, 8), 5, 5, 7, 7)
    gts = [[Instance(mask=gt_a, class_id=1), Instance(mask=gt_b, class_id=2)]]
    preds = [[Instance(mask=gt_a, class_id=1, score=0.9)]]  # class 2 missed
    mAP = mask_average_precision(gts, preds, num_classes=2, iou_threshold=AP50)
    assert mAP == pytest.approx(0.5)


def test_greedy_matching_best_iou_first() -> None:
    # One prediction matches one of two overlapping GTs; the best-IoU
    # candidate is taken (both above the threshold).
    gt_a = _rect((8, 8), 1, 1, 4, 4)
    gt_b = _rect((8, 8), 2, 2, 5, 5)
    pred = _rect((8, 8), 1, 1, 4, 4)  # IoU 1.0 with gt_a, 0.33 with gt_b
    gts, preds = _one_image([(gt_a, 1), (gt_b, 1)], [(pred, 1, 0.9)])
    assert mask_average_precision(gts, preds, num_classes=1, iou_threshold=AP50) == 0.5


def test_class_absent_on_both_sides_excluded() -> None:
    gt = _rect((8, 8), 1, 1, 3, 3)
    gts, preds = _one_image([(gt, 1)], [(gt, 1, 0.9)])
    # Class 2 has neither GT nor predictions: excluded from the mean.
    assert mask_average_precision(gts, preds, num_classes=2, iou_threshold=AP50) == 1.0


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="align"):
        mask_average_precision([[]], [[], []], num_classes=1, iou_threshold=AP50)
