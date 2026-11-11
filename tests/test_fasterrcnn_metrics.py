"""Tests for the Faster R-CNN box-mAP metric and box targets (numpy only)."""

from __future__ import annotations

import numpy as np
import pytest

from lidark.models.fasterrcnn.metrics import (
    AP50,
    AP75,
    Box,
    box_average_precision,
    box_iou_matrix,
    boxes_from_label,
)


def test_boxes_from_label_component_boxes_and_classes() -> None:
    label = np.zeros((8, 8), dtype=np.uint8)
    label[0:2, 0:2] = 1
    label[5:7, 5:7] = 1
    label[3:5, 3:5] = 2
    boxes = boxes_from_label(label)
    assert [b.class_id for b in boxes] == [1, 1, 2]
    assert all(b.score == 1.0 for b in boxes)
    assert boxes[0].xyxy == (0.0, 0.0, 2.0, 2.0)
    assert boxes[1].xyxy == (5.0, 5.0, 7.0, 7.0)
    assert boxes[2].xyxy == (3.0, 3.0, 5.0, 5.0)


def test_boxes_from_label_empty_map() -> None:
    assert boxes_from_label(np.zeros((4, 4), dtype=np.uint8)) == []


def test_box_iou_matrix() -> None:
    a = (0.0, 0.0, 2.0, 2.0)
    shifted = (1.0, 0.0, 3.0, 2.0)
    disjoint = (3.0, 3.0, 4.0, 4.0)
    iou = box_iou_matrix([Box(a, 1), Box(a, 1), Box(disjoint, 1)], [Box(a, 1), Box(shifted, 1)])
    assert iou.shape == (3, 2)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(2.0 / 6.0)  # inter 2, union 6
    assert iou[2, 0] == pytest.approx(0.0)
    # Empty inputs give empty-side-shaped zero matrices.
    assert box_iou_matrix([], [Box(a, 1)]).shape == (0, 1)
    assert box_iou_matrix([Box(a, 1)], []).shape == (1, 0)


def _one_image(gt: list[tuple], pred: list[tuple]) -> tuple[list[list[Box]], list[list[Box]]]:
    """One-image dataset: gt entries ``(xyxy, class_id)``, predictions
    ``(xyxy, class_id, score)``."""
    gts = [[Box(xyxy=b, class_id=c) for b, c in gt]]
    preds = [[Box(xyxy=b, class_id=c, score=s) for b, c, s in pred]]
    return gts, preds


def test_perfect_predictions_score_one() -> None:
    gt_box = (1.0, 1.0, 4.0, 4.0)
    gts, preds = _one_image([(gt_box, 1)], [(gt_box, 1, 0.9)])
    for iou in (AP50, AP75):
        assert box_average_precision(gts, preds, num_classes=1, iou_threshold=iou) == 1.0


def test_nothing_to_detect_is_perfect() -> None:
    gts: list[list[Box]] = [[]]
    preds: list[list[Box]] = [[]]
    assert box_average_precision(gts, preds, num_classes=1, iou_threshold=AP50) == 1.0


def test_false_predictions_without_gt_score_zero() -> None:
    preds = [[Box(xyxy=(1.0, 1.0, 4.0, 4.0), class_id=1, score=0.9)]]
    assert box_average_precision([[]], preds, num_classes=1, iou_threshold=AP50) == 0.0


def test_voc_ap50_vs_ap75_threshold_split() -> None:
    # IoU 0.6: a TP at mAP@0.5, an FP at AP75.
    gt_box = (1.0, 1.0, 5.0, 5.0)  # 16 px
    pred_box = (2.0, 1.0, 6.0, 5.0)  # 16 px, inter 12, union 20 -> IoU 0.6
    assert box_iou_matrix([Box(pred_box, 1)], [Box(gt_box, 1)])[0, 0] == pytest.approx(0.6)
    gts, preds = _one_image([(gt_box, 1)], [(pred_box, 1, 0.9)])
    assert box_average_precision(gts, preds, num_classes=1, iou_threshold=AP50) == 1.0
    assert box_average_precision(gts, preds, num_classes=1, iou_threshold=AP75) == 0.0


def test_missed_instance_lowers_ap() -> None:
    gt_a = (1.0, 1.0, 3.0, 3.0)
    gt_b = (5.0, 5.0, 7.0, 7.0)
    gts, preds = _one_image([(gt_a, 1), (gt_b, 1)], [(gt_a, 1, 0.9)])
    assert box_average_precision(gts, preds, num_classes=1, iou_threshold=AP50) == 0.5


def test_fp_ranked_before_tp_halves_ap() -> None:
    gt = (1.0, 1.0, 4.0, 4.0)
    fp = (5.0, 5.0, 8.0, 8.0)  # disjoint from the GT
    gts, preds = _one_image([(gt, 1)], [(fp, 1, 0.9), (gt, 1, 0.8)])
    # Rank: FP (precision 0), TP (precision 0.5, recall 1) -> AP 0.5.
    assert box_average_precision(gts, preds, num_classes=1, iou_threshold=AP50) == 0.5


def test_map_is_mean_over_classes() -> None:
    gt_a = (1.0, 1.0, 3.0, 3.0)
    gt_b = (5.0, 5.0, 7.0, 7.0)
    gts = [[Box(xyxy=gt_a, class_id=1), Box(xyxy=gt_b, class_id=2)]]
    preds = [[Box(xyxy=gt_a, class_id=1, score=0.9)]]  # class 2 missed
    mAP = box_average_precision(gts, preds, num_classes=2, iou_threshold=AP50)
    assert mAP == pytest.approx(0.5)


def test_class_absent_on_both_sides_excluded() -> None:
    gt = (1.0, 1.0, 3.0, 3.0)
    gts, preds = _one_image([(gt, 1)], [(gt, 1, 0.9)])
    # Class 2 has neither GT nor predictions: excluded from the mean.
    assert box_average_precision(gts, preds, num_classes=2, iou_threshold=AP50) == 1.0


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="align"):
        box_average_precision([[]], [[], []], num_classes=1, iou_threshold=AP50)
