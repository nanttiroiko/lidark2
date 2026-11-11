"""IoU metric: batch-accumulated intersection-over-union (torch).

Binary mode reproduces the original ``Training.iou`` semantics: predictions
thresholded at 0.5, intersection and union summed over the whole batch
(accumulated across batches, as a proper metric should). Multi-class mode
computes mean IoU over the foreground classes (1..C) from the argmax of a
softmax head; class 0 (background) is excluded. An empty union counts as a
perfect score (1.0) instead of the original NaN.
"""

from __future__ import annotations

from typing import Any


class IoU:
    """Batch-accumulated intersection-over-union.

    Args:
        threshold: Decision threshold for the binary head (original: 0.5).
        num_classes: 1 for the binary sigmoid head; ``C > 1`` switches to
            mean-IoU-over-classes on a ``(C + 1)``-channel softmax head.
    """

    def __init__(self, threshold: float = 0.5, num_classes: int = 1) -> None:
        self.threshold = threshold
        self.num_classes = num_classes
        self.intersect: float = 0.0
        self.union: float = 0.0

    def update(self, logits: Any, labels: Any) -> None:
        """Accumulate one batch of raw model output.

        Args:
            logits: ``(B, C, H, W)`` raw head output (no sigmoid/softmax);
                the activation is applied here.
            labels: ``(B, H, W)`` integer class maps.
        """
        import torch

        logits_t = torch.as_tensor(logits)
        labels_t = torch.as_tensor(labels)
        if labels_t.dim() == logits_t.dim():
            labels_t = labels_t[:, 0]  # drop a redundant channel axis
        if self.num_classes <= 1:
            predicted = torch.sigmoid(logits_t[:, 0]) > self.threshold
            actual = labels_t == 1
            both = predicted & actual
            self.intersect += int(both.sum())
            self.union += int(predicted.sum()) + int(actual.sum()) - int(both.sum())
        else:
            predicted_ids = logits_t.argmax(dim=1)
            for class_id in range(1, self.num_classes + 1):
                predicted = predicted_ids == class_id
                actual = labels_t == class_id
                both = predicted & actual
                self.intersect += int(both.sum())
                self.union += int(predicted.sum()) + int(actual.sum()) - int(both.sum())

    def result(self) -> float:
        """IoU; a perfect 1.0 when nothing was predicted or labeled."""
        if self.union <= 0:
            return 1.0
        return self.intersect / self.union

    def reset(self) -> None:
        self.intersect = 0.0
        self.union = 0.0


__all__ = ["IoU"]
