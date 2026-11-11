"""Faster R-CNN object detection backend (torchvision)."""

from lidark.models.fasterrcnn.backend import WEIGHTS_FILENAME, FasterRcnnBackend

__all__ = ["FasterRcnnBackend", "WEIGHTS_FILENAME"]
