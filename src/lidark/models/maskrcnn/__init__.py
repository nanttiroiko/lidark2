"""Mask R-CNN instance segmentation backend (torchvision)."""

from lidark.models.maskrcnn.backend import WEIGHTS_FILENAME, MaskRcnnBackend

__all__ = ["MaskRcnnBackend", "WEIGHTS_FILENAME"]
