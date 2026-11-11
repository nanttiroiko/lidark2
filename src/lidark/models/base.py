"""Model backend protocol and shared training types (plan section 3.1).

Every ML backend — the default U-Net today, YOLO / Faster R-CNN / Mask R-CNN
tomorrow — implements :class:`ModelBackend`. Nothing outside ``models/``
imports Keras/PyTorch, and nothing in this module is framework-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from lidark.config import (
    TASK_INSTANCE_DETECTION,
    TASK_INSTANCE_SEGMENTATION,
    TASK_SEMANTIC_SEGMENTATION,
    ModelConfig,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lidark._types import Array

__all__ = [
    "TASK_INSTANCE_DETECTION",
    "TASK_INSTANCE_SEGMENTATION",
    "TASK_SEMANTIC_SEGMENTATION",
    "TrainingData",
    "ModelBackend",
    "MODEL_FILENAME",
    "CONFIG_FILENAME",
]

# Backend-neutral artifact names inside a model directory.
MODEL_FILENAME = "model.pt"
CONFIG_FILENAME = "model_config.json"


@dataclass
class TrainingData:
    """A split, memmap-backed training dataset handed to a backend.

    Attributes:
        samples: (N, H, W, F) memmap of feature windows.
        labels: (N, H, W) uint8 per-pixel class maps (0 = background).
        train_indices: Sample indices used for training.
        val_indices: Sample indices used for validation (may be empty).
        num_classes: Number of foreground classes (>= 1).
        meta: The dataset's ``dataset_meta.json`` payload.
    """

    samples: Array
    labels: Array
    train_indices: Array
    val_indices: Array
    num_classes: int
    meta: dict[str, Any]


@runtime_checkable
class ModelBackend(Protocol):
    """The interface every ML backend implements (plan section 3.1).

    ``build`` constructs the model from a :class:`ModelConfig`; ``fit``
    trains it (best checkpoint written to ``<model_dir>/model.pt``);
    ``predict_tiles`` runs dense inference on a batch of windows;
    ``save``/``load`` persist the model plus ``model_config.json``.
    """

    @property
    def task(self) -> str:
        """One of the TASK_* constants."""
        ...

    def build(self, config: ModelConfig) -> None:
        """Construct and compile the model."""
        ...

    def fit(
        self, data: TrainingData, model_dir: Path, *, augment: bool = True
    ) -> dict[str, Sequence[float]]:
        """Train on the split dataset; returns the metric history."""
        ...

    def predict_tiles(self, tiles: Array) -> Array:
        """Score a (B, H, W, F) batch.

        Returns:
            (B, H, W) foreground probabilities for single-class models;
            (B, H, W, C) per-class probability maps for multi-class models.
        """
        ...

    def save(self, model_dir: Path) -> None:
        """Persist model metadata (``model_config.json``) to the model dir."""
        ...

    def load(self, model_dir: Path) -> None:
        """Restore a trained model and its config from the model dir."""
        ...
