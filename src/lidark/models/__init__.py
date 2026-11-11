"""ML backend abstraction: protocol, registry, and backend implementations."""

from lidark.models.base import TASK_SEMANTIC_SEGMENTATION, ModelBackend, TrainingData
from lidark.models.registry import create_backend, known_backends, register

__all__ = [
    "TASK_SEMANTIC_SEGMENTATION",
    "ModelBackend",
    "TrainingData",
    "create_backend",
    "known_backends",
    "register",
]
