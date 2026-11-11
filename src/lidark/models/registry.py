"""Backend registry: name -> factory (plan section 3.1).

Backends register a *lazy* factory so that selecting — or merely listing —
backends never imports their framework: importing torch happens only
when a ``"unet"`` backend is actually created.
"""

from __future__ import annotations

from collections.abc import Callable

from lidark.models.base import ModelBackend

Factory = Callable[[], ModelBackend]

_REGISTRY: dict[str, Factory] = {}


def register(name: str, factory: Factory) -> None:
    """Register a backend factory under ``name`` (idempotent overwrite)."""
    _REGISTRY[name] = factory


def create_backend(name: str) -> ModelBackend:
    """Instantiate the backend selected by ``ModelConfig.backend``.

    Raises:
        ValueError: If the name is unknown; lists the known backends.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise ValueError(f"unknown model backend {name!r}; known backends: {known}") from None
    return factory()


def known_backends() -> tuple[str, ...]:
    """Names of the registered backends."""
    return tuple(sorted(_REGISTRY))


def _unet_factory() -> ModelBackend:
    from lidark.models.unet.backend import UnetBackend

    return UnetBackend()


def _maskrcnn_factory() -> ModelBackend:
    from lidark.models.maskrcnn.backend import MaskRcnnBackend

    return MaskRcnnBackend()


def _fasterrcnn_factory() -> ModelBackend:
    from lidark.models.fasterrcnn.backend import FasterRcnnBackend

    return FasterRcnnBackend()


# The default backend; its torch import stays inside the factory.
register("unet", _unet_factory)
# Instance segmentation (torchvision); the torch import stays inside the
# factory, so listing backends works without lidark[torch] installed.
register("maskrcnn", _maskrcnn_factory)
# Object detection (torchvision); same lazy-import guarantee.
register("fasterrcnn", _fasterrcnn_factory)
