"""Logging configuration for lidark.

User-facing output goes through the :mod:`logging` module instead of
``print``. The ``lidark`` logger hierarchy is configured once by
:func:`configure_logging` (called by the CLI; ``-v``/``--verbose`` switches it
to DEBUG). Library modules obtain loggers through :func:`get_logger`.
"""

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(verbose: bool = False) -> None:
    """Configure the ``lidark`` logger hierarchy.

    Idempotent: safe to call more than once (only the level is updated).

    Args:
        verbose: If True, emit DEBUG-level records; INFO otherwise.
    """
    logger = logging.getLogger("lidark")
    level = logging.DEBUG if verbose else logging.INFO
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a logger in the ``lidark`` hierarchy.

    Args:
        name: Module name (e.g. ``"io.tiles"``); the ``lidark.`` prefix is
            added when absent.

    Returns:
        A :class:`logging.Logger` under the ``lidark`` hierarchy.
    """
    if not name.startswith("lidark"):
        name = f"lidark.{name}"
    return logging.getLogger(name)
