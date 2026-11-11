"""Phase 0 scaffolding smoke tests."""

import logging

import lidark
from lidark.logging import configure_logging, get_logger


def test_version() -> None:
    assert isinstance(lidark.__version__, str)
    assert lidark.__version__ == "0.1.0"


def test_configure_logging_is_idempotent() -> None:
    configure_logging(verbose=False)
    first = logging.getLogger("lidark").handlers.copy()
    configure_logging(verbose=False)
    second = logging.getLogger("lidark").handlers
    assert first == list(second)


def test_verbose_switches_level() -> None:
    configure_logging(verbose=True)
    assert logging.getLogger("lidark").level == logging.DEBUG
    configure_logging(verbose=False)
    assert logging.getLogger("lidark").level == logging.INFO


def test_get_logger_namespaces_under_lidark() -> None:
    assert get_logger("io.tiles").name == "lidark.io.tiles"
    assert get_logger("lidark.io.tiles").name == "lidark.io.tiles"
