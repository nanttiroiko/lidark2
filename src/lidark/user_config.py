"""Persistent user settings for the CLI (``~/.lidark/config.json``).

The stored ``project_folder`` frees every command of the ``--project-folder``
flag: ``lidark set-project-folder`` records it once, and a per-command
``--project-folder`` overrides it when needed. The file is user-level
(one folder at a time), plain JSON, and only touched by the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# User-level settings directory; overridable in tests.
CONFIG_DIR = Path.home() / ".lidark"
CONFIG_FILENAME = "config.json"


def config_path() -> Path:
    """Path of the settings file."""
    return CONFIG_DIR / CONFIG_FILENAME


def load_settings() -> dict[str, Any]:
    """Read the settings file (empty dict when absent)."""
    path = config_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"settings must be a JSON object: {path}")
    return payload


def save_settings(settings: dict[str, Any]) -> None:
    """Write the settings file, creating the directory."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def get_project_folder() -> Path | None:
    """The stored default project folder, or ``None`` when not set."""
    folder = load_settings().get("project_folder")
    return Path(str(folder)) if folder else None


def set_project_folder(path: Path | str) -> Path:
    """Store the default project folder (resolved, must exist).

    Raises:
        NotADirectoryError: If the path does not exist or is not a folder.
    """
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"project folder does not exist: {resolved}")
    settings = load_settings()
    settings["project_folder"] = str(resolved)
    save_settings(settings)
    return resolved


__all__ = [
    "CONFIG_DIR",
    "CONFIG_FILENAME",
    "config_path",
    "get_project_folder",
    "load_settings",
    "save_settings",
    "set_project_folder",
]
