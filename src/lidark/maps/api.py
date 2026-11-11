"""``create_map`` facade replacing the original ``CreateMap`` class.

Validates ``map_type`` against ``{"DTM", "TPI"}`` and dispatches to the
modernized generators (fixes the original's broken map-type validation, B1).
No constructor side effects: each product is one callable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lidark.config import DEFAULT_EPSG, ProjectLayout

if TYPE_CHECKING:
    from collections.abc import Sequence

MAP_TYPES = ("DTM", "TPI")
DEFAULT_TPI_RADIUS = 30.0


def create_map(
    project_folder: str | Path,
    project_area: str,
    map_type: str,
    *,
    buffer: float | None = None,
    pixel_size: float = 0.25,
    cores: int = 8,
    gpu: bool = False,
    epsg: str = DEFAULT_EPSG,
) -> list[Path]:
    """Generate a data-map product for a project area.

    Args:
        project_folder: Project root (contains ``Lidar/`` and ``DeepLearning/``).
        project_area: Area name (subdirectory of ``Lidar/``).
        map_type: ``"DTM"`` or ``"TPI"``. For DTM, ``buffer`` is the point-read
            margin in metres used when reading neighbouring LAZ tiles
            (default 2.0). For TPI, ``buffer`` is the TPI radius in metres
            (default 30.0; output folder ``02_TPI/radius_<buffer>/``).
        buffer: Product buffer/radius in metres; ``None`` selects the
            per-product default.
        pixel_size: Raster resolution in metres per pixel.
        cores: Worker processes (parallel tile pipelines).
        gpu: TPI only: batch tiles through a single torch ``conv2d`` per
            chunk when torch is available.
        epsg: Default CRS (config fallback for data without CRS metadata).

    Re-running a product resumes instead of restarting: tiles whose output
    already exists (and is a readable raster at the current pixel size) are
    skipped, with one log line per skipped tile.

    Returns:
        Paths of the product tiles (written and skipped alike).

    Raises:
        ValueError: If ``map_type`` is not ``DTM`` or ``TPI``.
    """
    if map_type not in MAP_TYPES:
        raise ValueError(f"map_type must be one of {MAP_TYPES}, got {map_type!r}")
    layout = ProjectLayout(project_folder=Path(project_folder), epsg=epsg)
    if map_type == "DTM":
        from lidark.maps.dtm import DEFAULT_BUFFER, generate_dtm

        return generate_dtm(
            layout,
            project_area,
            pixel_size=pixel_size,
            buffer=DEFAULT_BUFFER if buffer is None else buffer,
            cores=cores,
        )
    from lidark.maps.tpi import create_tpi

    return create_tpi(
        layout,
        project_area,
        radius=DEFAULT_TPI_RADIUS if buffer is None else buffer,
        gpu=gpu,
        cores=cores,
    )


def create_map_types() -> Sequence[str]:
    """Supported map types (for CLI help)."""
    return MAP_TYPES
