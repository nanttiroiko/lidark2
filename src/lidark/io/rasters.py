"""GeoTIFF band IO, profile helpers, and buffer trimming."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

from lidark._types import Array


def read_band(path: Path | str, band: int = 1) -> Array:
    """Read one raster band as a numpy array."""
    with rasterio.open(path) as src:
        return np.asarray(src.read(band))


def read_profile(path: Path | str) -> dict[str, Any]:
    """Return the raster profile (metadata dict) of ``path``."""
    with rasterio.open(path) as src:
        return dict(src.profile)


def read_transform(path: Path | str) -> Affine:
    """Return the affine transform of ``path``."""
    with rasterio.open(path) as src:
        return cast(Affine, src.transform)


def read_crs(path: Path | str) -> CRS:
    """Return the CRS of ``path``, or ``None`` if the raster has none."""
    with rasterio.open(path) as src:
        return cast(CRS, src.crs)


def is_valid_tile(path: Path | str, pixel_size: float | None = None) -> bool:
    """Whether ``path`` is an existing, readable raster tile.

    Used to skip already-processed product tiles on re-runs: a missing,
    empty, or unreadable file (e.g. an interrupted write) is not valid and
    gets reprocessed. When ``pixel_size`` is given, the raster resolution
    must match it, so a product built with a different ``--pixel-size`` is
    reprocessed instead of silently kept.
    """
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return False
    try:
        with rasterio.open(p) as src:
            if src.count < 1 or src.width <= 0 or src.height <= 0:
                return False
            if pixel_size is not None:
                res_x, res_y = src.res  # both positive (y resolution is absolute)
                if not (
                    math.isclose(res_x, pixel_size, rel_tol=0.0, abs_tol=1e-12)
                    and math.isclose(res_y, pixel_size, rel_tol=0.0, abs_tol=1e-12)
                ):
                    return False
            return True
    except Exception:
        return False


def write_band(
    path: Path | str,
    arr: Array,
    *,
    transform: Affine,
    crs: CRS | str | None,
    dtype: str | np.dtype[Any] | None = None,
    nodata: float | int | None = None,
    driver: str = "GTiff",
) -> None:
    """Write a single-band raster; parent directories are created.

    Args:
        path: Output raster path.
        arr: 2-D array to write.
        transform: Affine transform (pixel-to-world) of ``arr``.
        crs: CRS written to the raster; strings are parsed with
            :meth:`rasterio.crs.CRS.from_string`.
        dtype: Output dtype; defaults to ``arr.dtype``.
        nodata: Nodata value, or ``None`` for none.
        driver: GDAL driver (default GeoTIFF).
    """
    out_path = Path(path)
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D band, got shape {arr.shape}")
    out_dtype = np.dtype(dtype if dtype is not None else arr.dtype)
    crs_obj = CRS.from_string(crs) if isinstance(crs, str) else crs
    profile: dict[str, Any] = {
        "driver": driver,
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": out_dtype.name,
        "transform": transform,
        "crs": crs_obj,
        "nodata": nodata,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.asarray(arr, dtype=out_dtype), 1)


def trim_buffer(arr: Array, transform: Affine, buffer_m: float) -> tuple[Array, Affine]:
    """Trim ``buffer_m`` metres from each edge and shift the transform.

    Reproduces the original TPI buffer handling exactly: ``int(buffer_m /
    pixel_size)`` pixels are sliced from each side, and the origin shifts east
    by ``+buffer_m`` and south by ``-buffer_m``.

    Args:
        arr: 2-D array.
        transform: Affine transform of ``arr``.
        buffer_m: Buffer width in world units (metres).

    Returns:
        ``(trimmed_array, shifted_transform)``; unchanged inputs when
        ``buffer_m`` is smaller than one pixel.

    Raises:
        ValueError: If the buffer covers the whole array or the pixel size is
            not positive.
    """
    pixel_size = transform.a
    if pixel_size <= 0:
        raise ValueError(f"pixel size must be positive, got {pixel_size}")
    px = int(buffer_m / pixel_size)
    if px <= 0:
        return arr, transform
    if 2 * px >= min(arr.shape[0], arr.shape[1]):
        raise ValueError(f"buffer of {buffer_m} ({px} px) covers the array of shape {arr.shape}")
    trimmed = arr[px:-px, px:-px]
    shifted = Affine(
        transform.a,
        transform.b,
        transform.c + buffer_m,
        transform.d,
        transform.e,
        transform.f - buffer_m,
    )
    return trimmed, shifted
