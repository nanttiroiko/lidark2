"""Tests for lidark.io.rasters."""

from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from rasterio.transform import from_origin

from lidark.io.rasters import (
    read_band,
    read_crs,
    read_profile,
    read_transform,
    trim_buffer,
    write_band,
)


def test_write_read_roundtrip(tmp_path: Path) -> None:
    arr = np.linspace(1.0, 100.0, 64).reshape(8, 8).astype(np.float32)
    transform = from_origin(380000.0, 6678000.0, 2.0, 2.0)
    path = tmp_path / "tile.tif"
    write_band(path, arr, transform=transform, crs="EPSG:3067", nodata=0.0)

    assert np.array_equal(read_band(path), arr)
    profile = read_profile(path)
    assert profile["dtype"] == "float32"
    assert profile["nodata"] == 0.0
    assert read_transform(path) == transform
    assert read_crs(path).to_string() == "EPSG:3067"


def test_write_band_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "tile.tif"
    arr = np.zeros((4, 4), dtype=np.uint8)
    write_band(path, arr, transform=from_origin(0, 10, 1, 1), crs=None)
    assert path.exists()
    assert read_band(path).dtype == np.uint8


def test_write_band_rejects_non_2d(tmp_path: Path) -> None:
    arr = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        write_band(tmp_path / "x.tif", arr, transform=from_origin(0, 10, 1, 1), crs=None)


def test_trim_buffer_matches_original_math() -> None:
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    transform = from_origin(1000.0, 2000.0, 1.0, 1.0)

    trimmed, shifted = trim_buffer(arr, transform, 2.0)

    # Original: px = int(buffer / pixel_size) sliced from each side.
    px = int(2.0 / transform.a)
    assert np.array_equal(trimmed, arr[px:-px, px:-px])
    assert trimmed.shape == (6, 6)
    # Original transform shift: +buffer east, -buffer south.
    assert shifted.c == transform.c + 2.0
    assert shifted.f == transform.f - 2.0
    assert (shifted.a, shifted.b, shifted.d, shifted.e) == (
        transform.a,
        transform.b,
        transform.d,
        transform.e,
    )


def test_trim_buffer_subpixel_is_noop() -> None:
    arr = np.ones((5, 5), dtype=np.float32)
    transform = from_origin(0.0, 100.0, 1.0, 1.0)
    trimmed, shifted = trim_buffer(arr, transform, 0.5)
    assert trimmed is arr
    assert shifted == transform


def test_trim_buffer_rejects_oversized_buffer() -> None:
    arr = np.ones((4, 4), dtype=np.float32)
    transform = from_origin(0.0, 100.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="covers the array"):
        trim_buffer(arr, transform, 2.0)


def test_trim_buffer_rejects_bad_pixel_size() -> None:
    arr = np.ones((4, 4), dtype=np.float32)
    transform = Affine(0.0, 0.0, 0.0, 0.0, 1.0, 100.0)
    with pytest.raises(ValueError, match="pixel size"):
        trim_buffer(arr, transform, 1.0)
