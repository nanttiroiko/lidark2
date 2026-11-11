"""Tests for lidark.io.products: ensure tile index and VRT mosaic on load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin
from rasterio.windows import Window
from tests.conftest import (
    ORIGIN_X,
    ORIGIN_Y,
    PIXEL_SIZE,
    TILE_SHAPE,
    tile_array,
)

from lidark.io import products
from lidark.io.products import ensure_product, ensure_tile_index, ensure_vrt
from lidark.io.rasters import write_band
from lidark.io.tiles import load_tile_index, tile_index_path
from lidark.io.vrt import open_vrt


def test_ensure_product_rebuilds_missing_artifacts(dtm_tiles: list[Path], dtm_dir: Path) -> None:
    vrt, index = ensure_product(dtm_dir)

    # Both artifacts were missing and got built.
    assert vrt == dtm_dir / "dtm.vrt"
    assert vrt.exists()
    assert tile_index_path(dtm_dir).exists()
    assert len(index) == len(dtm_tiles)
    assert load_tile_index(dtm_dir)["tile_id"].tolist() == index["tile_id"].tolist()

    # The rebuilt mosaic reads the tiles (windowed read of tile 0).
    height, width = TILE_SHAPE
    with open_vrt(vrt) as mosaic:
        tile0 = mosaic.read(1, window=Window(0, 0, width, height))
    assert np.array_equal(tile0, tile_array(0, 0))


def test_ensure_product_trusts_present_artifacts(
    dtm_product: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("present artifacts must not be rebuilt")

    monkeypatch.setattr(products, "build_tile_index", _boom)
    monkeypatch.setattr(products, "build_vrt", _boom)

    vrt, index = ensure_product(dtm_product)
    assert vrt == dtm_product / "dtm.vrt"
    assert len(index) == 4


def test_stale_index_rebuilt_with_vrt(dtm_product: Path) -> None:
    """A removed tile makes the index stale: index and VRT are rebuilt."""
    (dtm_product / "tile-number-00000004.tif").unlink()

    vrt, index = ensure_product(dtm_product)
    assert len(index) == 3
    # The rebuilt index matches the tiles on disk.
    on_disk = {p.name for p in dtm_product.glob("*.tif")}
    assert {Path(str(p)).name for p in index["path"]} == on_disk

    # The rebuilt mosaic covers the remaining tiles and leaves the removed
    # tile's footprint empty (zero fill, no source tile).
    height, width = TILE_SHAPE
    with open_vrt(vrt) as mosaic:
        kept = mosaic.read(1, window=Window(0, 0, width, height))
        removed = mosaic.read(1, window=Window(width, height, width, height), boundless=True)
    assert np.array_equal(kept, tile_array(0, 0))
    assert np.all(removed == 0)


def test_stale_index_picks_up_new_tile(dtm_product: Path) -> None:
    """A tile added after the index was built enters the rebuilt index."""
    arr = np.ones(TILE_SHAPE, dtype=np.float32)
    x0 = ORIGIN_X + 2 * TILE_SHAPE[1] * PIXEL_SIZE  # grid column 2
    transform = from_origin(x0, ORIGIN_Y, PIXEL_SIZE, PIXEL_SIZE)
    write_band(dtm_product / "extra_tile.tif", arr, transform=transform, crs="EPSG:3067")

    index = ensure_tile_index(dtm_product)
    assert len(index) == 5
    assert "extra_tile" in set(index["tile_id"])


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="product directory not found"):
        ensure_tile_index(tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match="product directory not found"):
        ensure_product(tmp_path / "nowhere")


def test_product_without_tiles_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no GeoTIFF tiles"):
        ensure_tile_index(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="no GeoTIFF tiles"):
        ensure_product(tmp_path / "empty")


def test_ensure_vrt_names_by_product(tmp_path: Path) -> None:
    """The rebuilt mosaic uses the product-specific name (find_vrt order)."""
    arr = np.ones((4, 4), dtype=np.float32)
    for name, expected in (("01_DTM", "dtm.vrt"), ("radius_20", "tpi.vrt"), ("misc", "mosaic.vrt")):
        product_dir = tmp_path / name
        product_dir.mkdir()
        write_band(
            product_dir / "a.tif",
            arr,
            transform=from_origin(0, 8, 1, 1),
            crs="EPSG:3067",
        )
        index = ensure_tile_index(product_dir)
        assert ensure_vrt(product_dir, index) == product_dir / expected


def test_ensure_vrt_reuses_present_mosaic(dtm_product: Path) -> None:
    index = ensure_tile_index(dtm_product)
    # The index is trusted as present; only the VRT lookup runs.
    assert ensure_vrt(dtm_product, index) == dtm_product / "dtm.vrt"
