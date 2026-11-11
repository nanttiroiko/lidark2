"""Tests for lidark.io.tiles: tile index build/load and spatial matching."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box
from tests.conftest import ORIGIN_X, ORIGIN_Y, TILE_NAMES, TILE_SHAPE

from lidark.io import tiles
from lidark.io.rasters import write_band
from lidark.io.tiles import (
    build_tile_index,
    intersecting_tiles,
    load_tile_index,
    tile_index_path,
    tile_path,
)


def test_build_and_load_index_from_arbitrary_filenames(
    dtm_tiles: list[Path], dtm_dir: Path
) -> None:
    index = build_tile_index(dtm_dir)

    assert tile_index_path(dtm_dir).exists()
    # tile_ids are filename stems: opaque labels of arbitrary lengths (1..22 chars),
    # in deterministic (case-insensitive) directory-scan order.
    expected_stems = [Path(n).stem for n in sorted(TILE_NAMES, key=str.lower)]
    assert index["tile_id"].tolist() == expected_stems
    assert {len(t) for t in index["tile_id"]} == {1, 7, 20, 23}

    loaded = load_tile_index(dtm_dir)
    assert len(loaded) == len(dtm_tiles)
    assert loaded["tile_id"].tolist() == index["tile_id"].tolist()
    # Footprints match the raster bounds (spatial, not filename, identity).
    for _, row in loaded.iterrows():
        assert row.geometry.bounds == pytest_bounds(row["path"])


def pytest_bounds(path: str):
    import rasterio

    with rasterio.open(path) as src:
        b = src.bounds
    return (b.left, b.bottom, b.right, b.top)


def test_tile_path_resolves_by_id(dtm_tiles: list[Path], dtm_dir: Path) -> None:
    index = build_tile_index(dtm_dir)
    assert tile_path(index, "Q") == (dtm_dir / "Q.tif").resolve()
    with pytest.raises(KeyError, match="not in tile index"):
        tile_path(index, "nonexistent")


def test_intersecting_tiles_strict_interior(dtm_tiles: list[Path], dtm_dir: Path) -> None:
    index = build_tile_index(dtm_dir)
    # A small polygon strictly inside tile 3 (row 1, col 1).
    x0 = ORIGIN_X + 1.5 * TILE_SHAPE[1] * 2.0
    y0 = ORIGIN_Y - 1.5 * TILE_SHAPE[0] * 2.0
    probe = box(x0, y0 - 5.0, x0 + 5.0, y0)
    hits = intersecting_tiles(index, probe)
    assert len(hits) == 1
    assert hits["tile_id"].iloc[0] == "tile-number-00000004"


def test_crs_fallback_to_default(tmp_path: Path) -> None:
    arr = np.ones((4, 4), dtype=np.float32)
    write_band(tmp_path / "no_crs.tif", arr, transform=from_origin(0, 10, 1, 1), crs=None)
    index = build_tile_index(tmp_path, default_epsg="EPSG:3067")
    assert index.crs is not None
    assert index.crs.to_string() == "EPSG:3067"


def test_crs_mismatch_raises(tmp_path: Path) -> None:
    arr = np.ones((4, 4), dtype=np.float32)
    write_band(tmp_path / "a.tif", arr, transform=from_origin(0, 10, 1, 1), crs="EPSG:3067")
    write_band(tmp_path / "b.tif", arr, transform=from_origin(0, 10, 1, 1), crs="EPSG:25832")
    with pytest.raises(ValueError, match="CRS mismatch"):
        build_tile_index(tmp_path)


def test_empty_product_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no GeoTIFF tiles"):
        build_tile_index(tmp_path / "empty")


def test_load_without_build_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tile index not found"):
        load_tile_index(tmp_path)


def test_ignores_non_tif_files(dtm_tiles: list[Path], dtm_dir: Path) -> None:
    (dtm_dir / "notes.txt").write_text("not a raster", encoding="utf-8")
    index = build_tile_index(dtm_dir)
    assert len(index) == len(dtm_tiles)
    assert tiles.INDEX_FILENAME == "tiles.gpkg"
    assert isinstance(index, gpd.GeoDataFrame)
