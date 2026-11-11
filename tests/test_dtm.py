"""Tests for lidark.maps.dtm: pipeline construction and PDAL execution."""

from pathlib import Path

import numpy as np
import pytest
from tests.conftest import AREA
from tests.lasfile import write_las

from lidark.maps import dtm
from lidark.maps.dtm import (
    DTM_NODATA,
    Bounds,
    expand_bounds,
    generate_dtm,
    las_bounds,
    snap_bounds,
    tile_pipeline,
)

# ---------------------------------------------------------------- pipeline dicts


def test_snap_bounds() -> None:
    bounds: Bounds = (10.3, -5.7, 20.1, 4.2)
    assert snap_bounds(bounds, 1.0) == (10.0, -6.0, 21.0, 5.0)
    assert snap_bounds(bounds, 0.5) == (10.0, -6.0, 20.5, 4.5)


def test_expand_bounds() -> None:
    assert expand_bounds((0.0, 0.0, 10.0, 10.0), 5.0) == (-5.0, -5.0, 15.0, 15.0)


def test_tile_pipeline_structure(tmp_path: Path) -> None:
    tile_bounds = (0.0, 0.0, 100.0, 100.0)
    crop_bounds = (-2.0, -2.0, 102.0, 102.0)
    pipe = tile_pipeline(
        [tmp_path / "a.laz", tmp_path / "b.laz"],
        tile_bounds,
        crop_bounds,
        tmp_path / "tile_DTM.tif",
        0.5,
    )
    stages = pipe["pipeline"]
    # Neighbour reads -> merge -> ground -> crop -> TIN -> raster, no
    # intermediate point cloud written anywhere.
    assert [s["type"] for s in stages] == [
        "readers.las",
        "readers.las",
        "filters.merge",
        "filters.range",
        "filters.crop",
        "filters.delaunay",
        "filters.faceraster",
        "writers.raster",
    ]
    assert stages[0]["filename"] == str(tmp_path / "a.laz")
    assert stages[3]["limits"] == "Classification[2:2]"  # -keep_class 2
    assert stages[4]["bounds"] == "([-2.0,102.0],[-2.0,102.0])"
    face = stages[6]
    assert face["resolution"] == 0.5
    assert face["nodata"] == DTM_NODATA
    # The raster grid is pinned to the original tile bounds; origin_y is the
    # bottom edge.
    assert face["origin_x"] == 0.0
    assert face["origin_y"] == 0.0
    assert face["width"] == 200
    assert face["height"] == 200
    writer = stages[7]
    assert writer["gdaldriver"] == "GTiff"
    assert writer["data_type"] == "float"
    assert writer["nodata"] == DTM_NODATA


def test_intersecting_selects_neighbours(tmp_path: Path) -> None:
    tiles = [
        ("a", (0.0, 0.0, 10.0, 10.0), tmp_path / "a.laz", tmp_path / "a.tif"),
        ("b", (10.0, 0.0, 20.0, 10.0), tmp_path / "b.laz", tmp_path / "b.tif"),
        ("far", (100.0, 0.0, 110.0, 10.0), tmp_path / "far.laz", tmp_path / "far.tif"),
    ]
    # Tile a + 2 m buffer reaches into tile b, but not into "far".
    assert dtm._intersecting(tiles, (0.0, 0.0, 10.0, 10.0), 2.0) == [
        tmp_path / "a.laz",
        tmp_path / "b.laz",
    ]


def test_generate_dtm_requires_originals(layout) -> None:
    with pytest.raises(FileNotFoundError, match="no .las/.laz files"):
        generate_dtm(layout, AREA)


def test_generate_dtm_rejects_bad_args(layout) -> None:
    with pytest.raises(ValueError, match="pixel_size"):
        generate_dtm(layout, AREA, pixel_size=0.0)
    with pytest.raises(ValueError, match="buffer"):
        generate_dtm(layout, AREA, buffer=-1.0)
    with pytest.raises(ValueError, match="cores"):
        generate_dtm(layout, AREA, cores=0)


def test_bounds_str() -> None:
    assert dtm._bounds_str((1.0, 2.0, 3.0, 4.0)) == "([1.0,3.0],[2.0,4.0])"


# ---------------------------------------------------------------- PDAL execution

pdal = pytest.importorskip("pdal", reason="PDAL not installed")


def _write_synthetic_area(layout, area: str, n_tiles: int = 2) -> None:
    """Adjacent 40x40 m LAS tiles with ground (class 2, z~10) and noise points.

    A dense cluster of non-ground points (class 1, z=1000) sits in the middle
    of the first tile.
    """
    originals = layout.originals_dir(area)
    rng = np.random.default_rng(11)
    for i in range(n_tiles):
        x0 = 1000.0 + 40.0 * i
        xs = rng.uniform(x0, x0 + 40.0, size=2500)
        ys = rng.uniform(2000.0, 2040.0, size=2500)
        zs = 10.0 + np.sin(xs / 5.0) + np.cos(ys / 7.0)
        cls = np.full(2500, 2, dtype=np.uint8)
        if i == 0:
            # Non-ground cluster far above the terrain.
            xs = np.concatenate([xs, rng.uniform(1015.0, 1025.0, size=300)])
            ys = np.concatenate([ys, rng.uniform(2015.0, 2025.0, 300)])
            zs = np.concatenate([zs, np.full(300, 1000.0)])
            cls = np.concatenate([cls, np.full(300, 1, dtype=np.uint8)])
        write_las(originals / f"orig_{i}.las", xs, ys, zs, cls)


@pytest.fixture()
def synthetic_area(layout) -> None:
    _write_synthetic_area(layout, AREA)


def test_las_bounds_reads_header(synthetic_area, layout) -> None:
    path = layout.originals_dir(AREA) / "orig_0.las"
    bounds = las_bounds(path)
    # Random points inside [1000, 1040] x [2000, 2040].
    assert 1000.0 < bounds[0] < 1001.0
    assert 1039.0 < bounds[2] < 1040.0
    assert 2000.0 < bounds[1] < 2001.0
    assert 2039.0 < bounds[3] < 2040.0


def test_generate_dtm_tiles_clipped_to_tile_bounds(synthetic_area, layout) -> None:
    tile_paths = generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=1)
    assert len(tile_paths) == 2
    assert {p.name for p in tile_paths} == {"orig_0_DTM.tif", "orig_1_DTM.tif"}

    import rasterio

    for i, path in enumerate(tile_paths):
        with rasterio.open(path) as src:
            assert src.res == (1.0, 1.0)
            assert src.dtypes[0] == "float32"
            assert src.nodata == DTM_NODATA
            # Output tiles are clipped to the original (snapped) tile bounds:
            # 40x40 m at 1 m pixels, no buffer overlap.
            assert src.width == 40 and src.height == 40
            assert src.bounds.left == pytest.approx(1000.0 + 40.0 * i)
            assert src.bounds.top == pytest.approx(2040.0)
            arr = src.read(1)
        if i == 0:
            # Ground-only: the z=1000 non-ground cluster must not leak in.
            assert arr.max() < 100.0
            # The TIN covers the tile interior; only the boundary sliver
            # beyond the point cloud is nodata.
            interior = arr[3:-3, 3:-3]
            assert np.all(interior != DTM_NODATA)
            assert 7.0 < float(interior.mean()) < 13.0

    # Adjacent tiles abut exactly: no overlap, no gap.
    with rasterio.open(tile_paths[0]) as a, rasterio.open(tile_paths[1]) as b:
        assert a.bounds.right == pytest.approx(b.bounds.left)

    # No intermediate point clouds anywhere next to the originals.
    assert not (layout.originals_dir(AREA) / "ground_intermediate").exists()

    # Product artifacts: mosaic + tile index with opaque tile IDs and the
    # (unbuffered) tile footprints.
    assert (layout.dtm_dir(AREA) / "dtm.vrt").exists()
    from lidark.io.tiles import load_tile_index

    index = load_tile_index(layout.dtm_dir(AREA))
    assert index["tile_id"].tolist() == ["orig_0", "orig_1"]
    assert index.geometry.iloc[0].bounds == pytest.approx((1000.0, 2000.0, 1040.0, 2040.0))


def test_generate_dtm_ground_free_tile_is_all_nodata(layout) -> None:
    """A tile without ground points yields an all-nodata DTM tile, gracefully."""
    originals = layout.originals_dir(AREA)
    rng = np.random.default_rng(12)
    # Tile 0: normal ground. Tile 1: class 1 (noise) only.
    xs = rng.uniform(1000.0, 1040.0, size=2500)
    ys = rng.uniform(2000.0, 2040.0, size=2500)
    zs = 10.0 + np.sin(xs / 5.0)
    write_las(originals / "orig_0.las", xs, ys, zs, np.full(2500, 2, dtype=np.uint8))
    xs = rng.uniform(1040.0, 1080.0, size=500)
    ys = rng.uniform(2000.0, 2040.0, size=500)
    zs = np.full(500, 100.0)
    write_las(originals / "orig_1.las", xs, ys, zs, np.full(500, 1, dtype=np.uint8))

    tile_paths = generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=1)
    import rasterio

    with rasterio.open(tile_paths[0]) as src:
        assert (src.read(1) != DTM_NODATA).any()
    with rasterio.open(tile_paths[1]) as src:
        arr = src.read(1)
        assert src.nodata == DTM_NODATA
        assert np.all(arr == DTM_NODATA)


def test_generate_dtm_parallel_matches_serial(synthetic_area, layout) -> None:
    serial = generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=1)
    import rasterio

    serial_arrays = [rasterio.open(p).read(1) for p in serial]
    # Existing tiles are skipped on re-runs; delete them to force reprocessing.
    for p in serial:
        p.unlink()
    parallel = generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=2)
    for path, arr in zip(parallel, serial_arrays, strict=True):
        np.testing.assert_array_equal(rasterio.open(path).read(1), arr)


def test_dtm_to_tpi_integration(synthetic_area, layout) -> None:
    """Full Phase 2 chain: DTM (neighbour reads) -> TPI from the DTM product."""
    from lidark.io.tiles import load_tile_index
    from lidark.maps.tpi import TPI_NODATA, create_tpi

    generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=1)
    tpi_paths = create_tpi(layout, AREA, radius=4.0, dtype=np.float64)
    assert len(tpi_paths) == 2
    import rasterio

    for path in tpi_paths:
        with rasterio.open(path) as src:
            assert src.dtypes[0] == "int16"
            assert src.nodata == TPI_NODATA
            # TPI tiles cover the DTM tile footprints exactly (40x40 px).
            assert src.width == 40 and src.height == 40
            arr = src.read(1)
            # Nodata DTM cells (boundary sliver) propagate as TPI nodata;
            # the interior is finite and in range.
            interior = arr[5:-5, 5:-5]
            assert np.all(interior != TPI_NODATA)
            assert int(interior.min()) >= -100 and int(interior.max()) <= 100
    tpi_index = load_tile_index(layout.tpi_dir(AREA, 4.0))
    assert tpi_index["tile_id"].tolist() == ["orig_0", "orig_1"]


def test_generate_dtm_skips_unreadable_laz(layout, synthetic_area) -> None:
    """A corrupt LAZ file is skipped with a warning; the other tiles are
    still processed and the product artifacts are written."""
    from lidark.io.tiles import load_tile_index

    originals = layout.originals_dir(AREA)
    (originals / "corrupt.laz").write_bytes(b"this is not a LAZ file")
    tile_paths = generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=1)
    assert {p.name for p in tile_paths} == {"orig_0_DTM.tif", "orig_1_DTM.tif"}
    assert (layout.dtm_dir(AREA) / "dtm.vrt").exists()
    index = load_tile_index(layout.dtm_dir(AREA))
    assert index["tile_id"].tolist() == ["orig_0", "orig_1"]


def test_generate_dtm_pipeline_failure_writes_nodata_tile(
    layout, synthetic_area, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One failing tile pipeline becomes an all-nodata tile; the run completes."""
    import json

    import rasterio

    from lidark.io.tiles import load_tile_index

    real_run = dtm._run_pipeline

    def flaky_run(pipe):
        if "orig_1_DTM.tif" in json.dumps(pipe):
            raise RuntimeError("simulated PDAL failure")
        return real_run(pipe)

    monkeypatch.setattr(dtm, "_run_pipeline", flaky_run)
    tile_paths = generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=1)
    assert {p.name for p in tile_paths} == {"orig_0_DTM.tif", "orig_1_DTM.tif"}
    with rasterio.open(layout.dtm_dir(AREA) / "orig_0_DTM.tif") as src:
        assert (src.read(1) != DTM_NODATA).any()
    with rasterio.open(layout.dtm_dir(AREA) / "orig_1_DTM.tif") as src:
        assert src.nodata == DTM_NODATA
        assert np.all(src.read(1) == DTM_NODATA)
    assert (layout.dtm_dir(AREA) / "dtm.vrt").exists()
    assert load_tile_index(layout.dtm_dir(AREA))["tile_id"].tolist() == ["orig_0", "orig_1"]


def test_generate_dtm_unwritable_tile_is_excluded(
    layout, synthetic_area, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tile whose pipeline AND fallback write both fail is excluded from
    the mosaic and tile index; the run still completes with the rest."""
    import json

    from lidark.io.tiles import load_tile_index

    real_run = dtm._run_pipeline
    real_write = dtm._write_nodata_tile

    def flaky_run(pipe):
        if "orig_1_DTM.tif" in json.dumps(pipe):
            raise RuntimeError("simulated PDAL failure")
        return real_run(pipe)

    def flaky_write(out_path, bounds, pixel_size, epsg):
        if out_path.name == "orig_1_DTM.tif":
            raise OSError("simulated disk failure")
        return real_write(out_path, bounds, pixel_size, epsg)

    monkeypatch.setattr(dtm, "_run_pipeline", flaky_run)
    monkeypatch.setattr(dtm, "_write_nodata_tile", flaky_write)
    tile_paths = generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=1)
    assert [p.name for p in tile_paths] == ["orig_0_DTM.tif"]
    assert (layout.dtm_dir(AREA) / "dtm.vrt").exists()
    assert not (layout.dtm_dir(AREA) / "orig_1_DTM.tif").exists()
    assert load_tile_index(layout.dtm_dir(AREA))["tile_id"].tolist() == ["orig_0"]


def test_generate_dtm_broken_pool_falls_back_to_serial(
    layout, synthetic_area, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker pool that breaks (native crash) re-runs the unfinished tiles
    serially; the run completes with all tiles."""
    from concurrent.futures.process import BrokenProcessPool

    class FakePool:
        def __init__(self, max_workers: int) -> None:
            pass

        def __enter__(self) -> "FakePool":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def map(self, fn, jobs):
            raise BrokenProcessPool("simulated native crash")

    monkeypatch.setattr(dtm, "ProcessPoolExecutor", FakePool)
    tile_paths = generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=4)
    assert {p.name for p in tile_paths} == {"orig_0_DTM.tif", "orig_1_DTM.tif"}


def test_generate_dtm_rerun_skips_processed_tiles(
    layout, synthetic_area, lidark_log_records: list[str]
) -> None:
    """Re-running skips every existing tile, logs one line per skipped tile,
    and leaves the product (files, VRT, index) complete and untouched."""
    from lidark.io.tiles import load_tile_index

    first = generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=1)
    assert len(first) == 2
    mtimes = {p.name: p.stat().st_mtime_ns for p in first}

    lidark_log_records.clear()
    second = generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=1)
    assert [p.name for p in second] == [p.name for p in first]
    # Nothing was recomputed.
    assert {p.name: p.stat().st_mtime_ns for p in second} == mtimes
    skipped = [
        m for m in lidark_log_records if m.startswith("skipping ") and "already processed" in m
    ]
    assert len(skipped) == 2
    assert any("orig_0_DTM.tif" in m for m in skipped)
    assert any("orig_1_DTM.tif" in m for m in skipped)
    # Product artifacts still complete.
    assert (layout.dtm_dir(AREA) / "dtm.vrt").exists()
    assert load_tile_index(layout.dtm_dir(AREA))["tile_id"].tolist() == ["orig_0", "orig_1"]


def test_generate_dtm_resume_recomputes_only_missing_or_invalid(
    layout, lidark_log_records: list[str]
) -> None:
    """Resume after an interruption: only missing or unreadable output tiles
    are recomputed; valid ones are skipped (and logged)."""
    import rasterio

    _write_synthetic_area(layout, AREA, n_tiles=3)
    first = generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=1)
    assert len(first) == 3
    dtm_dir = layout.dtm_dir(AREA)
    mtimes = {p.name: p.stat().st_mtime_ns for p in first}
    (dtm_dir / "orig_1_DTM.tif").unlink()  # missing -> recompute
    (dtm_dir / "orig_2_DTM.tif").write_bytes(b"corrupt")  # unreadable -> recompute

    lidark_log_records.clear()
    second = generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=1)
    assert len(second) == 3
    # orig_0 was valid: skipped, file untouched, one log line.
    assert (dtm_dir / "orig_0_DTM.tif").stat().st_mtime_ns == mtimes["orig_0_DTM.tif"]
    skipped = [
        m for m in lidark_log_records if m.startswith("skipping ") and "already processed" in m
    ]
    assert len(skipped) == 1 and "orig_0_DTM.tif" in skipped[0]
    # orig_1/orig_2 recomputed and readable again at the right grid.
    for name in ("orig_1_DTM.tif", "orig_2_DTM.tif"):
        assert (dtm_dir / name).stat().st_mtime_ns != mtimes[name]
        with rasterio.open(dtm_dir / name) as src:
            assert src.res == (1.0, 1.0)
            assert src.width == 40 and src.height == 40


def test_generate_dtm_pixel_size_change_reprocesses(layout, synthetic_area) -> None:
    """A product built with a different --pixel-size is not skipped: the
    resolution check invalidates the existing tiles and they are rebuilt."""
    import rasterio

    generate_dtm(layout, AREA, pixel_size=1.0, buffer=2.0, cores=1)
    paths = generate_dtm(layout, AREA, pixel_size=0.5, buffer=2.0, cores=1)
    assert len(paths) == 2
    for p in paths:
        with rasterio.open(p) as src:
            assert src.res == (0.5, 0.5)
            assert src.width == 80 and src.height == 80
