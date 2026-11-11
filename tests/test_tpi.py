"""Tests for lidark.maps.tpi: kernel, post-processing, and the tile driver."""

from pathlib import Path

import numpy as np
import pytest
from tests.conftest import (
    AREA,
    DEFAULT_EPSG,
    ORIGIN_X,
    ORIGIN_Y,
    PIXEL_SIZE,
    TILE_SHAPE,
    tile_transform,
)

from lidark.io.rasters import read_band, read_transform, write_band
from lidark.io.tiles import build_tile_index, load_tile_index
from lidark.io.vrt import build_vrt
from lidark.maps.tpi import (
    TPI_NODATA,
    circular_window,
    create_tpi,
    tpi_kernel,
    tpi_postprocess,
    tpi_reference,
)


def test_circular_window_weights() -> None:
    win = circular_window(2)
    assert win.shape == (5, 5)
    assert win[2, 2] == 0.0  # center excluded
    assert win[2, 0] == 1.0  # dist 2 < 2.1: full weight
    assert win[1, 1] == 1.0  # dist sqrt(2): full weight
    assert win[0, 2] == 1.0  # dist 2: full weight
    assert win[1, 0] == 0.5  # dist sqrt(5) ~ 2.24: edge ring
    assert win[0, 1] == 0.5
    assert win[0, 0] == 0.0  # dist sqrt(8) ~ 2.83 >= 2.5: outside
    with pytest.raises(ValueError, match="radius_px"):
        circular_window(0)


def test_kernel_circular_weights() -> None:
    """Full weight inside the radius, half weight on the edge ring, zero
    beyond the circle — verified against a hand-computed weighted mean."""
    arr = np.full((9, 9), 50.0, dtype=np.float32)
    c = 4
    arr[c, c + 2] = 90.0  # dist 2: full weight
    arr[c - 1, c - 2] = 60.0  # dist sqrt(5) ~ 2.24: edge ring, half weight
    arr[c + 2, c + 2] = 1000.0  # dist sqrt(8) ~ 2.83: outside the circle
    out = tpi_kernel(arr, 2, dtype=np.float64)
    # Radius-2 window: 12 cells at weight 1.0 + 8 cells at weight 0.5 = 16.
    expected_mean = (16 * 50.0 + 1.0 * 40.0 + 0.5 * 10.0) / 16.0
    assert out[c, c] == pytest.approx(50.0 - expected_mean)


def test_kernel_rejects_bad_arguments() -> None:
    arr = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        tpi_kernel(np.ones((2, 4, 4)), 1)
    with pytest.raises(ValueError, match="radius_px"):
        tpi_kernel(arr, 0)
    with pytest.raises(ValueError, match="float"):
        tpi_kernel(arr, 1, dtype=np.int32)


def test_kernel_flat_surface_is_zero() -> None:
    # A plateau has TPI 0 everywhere (z - mean of identical neighbours).
    arr = np.full((16, 16), 50.0, dtype=np.float32)
    out = tpi_kernel(arr, 3, dtype=np.float64)
    assert np.allclose(out, 0.0)


def test_kernel_excludes_nodata_neighbours() -> None:
    # Nodata neighbours are excluded from the mean: they do not prevent
    # computation. Only the nodata center itself is NaN.
    arr = np.full((16, 16), 50.0, dtype=np.float32)
    arr[8, 8] = 0.0  # nodata: values <= 0 are ignored
    out = tpi_kernel(arr, 3, dtype=np.float64)
    assert np.isnan(out[8, 8])  # nodata center -> nodata
    assert out[9, 8] == 0.0  # valid neighbour: the mean excludes the nodata cell
    assert np.isfinite(out).sum() == 16 * 16 - 1


def test_kernel_no_valid_neighbours_is_nodata() -> None:
    # An isolated valid cell has no valid neighbour to compare against.
    arr = np.zeros((8, 8), dtype=np.float32)  # all nodata
    arr[4, 4] = 42.0
    out = tpi_kernel(arr, 2, dtype=np.float64)
    assert np.all(np.isnan(out))


def test_kernel_uses_available_neighbours_at_borders() -> None:
    # Border pixels average only their in-bounds neighbours (original view
    # behaviour); verified against the reference implementation.
    rng = np.random.default_rng(3)
    arr = rng.uniform(1.0, 100.0, size=(9, 9)).astype(np.float32)
    arr[0, 0] = 0.0  # nodata in the corner
    out = tpi_kernel(arr, 4, dtype=np.float64)
    ref = tpi_reference(arr, 4, dtype=np.float64)
    np.testing.assert_array_equal(out, ref)


def test_postprocess_maps_nodata_to_constant() -> None:
    out = np.array([[np.nan, 0.5], [-0.6, 200.0]])
    result = tpi_postprocess(out)
    expected = np.array([[TPI_NODATA, 50], [-60, 100]], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.int16


def test_postprocess_truncates_toward_zero() -> None:
    result = tpi_postprocess(np.array([[0.999, -0.999]]))
    np.testing.assert_array_equal(result, np.array([[99, -99]], dtype=np.int16))


def test_create_tpi_produces_tiles(layout, dtm_product: Path) -> None:
    radius = 4.0  # metres; pixel size 2 m -> radius_px = 2
    paths = create_tpi(layout, AREA, radius, dtype=np.float64)
    index = load_tile_index(layout.dtm_dir(AREA))
    assert len(paths) == len(index) == 4

    dtm_index = load_tile_index(layout.dtm_dir(AREA))
    for _, row in dtm_index.iterrows():
        tile_id = str(row["tile_id"])
        out_path = layout.tpi_dir(AREA, radius) / f"{tile_id}_TPI.tif"
        assert out_path.exists()
        import rasterio

        with rasterio.open(out_path) as src:
            assert src.nodata == TPI_NODATA
        arr = read_band(out_path)
        assert arr.dtype == np.int16
        assert arr.shape == TILE_SHAPE
        # Output transform equals the DTM tile transform (margin trimmed).
        assert read_transform(out_path) == tile_transform(
            int((ORIGIN_Y - row.geometry.bounds[3]) // (TILE_SHAPE[0] * PIXEL_SIZE)),
            int((row.geometry.bounds[0] - ORIGIN_X) // (TILE_SHAPE[1] * PIXEL_SIZE)),
        )
        # Expected values: reference kernel on the buffered window.
        b = row.geometry.bounds
        expected = _expected_tpi(dtm_product, b, radius)
        np.testing.assert_array_equal(arr, expected)

    # VRT and tile index written for the TPI product.
    assert (layout.tpi_dir(AREA, radius) / "tpi.vrt").exists()
    tpi_index = load_tile_index(layout.tpi_dir(AREA, radius))
    assert tpi_index["tile_id"].tolist() == dtm_index["tile_id"].tolist()


def test_create_tpi_parallel_matches_serial(layout, dtm_product: Path) -> None:
    radius = 4.0
    create_tpi(layout, AREA, radius, dtype=np.float64, cores=1)
    tpi_dir = layout.tpi_dir(AREA, radius)
    serial = {p.name: read_band(p) for p in tpi_dir.glob("*_TPI.tif")}
    # Existing tiles are skipped on re-runs; delete them to force reprocessing.
    for p in tpi_dir.glob("*_TPI.tif"):
        p.unlink()
    create_tpi(layout, AREA, radius, dtype=np.float64, cores=2)
    for name, arr in serial.items():
        np.testing.assert_array_equal(read_band(tpi_dir / name), arr)


def test_create_tpi_requires_dtm(layout) -> None:
    with pytest.raises(FileNotFoundError, match="product directory not found"):
        create_tpi(layout, AREA, radius=4.0)


def test_create_tpi_rebuilds_missing_dtm_artifacts(layout, dtm_tiles: list[Path]) -> None:
    """A missing DTM VRT/tile index is rebuilt from the DTM tiles on disk,
    with output identical to a run over the fully built product."""
    dtm_dir = layout.dtm_dir(AREA)
    assert not (dtm_dir / "dtm.vrt").exists()
    assert not (dtm_dir / "tiles.gpkg").exists()

    radius = 4.0
    paths = create_tpi(layout, AREA, radius, dtype=np.float64)
    assert len(paths) == len(dtm_tiles)
    assert (dtm_dir / "dtm.vrt").exists()
    assert (dtm_dir / "tiles.gpkg").exists()

    index = load_tile_index(dtm_dir)
    for _, row in index.iterrows():
        out_path = layout.tpi_dir(AREA, radius) / f"{row['tile_id']}_TPI.tif"
        expected = _expected_tpi(dtm_dir, row.geometry.bounds, radius)
        np.testing.assert_array_equal(read_band(out_path), expected)


def test_create_tpi_rejects_bad_radius(layout, dtm_product: Path) -> None:
    with pytest.raises(ValueError, match="radius"):
        create_tpi(layout, AREA, radius=0.0)
    with pytest.raises(ValueError, match="pixel"):
        create_tpi(layout, AREA, radius=1.0)  # below the 2 m pixel size


def test_create_tpi_handles_nodata_and_missing_neighbours(layout, dtm_dir: Path) -> None:
    """Nodata cells are excluded from TPI (they do not prevent it); cells
    with no neighbouring tile (area edge) become nodata."""
    import rasterio

    # One tile with nodata holes; everything around it is missing (no
    # neighbouring tiles at the area edge).
    arr = np.full(TILE_SHAPE, 20.0, dtype=np.float32)
    arr[10:14, 10:14] = -9999.0  # nodata hole
    arr[0, 0] = -9999.0
    write_band(
        dtm_dir / "solo.tif",
        arr,
        transform=tile_transform(0, 0),
        crs=DEFAULT_EPSG,
        nodata=-9999.0,
    )
    build_vrt([dtm_dir / "solo.tif"], dtm_dir / "dtm.vrt")
    build_tile_index(dtm_dir, default_epsg=DEFAULT_EPSG)

    radius = 4.0  # pixel size 2 m -> radius_px = 2
    paths = create_tpi(layout, AREA, radius, dtype=np.float64)
    assert len(paths) == 1
    with rasterio.open(paths[0]) as src:
        assert src.nodata == TPI_NODATA
        out = src.read(1)
    assert out.shape == TILE_SHAPE
    # Nodata DTM cells propagate as TPI nodata.
    assert np.all(out[10:14, 10:14] == TPI_NODATA)
    assert out[0, 0] == TPI_NODATA
    # Every other cell is computed from its valid neighbours — including the
    # area-edge cells whose windows reach into missing neighbouring tiles
    # (filled with nodata and excluded). A flat plateau is TPI 0.
    valid = out != TPI_NODATA
    assert np.all(out[valid] == 0)
    assert np.all(np.isfinite(out[valid]))


def test_create_tpi_tile_failure_writes_nodata_tile(
    layout, dtm_product: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One failing TPI tile becomes an all-nodata tile of the right shape;
    the run completes with the VRT and a full tile index."""
    import rasterio

    import lidark.maps.tpi as tpi_module

    calls = {"n": 0}
    real_read = tpi_module._read_window

    def flaky_read(vrt_path, window):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated read failure")
        return real_read(vrt_path, window)

    monkeypatch.setattr(tpi_module, "_read_window", flaky_read)
    radius = 4.0
    paths = create_tpi(layout, AREA, radius, dtype=np.float64, cores=1)
    assert len(paths) == 4
    tpi_dir = layout.tpi_dir(AREA, radius)
    assert (tpi_dir / "tpi.vrt").exists()
    assert len(load_tile_index(tpi_dir)) == 4
    nodata_tiles = []
    for p in paths:
        with rasterio.open(p) as src:
            assert src.nodata == TPI_NODATA
            arr = src.read(1)
            assert arr.shape == TILE_SHAPE
            if np.all(arr == TPI_NODATA):
                nodata_tiles.append(p.name)
    # Exactly the failed tile is all-nodata; the synthetic DTM has valid
    # values everywhere, so no computed tile can be all-nodata.
    assert len(nodata_tiles) == 1


def test_create_tpi_gpu_batch_failure_falls_back_to_per_tile(
    layout, dtm_product: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing batched kernel degrades to per-tile CPU processing on the
    GPU path; all tiles are still written with computed values."""
    import lidark.maps.tpi as tpi_module

    def boom(batch, radius_px):
        raise RuntimeError("simulated kernel failure")

    monkeypatch.setattr(tpi_module, "_tpi_kernel_torch", boom)
    radius = 4.0
    paths = create_tpi(layout, AREA, radius, gpu=True, dtype=np.float64, cores=1)
    assert len(paths) == 4
    for p in paths:
        assert p.exists()
        assert (read_band(p) != TPI_NODATA).any()


def test_create_tpi_gpu_read_failure_writes_nodata_tile(
    layout, dtm_product: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the GPU path a failed window read becomes an all-nodata tile; the
    remaining tiles are computed normally."""
    import lidark.maps.tpi as tpi_module

    calls = {"n": 0}
    real_read = tpi_module._read_window

    def flaky_read(vrt_path, window):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated read failure")
        return real_read(vrt_path, window)

    monkeypatch.setattr(tpi_module, "_read_window", flaky_read)
    radius = 4.0
    paths = create_tpi(layout, AREA, radius, gpu=True, dtype=np.float64, cores=1)
    assert len(paths) == 4
    nodata_tiles = [p.name for p in paths if np.all(read_band(p) == TPI_NODATA)]
    assert len(nodata_tiles) == 1


def test_create_tpi_skips_bad_index_row(layout, dtm_product: Path) -> None:
    """An index row with an unusable footprint is skipped; the other tiles
    are processed and the TPI index excludes the skipped one."""
    from lidark.io.tiles import tile_index_path

    dtm_dir = layout.dtm_dir(AREA)
    index = load_tile_index(dtm_dir)
    index.loc[index.index[0], "geometry"] = None
    index.to_file(tile_index_path(dtm_dir), layer="tiles")

    radius = 4.0
    paths = create_tpi(layout, AREA, radius, dtype=np.float64, cores=1)
    assert len(paths) == 3
    assert len(load_tile_index(layout.tpi_dir(AREA, radius))) == 3


def test_create_tpi_broken_pool_falls_back_to_serial(
    layout, dtm_product: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker pool that breaks (native crash) re-runs the unfinished tiles
    serially; the run completes with all tiles."""
    from concurrent.futures.process import BrokenProcessPool

    import lidark.maps.tpi as tpi_module

    class FakePool:
        def __init__(self, max_workers: int) -> None:
            pass

        def __enter__(self) -> "FakePool":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def map(self, fn, tasks):
            raise BrokenProcessPool("simulated native crash")

    monkeypatch.setattr(tpi_module, "ProcessPoolExecutor", FakePool)
    radius = 4.0
    paths = create_tpi(layout, AREA, radius, dtype=np.float64, cores=2)
    assert len(paths) == 4
    for p in paths:
        assert (read_band(p) != TPI_NODATA).any()


def test_create_tpi_rerun_skips_processed_tiles(
    layout, dtm_product: Path, lidark_log_records: list[str]
) -> None:
    """Re-running skips every existing tile, logs one line per skipped tile,
    and leaves the product (files, VRT, index) complete and untouched."""
    radius = 4.0
    first = create_tpi(layout, AREA, radius, dtype=np.float64, cores=1)
    assert len(first) == 4
    mtimes = {p.name: p.stat().st_mtime_ns for p in first}

    lidark_log_records.clear()
    second = create_tpi(layout, AREA, radius, dtype=np.float64, cores=1)
    assert [p.name for p in second] == [p.name for p in first]
    assert {p.name: p.stat().st_mtime_ns for p in second} == mtimes
    skipped = [
        m for m in lidark_log_records if m.startswith("skipping ") and "already processed" in m
    ]
    assert len(skipped) == 4
    tpi_dir = layout.tpi_dir(AREA, radius)
    assert (tpi_dir / "tpi.vrt").exists()
    assert len(load_tile_index(tpi_dir)) == 4


def test_create_tpi_resume_recomputes_only_missing_or_invalid(
    layout, dtm_product: Path, lidark_log_records: list[str]
) -> None:
    """Resume after an interruption: only missing or unreadable TPI tiles are
    recomputed; valid ones are skipped (and logged)."""
    radius = 4.0
    first = create_tpi(layout, AREA, radius, dtype=np.float64, cores=1)
    tpi_dir = layout.tpi_dir(AREA, radius)
    mtimes = {p.name: p.stat().st_mtime_ns for p in first}
    names = sorted(p.name for p in first)
    (tpi_dir / names[0]).unlink()  # missing -> recompute
    (tpi_dir / names[1]).write_bytes(b"corrupt")  # unreadable -> recompute

    lidark_log_records.clear()
    second = create_tpi(layout, AREA, radius, dtype=np.float64, cores=1)
    assert len(second) == 4
    # The two untouched tiles were skipped (mtime unchanged, one log line each).
    for name in names[2:]:
        assert (tpi_dir / name).stat().st_mtime_ns == mtimes[name]
    skipped = [
        m for m in lidark_log_records if m.startswith("skipping ") and "already processed" in m
    ]
    assert len(skipped) == 2
    # The missing/corrupt tiles were rebuilt, readable, right shape.
    for name in names[:2]:
        assert (tpi_dir / name).stat().st_mtime_ns != mtimes[name]
        assert read_band(tpi_dir / name).shape == TILE_SHAPE
    assert len(load_tile_index(tpi_dir)) == 4


def _expected_tpi(dtm_dir: Path, bounds: tuple, radius: float) -> np.ndarray:
    """Reference TPI (original kernel, float64) for one tile footprint."""
    import rasterio

    with rasterio.open(dtm_dir / "dtm.vrt") as mosaic:
        transform = mosaic.transform
        window = rasterio.windows.from_bounds(
            bounds[0] - radius,
            bounds[1] - radius,
            bounds[2] + radius,
            bounds[3] + radius,
            transform=transform,
        )
        col0 = int(np.floor(window.col_off))
        row0 = int(np.floor(window.row_off))
        col1 = int(np.ceil(window.col_off + window.width))
        row1 = int(np.ceil(window.row_off + window.height))
        arr = mosaic.read(
            1,
            window=rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0),
            boundless=True,
        )
    out = tpi_reference(arr, radius_px=int(round(radius / PIXEL_SIZE)), dtype=np.float64)
    post = tpi_postprocess(out)
    trimmed = post[
        int(radius / PIXEL_SIZE) : -int(radius / PIXEL_SIZE),
        int(radius / PIXEL_SIZE) : -int(radius / PIXEL_SIZE),
    ]
    return trimmed
