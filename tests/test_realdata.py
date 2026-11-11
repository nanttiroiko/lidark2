"""Opt-in tests against the real LiDAR sample in ``test_data/laz``.

Synthetic data remains the default for the test suite: it is deterministic,
fast, and proves the same logic. These tests cover the cases where it does
not - genuine point density, classification noise, LAZ decompression cost, and
DTM/TPI quality on real terrain (plan section 6: DTM acceptance is
coverage/interpolation quality, verified statistically, not bit parity).

All tests are marked ``@pytest.mark.realdata`` and skip automatically when
``test_data/laz/`` is empty or missing; CI runs the suite with
``-m "not realdata"`` so it never depends on the sample.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lidark.config import DEFAULT_EPSG, ProjectLayout
from lidark.logging import get_logger

pytestmark = [pytest.mark.realdata]

pdal = pytest.importorskip("pdal", reason="PDAL not installed")

LOGGER = get_logger("realdata")

PIXEL = 2.0  # coarse resolution keeps the real-data runtime bounded
BUFFER = 10.0
AREA = "RealSample"


def test_real_samples_are_readable(real_laz_files: list[Path]) -> None:
    """Every sample file must expose bounds through the PDAL header read."""
    from lidark.maps.dtm import las_bounds

    for path in real_laz_files:
        bounds = las_bounds(path)
        assert bounds[0] < bounds[2] and bounds[1] < bounds[3], path


def test_real_sample_point_density(real_laz_files: list[Path]) -> None:
    """Real point density: full decompression of one sample, timed.

    This is the LAZ-decompression profile the tile-index neighbour reads
    target (7, P1): each output tile opens its intersecting neighbours.
    """
    import json

    path = real_laz_files[0]
    start = time.perf_counter()
    pipeline = pdal.Pipeline(
        json.dumps({"pipeline": [{"type": "readers.las", "filename": str(path)}]})
    )
    count = int(pipeline.execute())
    duration = time.perf_counter() - start
    from lidark.maps.dtm import las_bounds

    bounds = las_bounds(path)
    area_m2 = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    density = count / max(area_m2, 1.0)
    LOGGER.info(
        "real sample %s: %d points, %.1f pts/m2, decompressed in %.1fs (%.1f Mpts/s)",
        path.name,
        count,
        density,
        duration,
        count / max(duration, 1e-9) / 1e6,
    )
    assert count > 0
    assert density > 0.0


@pytest.fixture(scope="module")
def real_project(real_laz_files: list[Path], tmp_path_factory) -> SimpleNamespace:
    """A project area built from one real sample file, DTM included.

    Module-scoped: the ~68 MB LAZ is decompressed once for the whole test
    module.
    """
    from lidark.maps.dtm import generate_dtm

    layout = ProjectLayout(project_folder=tmp_path_factory.mktemp("realdata"), epsg=DEFAULT_EPSG)
    originals = layout.originals_dir(AREA)
    originals.mkdir(parents=True)
    shutil.copy(real_laz_files[0], originals / real_laz_files[0].name)

    start = time.perf_counter()
    tiles = generate_dtm(layout, AREA, pixel_size=PIXEL, buffer=BUFFER, cores=1)
    duration = time.perf_counter() - start
    LOGGER.info(
        "real DTM: %d tiles in %.1fs (%.2f tiles/s)",
        len(tiles),
        duration,
        len(tiles) / duration,
    )
    return SimpleNamespace(layout=layout, tiles=tiles)


def test_real_dtm_coverage_and_range(real_project: SimpleNamespace) -> None:
    """Quality-level DTM acceptance (plan section 6): coverage and plausible
    elevations on real point density and classification noise."""
    import rasterio

    from lidark.maps.dtm import DTM_NODATA

    path = real_project.tiles[0]
    with rasterio.open(path) as src:
        assert src.res == (PIXEL, PIXEL)
        assert src.dtypes[0] == "float32"
        assert src.nodata == DTM_NODATA
        arr = src.read(1)
    covered = arr != DTM_NODATA
    coverage = covered.mean()
    elevations = arr[covered]
    LOGGER.info(
        "real DTM %s: shape %s, coverage %.1f%%, elevation %.1f..%.1f m",
        path.name,
        arr.shape,
        100.0 * coverage,
        float(elevations.min()),
        float(elevations.max()),
    )
    assert coverage > 0.1, "DTM is nearly empty"
    # Plausible ground elevations for Finnish terrain (sanity, not quality).
    assert elevations.min() > -100.0
    assert elevations.max() < 1000.0


def test_real_tpi_from_real_dtm(real_project: SimpleNamespace) -> None:
    """TPI over real terrain: range, dtype, and product artifacts."""
    import rasterio

    from lidark.io.tiles import load_tile_index
    from lidark.maps.tpi import TPI_NODATA, create_tpi

    layout = real_project.layout
    radius = 4.0  # metres; 2 px at the coarse resolution
    start = time.perf_counter()
    tpi_tiles = create_tpi(layout, AREA, radius, dtype=np.float64)
    duration = time.perf_counter() - start
    LOGGER.info(
        "real TPI: %d tiles in %.1fs (%.2f tiles/s)",
        len(tpi_tiles),
        duration,
        len(tpi_tiles) / duration,
    )

    tpi_dir = layout.tpi_dir(AREA, radius)
    assert (tpi_dir / "tpi.vrt").exists()
    assert (tpi_dir / "tiles.gpkg").exists()
    index = load_tile_index(tpi_dir)
    assert len(index) == len(real_project.tiles)

    signal = 0
    for path in tpi_tiles:
        with rasterio.open(path) as src:
            assert src.dtypes[0] == "int16"
            assert src.nodata == TPI_NODATA
            arr = src.read(1)
        valid = arr[arr != TPI_NODATA]
        assert int(valid.min()) >= -100 and int(valid.max()) <= 100
        signal += int(np.count_nonzero(valid))
    assert signal > 0, "TPI is all zeros - no relief signal in the sample"


def test_real_buffer_reads_neighbouring_points(
    real_laz_files: list[Path], real_project: SimpleNamespace, tmp_path_factory
) -> None:
    """Determinism and neighbour reads on real data: the same input produces
    the same TIN tiles, and no intermediate point cloud is written next to
    the originals."""
    import rasterio

    from lidark.maps.dtm import generate_dtm

    layout = ProjectLayout(
        project_folder=tmp_path_factory.mktemp("realdata_buffer"), epsg=DEFAULT_EPSG
    )
    originals = layout.originals_dir(AREA)
    originals.mkdir(parents=True)
    shutil.copy(real_laz_files[0], originals / real_laz_files[0].name)
    rerun = generate_dtm(layout, AREA, pixel_size=PIXEL, buffer=BUFFER, cores=1)
    for first, second in zip(real_project.tiles, rerun, strict=True):
        with rasterio.open(first) as a, rasterio.open(second) as b:
            assert a.transform == b.transform
            np.testing.assert_array_equal(a.read(1), b.read(1))
    assert list(originals.iterdir()) == [originals / real_laz_files[0].name]
