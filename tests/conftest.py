"""Shared pytest fixtures: synthetic project tree with tiny rasters.

The synthetic tiles use arbitrary (non-mapsheet) filenames of varying lengths
to prove nothing in the pipeline depends on filename content or length.

Real LiDAR samples may live in ``test_data/laz/``; they are opt-in only
(``@pytest.mark.realdata`` + the ``real_laz_files`` fixture, which skips when
the folder is empty or missing). Synthetic fixtures remain the default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from lidark.config import DEFAULT_EPSG, ProjectLayout
from lidark.io.rasters import write_band
from lidark.io.tiles import LAYER_NAME, build_tile_index, load_tile_index, tile_index_path
from lidark.io.vrt import build_vrt
from lidark.logging import configure_logging

if TYPE_CHECKING:
    from collections.abc import Iterator

AREA = "Naljanka"
# A second project area (see below, after the origins, for its offset).
AREA2 = "Kuivaniemi"
PIXEL_SIZE = 2.0
TILE_SHAPE = (32, 32)
# Arbitrary filenames: different lengths, no mapsheet convention.
TILE_NAMES = (
    "chunk_a.tif",
    "survey2024_ne_part-0002.tif",
    "Q.tif",
    "tile-number-00000004.tif",
)
ORIGIN_X = 380000.0
ORIGIN_Y = 6678000.0
# The second area's grid origin: offset far enough that its tiles never
# overlap the first area's grid (the grids are 2 x 64 m).
AREA2_ORIGIN_X = ORIGIN_X + 2000.0
AREA2_ORIGIN_Y = ORIGIN_Y
# Repository-rooted path to the optional real-data samples.
REAL_LAZ_DIR = Path(__file__).resolve().parent.parent / "test_data" / "laz"


@pytest.fixture(autouse=True)
def _quiet_logging() -> None:
    """Give every test a configured logger without verbose output."""
    configure_logging(verbose=False)


@pytest.fixture()
def lidark_log_records() -> Iterator[list[str]]:
    """Capture ``lidark`` log messages as plain strings.

    The ``lidark`` logger sets ``propagate = False``, so pytest's ``caplog``
    (root-attached handler) never sees these records; this handler attaches
    directly to the ``lidark`` logger instead.
    """
    records: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _ListHandler(level=logging.DEBUG)
    logger = logging.getLogger("lidark")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


@pytest.fixture(scope="session")
def real_laz_files() -> list[Path]:
    """Real LiDAR samples (opt-in; synthetic data is the default).

    Skips the test when ``test_data/laz/`` is empty or missing, so the suite
    never depends on the sample being present. Session-scoped: the file list
    is immutable, and module-scoped project fixtures may build on it.
    """
    files = (
        sorted(p for p in REAL_LAZ_DIR.glob("*.la[sz]") if p.is_file())
        if REAL_LAZ_DIR.exists()
        else []
    )
    if not files:
        pytest.skip(f"no real LiDAR samples in {REAL_LAZ_DIR}")
    return files


@pytest.fixture()
def layout(tmp_path: Path) -> ProjectLayout:
    """Project layout rooted at tmp_path."""
    return ProjectLayout(project_folder=tmp_path, epsg=DEFAULT_EPSG)


@pytest.fixture()
def dtm_dir(layout: ProjectLayout) -> Path:
    """Empty product directory for DTM-like tiles."""
    path = layout.dtm_dir(AREA)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def dtm_tiles(dtm_dir: Path) -> list[Path]:
    """A 2x2 grid of 32x32 float32 tiles (2 m pixels, EPSG:3067).

    Deterministic elevation pattern: value increases linearly along the row
    index, plus a per-tile offset.
    """
    height, width = TILE_SHAPE
    paths: list[Path] = []
    for i, name in enumerate(TILE_NAMES):
        row, col = divmod(i, 2)
        x0 = ORIGIN_X + col * width * PIXEL_SIZE
        y_top = ORIGIN_Y - row * height * PIXEL_SIZE
        transform = from_origin(x0, y_top, PIXEL_SIZE, PIXEL_SIZE)
        arr = (10.0 + 5.0 * row + 3.0 * col + np.indices((height, width))[0]).astype(np.float32)
        path = dtm_dir / name
        write_band(path, arr, transform=transform, crs=DEFAULT_EPSG)
        paths.append(path)
    return paths


@pytest.fixture()
def dtm_product(dtm_tiles: list[Path], dtm_dir: Path) -> Path:
    """A complete DTM product: tiles + dtm.vrt + tiles.gpkg."""
    build_vrt(dtm_tiles, dtm_dir / "dtm.vrt")
    build_tile_index(dtm_dir, default_epsg=DEFAULT_EPSG)
    return dtm_dir


@pytest.fixture()
def label_polygons() -> list[box]:
    """Two polygons inside the synthetic tile grid."""
    return [
        box(ORIGIN_X + 10.0, ORIGIN_Y - 30.0, ORIGIN_X + 40.0, ORIGIN_Y - 10.0),
        box(ORIGIN_X + 50.0, ORIGIN_Y - 60.0, ORIGIN_X + 70.0, ORIGIN_Y - 45.0),
    ]


@pytest.fixture()
def labels(label_polygons: list[box]) -> Iterator[gpd.GeoDataFrame]:
    """Label frame over the synthetic tiles."""
    yield gpd.GeoDataFrame({"label": [1, 1]}, geometry=label_polygons, crs=DEFAULT_EPSG)


def tile_transform(row: int, col: int):
    """Transform of the synthetic tile at grid position (row, col)."""
    height, width = TILE_SHAPE
    return from_origin(
        ORIGIN_X + col * width * PIXEL_SIZE,
        ORIGIN_Y - row * height * PIXEL_SIZE,
        PIXEL_SIZE,
        PIXEL_SIZE,
    )


def tile_array(row: int, col: int) -> np.ndarray:
    """Expected array content of the synthetic tile at (row, col)."""
    height, width = TILE_SHAPE
    return (10.0 + 5.0 * row + 3.0 * col + np.indices((height, width))[0]).astype(np.float32)


def write_annotation(
    layout: ProjectLayout,
    name: str,
    geometry: list,
    class_ids: list[int] | None = None,
) -> Path:
    """Write a label shapefile into DeepLearning/train_annotations/."""
    frame = gpd.GeoDataFrame(geometry=geometry, crs=DEFAULT_EPSG)
    if class_ids is not None:
        frame["class_id"] = class_ids
    out_dir = layout.train_annotations_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    frame.to_file(out)
    return out


@pytest.fixture()
def tpi_layer(layout: ProjectLayout, dtm_product: Path) -> Path:
    """A second feature layer (``02_TPI/radius_20``) aligned with the DTM.

    int16 tiles with the same tile IDs and footprints (values in the original
    TPI range), plus the product VRT and tile index.
    """
    feature_dir = layout.tpi_dir(AREA, 20)
    feature_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths_by_id: dict[str, Path] = {}
    for i, name in enumerate(TILE_NAMES):
        row, col = divmod(i, 2)
        transform = tile_transform(row, col)
        arr = (np.indices(TILE_SHAPE)[0] * 3 - 100 + 30 * row + 60 * col).astype(np.int16)
        out = feature_dir / f"{Path(name).stem}_TPI.tif"
        write_band(out, arr, transform=transform, crs=DEFAULT_EPSG, nodata=None)
        paths.append(out)
        paths_by_id[Path(name).stem] = out
    build_vrt(paths, feature_dir / "tpi.vrt")
    index = load_tile_index(dtm_product)
    index = index.copy()
    index["path"] = [str(paths_by_id[tile_id]) for tile_id in index["tile_id"]]
    index.to_file(tile_index_path(feature_dir), layer=LAYER_NAME)
    return feature_dir


@pytest.fixture()
def dtm_product2(layout: ProjectLayout) -> Path:
    """A complete DTM product for the second project area (AREA2).

    Same 2x2 tile grid and values as ``dtm_product``, offset to the second
    area's origin so the two areas never overlap spatially.
    """
    path = layout.dtm_dir(AREA2)
    path.mkdir(parents=True, exist_ok=True)
    height, width = TILE_SHAPE
    paths: list[Path] = []
    for i, name in enumerate(TILE_NAMES):
        row, col = divmod(i, 2)
        x0 = AREA2_ORIGIN_X + col * width * PIXEL_SIZE
        y_top = AREA2_ORIGIN_Y - row * height * PIXEL_SIZE
        transform = from_origin(x0, y_top, PIXEL_SIZE, PIXEL_SIZE)
        arr = (10.0 + 5.0 * row + 3.0 * col + np.indices((height, width))[0]).astype(np.float32)
        out = path / name
        write_band(out, arr, transform=transform, crs=DEFAULT_EPSG)
        paths.append(out)
    build_vrt(paths, path / "dtm.vrt")
    build_tile_index(path, default_epsg=DEFAULT_EPSG)
    return path


@pytest.fixture()
def tpi_layer2(layout: ProjectLayout, dtm_product2: Path) -> Path:
    """The second area's ``02_TPI/radius_20`` feature layer (aligned)."""
    feature_dir = layout.tpi_dir(AREA2, 20)
    feature_dir.mkdir(parents=True, exist_ok=True)
    height, width = TILE_SHAPE
    paths: list[Path] = []
    paths_by_id: dict[str, Path] = {}
    for i, name in enumerate(TILE_NAMES):
        row, col = divmod(i, 2)
        x0 = AREA2_ORIGIN_X + col * width * PIXEL_SIZE
        y_top = AREA2_ORIGIN_Y - row * height * PIXEL_SIZE
        transform = from_origin(x0, y_top, PIXEL_SIZE, PIXEL_SIZE)
        arr = (np.indices((height, width))[0] * 3 - 100 + 30 * row + 60 * col).astype(np.int16)
        out = feature_dir / f"{Path(name).stem}_TPI.tif"
        write_band(out, arr, transform=transform, crs=DEFAULT_EPSG, nodata=None)
        paths.append(out)
        paths_by_id[Path(name).stem] = out
    build_vrt(paths, feature_dir / "tpi.vrt")
    index = load_tile_index(dtm_product2)
    index = index.copy()
    index["path"] = [str(paths_by_id[tile_id]) for tile_id in index["tile_id"]]
    index.to_file(tile_index_path(feature_dir), layer=LAYER_NAME)
    return feature_dir
