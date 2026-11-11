"""End-to-end smoke test (plan Phase 7): the complete workflow on one
synthetic project.

Synthetic LAS -> DTM (PDAL, tile-index neighbour reads) -> TPI -> training
dataset -> 1-epoch train -> predict -> evaluate, asserting the folder/file
layout at every step.
Stage durations are logged (tiles/s and img/s) so performance regressions are
visible in test output; no hard threshold is enforced (plan section 7).
"""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box
from tests.lasfile import write_las

from lidark.config import DEFAULT_EPSG, ModelConfig, ProjectLayout
from lidark.logging import get_logger
from lidark.maps.tpi import TPI_NODATA

pdal = pytest.importorskip("pdal", reason="PDAL not installed")
pytest.importorskip("torch", reason="torch not installed")

LOGGER = get_logger("smoke")

AREA = "SmokeArea"
DTM_BUFFER = 5.0
PIXEL = 1.0
TPI_RADIUS = 2.0


def _write_synthetic_originals(layout: ProjectLayout) -> None:
    """Two adjacent 40x40 m LAS tiles: ground with gentle relief + noise."""
    rng = np.random.default_rng(21)
    for i in range(2):
        x0 = 1000.0 + 40.0 * i
        xs = rng.uniform(x0, x0 + 40.0, size=3000)
        ys = rng.uniform(2000.0, 2040.0, size=3000)
        zs = 10.0 + 3.0 * np.sin(xs / 8.0) + 2.0 * np.cos(ys / 11.0)
        cls = np.full(3000, 2, dtype=np.uint8)
        # Non-ground noise far above the terrain (must not reach the DTM).
        xs = np.concatenate([xs, rng.uniform(x0 + 10.0, x0 + 30.0, size=200)])
        ys = np.concatenate([ys, rng.uniform(2010.0, 2030.0, size=200)])
        zs = np.concatenate([zs, np.full(200, 1000.0)])
        cls = np.concatenate([cls, np.full(200, 1, dtype=np.uint8)])
        write_las(layout.originals_dir(AREA) / f"orig_{i}.las", xs, ys, zs, cls)


def _timed(label: str, fn, *args, **kwargs):
    """Run a stage, log its duration (plan: tiles/s and img/s visibility)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    duration = time.perf_counter() - start
    LOGGER.info("smoke stage %s finished in %.2fs", label, duration)
    return result, duration


@pytest.fixture()
def layout(tmp_path: Path) -> ProjectLayout:
    project = ProjectLayout(project_folder=tmp_path, epsg=DEFAULT_EPSG)
    project.originals_dir(AREA).mkdir(parents=True)
    return project


def test_end_to_end_smoke(layout: ProjectLayout) -> None:
    from lidark import create_map, evaluate, predict, train
    from lidark.io.tiles import load_tile_index

    _write_synthetic_originals(layout)

    # ---------------------------------------------------------- 1. DTM (PDAL)
    (dtm_tiles, dtm_seconds) = _timed(
        "DTM",
        create_map,
        layout.project_folder,
        AREA,
        "DTM",
        buffer=DTM_BUFFER,
        pixel_size=PIXEL,
        cores=1,
    )
    dtm_dir = layout.dtm_dir(AREA)
    assert len(dtm_tiles) == 2
    assert (dtm_dir / "dtm.vrt").exists()
    assert (dtm_dir / "tiles.gpkg").exists()
    dtm_index = load_tile_index(dtm_dir)
    assert dtm_index["tile_id"].tolist() == ["orig_0", "orig_1"]
    LOGGER.info(
        "DTM throughput: %.2f tiles/s (%d tiles, %.1fs)",
        len(dtm_tiles) / dtm_seconds,
        len(dtm_tiles),
        dtm_seconds,
    )

    import rasterio

    with rasterio.open(dtm_tiles[0]) as src:
        dtm = src.read(1)
        assert src.res == (PIXEL, PIXEL)
        assert src.dtypes[0] == "float32"
    # Ground-only: the z=1000 noise cluster must not leak into the terrain.
    assert dtm.max() < 100.0
    assert (dtm != 0).sum() > 0

    # ------------------------------------------------------------- 2. TPI
    (tpi_tiles, tpi_seconds) = _timed(
        "TPI",
        create_map,
        layout.project_folder,
        AREA,
        "TPI",
        buffer=TPI_RADIUS,
        cores=1,
    )
    tpi_dir = layout.tpi_dir(AREA, TPI_RADIUS)
    assert len(tpi_tiles) == 2
    assert (tpi_dir / "tpi.vrt").exists()
    assert (tpi_dir / "tiles.gpkg").exists()
    tpi_index = load_tile_index(tpi_dir)
    assert tpi_index["tile_id"].tolist() == ["orig_0", "orig_1"]
    LOGGER.info(
        "TPI throughput: %.2f tiles/s (%d tiles, %.1fs)",
        len(tpi_tiles) / tpi_seconds,
        len(tpi_tiles),
        tpi_seconds,
    )
    with rasterio.open(tpi_tiles[0]) as src:
        tpi_arr = src.read(1)
        assert src.dtypes[0] == "int16"
        assert src.nodata == TPI_NODATA
    # Nodata cells (DTM gaps) carry the nodata constant; valid cells stay in
    # the original TPI range.
    tpi_valid = tpi_arr[tpi_arr != TPI_NODATA]
    assert tpi_valid.size > 0
    assert int(tpi_valid.min()) >= -100 and int(tpi_valid.max()) <= 100

    # -------------------------------------------- 3. training dataset + model
    from tests.conftest import write_annotation

    true_geometry = [
        box(1010.0, 2010.0, 1018.0, 2018.0),
        box(1050.0, 2015.0, 1058.0, 2023.0),
    ]
    write_annotation(layout, "kilns.shp", true_geometry, class_ids=[1, 1])
    write_annotation(layout, "noise.shp", [box(1025.0, 2030.0, 1033.0, 2038.0)])

    features = ["01_DTM", f"02_TPI/radius_{TPI_RADIUS:.0f}"]
    config = ModelConfig(
        feature_list=tuple(features),
        img_size=(16, 16),
        unet_size=4,
        batch_size=4,
        max_epochs=1,
        seed=0,
        pred_stride=8,
        pred_batch_size=8,
    )
    (timestamp, train_seconds) = _timed(
        "train",
        train,
        layout,
        AREA,
        features,
        "kilns.shp",
        "noise.shp",
        model_name="smoke",
        model_config=config,
    )

    model_dir = layout.model_dir(timestamp)
    assert (model_dir / "model.pt").exists()
    assert (model_dir / "model_config.json").exists()
    assert (layout.model_dir(timestamp) / "training_log.json").exists()

    dataset_dirs = list(layout.samples_dir.iterdir())
    assert len(dataset_dirs) == 1
    dataset_dir = dataset_dirs[0]
    assert (dataset_dir / "samples.dat").exists()
    assert (dataset_dir / "labels.dat").exists()
    assert (dataset_dir / "dataset_meta.json").exists()

    import json

    meta = json.loads((dataset_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    n_samples = meta["samples_shape"][0]
    LOGGER.info(
        "train throughput: %.1f img/s (%d samples, %.1fs)",
        n_samples / train_seconds,
        n_samples,
        train_seconds,
    )

    # ---------------------------------------------------------- 4. predict
    (_results, predict_seconds) = _timed(
        "predict",
        predict,
        layout,
        [AREA],
        timestamp,
        threshold=0.0,
    )
    merged_path = layout.predictions_dir(timestamp, AREA) / "merged_shapes_all.gpkg"
    assert merged_path.exists()
    rasters = list(layout.prediction_raster_dir(timestamp, AREA).glob("*.tif"))
    assert len(rasters) == 2
    merged = gpd.read_file(merged_path)
    for column in ("score", "class_id", "class_name", "area"):
        assert column in merged.columns
    LOGGER.info(
        "predict throughput: %.2f tiles/s (%d tiles, %.1fs)",
        len(rasters) / predict_seconds,
        len(rasters),
        predict_seconds,
    )

    # --------------------------------------------------------- 5. evaluate
    eval_dir = layout.evaluate_annotations_dir("smoke_eval")
    eval_dir.mkdir(parents=True)
    gpd.GeoDataFrame(geometry=[box(1010.0, 2010.0, 1018.0, 2018.0)], crs=DEFAULT_EPSG).to_file(
        eval_dir / "evaluate_true.shp"
    )

    from lidark.evaluate import COUNTERS

    report = evaluate(
        layout,
        "evaluate_true.shp",
        timestamp,
        "smoke_eval",
        prob_cut=0.0,
        size_cut=0.0,
    )
    report_path = layout.reports_dir(timestamp) / "smoke_eval.csv"
    assert report_path.exists()
    assert set(COUNTERS) <= set(report.columns)
    overall = report.loc[report["class_id"] == "overall"].iloc[0]
    # "Predicted" counts the predictions clipped to the evaluation AOI
    # (label bbox), not the raw merged set.
    aoi = box(1010.0, 2010.0, 1018.0, 2018.0)
    assert overall["Predicted"] == int(merged.intersects(aoi).sum())
    assert (
        overall["Fully detected"] + overall["Partly detected"] + overall["Not detected"] == 1
    )  # exactly one evaluation label
