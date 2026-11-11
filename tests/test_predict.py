"""Tests for lidark.predict: e2e inference, batching, pred_type, failures."""

from __future__ import annotations

import importlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box
from tests.conftest import AREA, ORIGIN_X, ORIGIN_Y, TILE_SHAPE, tile_transform, write_annotation

from lidark.config import ModelConfig
from lidark.io.detections import COLUMNS, validate_detections
from lidark.predict import predict
from lidark.train import train

FEATURES = ["01_DTM", "02_TPI/radius_20"]


@pytest.fixture(scope="module")
def annotations(layout_module) -> None:
    """Labels in every tile so all four tiles contribute training windows."""
    w = TILE_SHAPE[1] * 2.0
    h = TILE_SHAPE[0] * 2.0
    geometry = []
    for row in range(2):
        for col in range(2):
            x0 = ORIGIN_X + col * w + 20.0
            y_top = ORIGIN_Y - row * h - 20.0
            geometry.append(box(x0, y_top - 8.0, x0 + 8.0, y_top))
    write_annotation(layout_module, "true.shp", geometry, class_ids=[1] * 4)
    write_annotation(
        layout_module,
        "false.shp",
        [box(ORIGIN_X + 4, ORIGIN_Y - 60, ORIGIN_X + 12, ORIGIN_Y - 52)],
    )


@pytest.fixture(scope="module")
def layout_module(tmp_path_factory):
    """Module-scoped layout + products shared by the predict tests."""
    from lidark.config import DEFAULT_EPSG, ProjectLayout

    layout = ProjectLayout(
        project_folder=tmp_path_factory.mktemp("predict_project"), epsg=DEFAULT_EPSG
    )
    # Rebuild the DTM product and TPI layer once for this module.

    from tests.conftest import TILE_NAMES, tile_array, tile_transform

    from lidark.io.rasters import write_band
    from lidark.io.tiles import build_tile_index, load_tile_index, tile_index_path
    from lidark.io.vrt import build_vrt

    dtm_dir = layout.dtm_dir(AREA)
    dtm_dir.mkdir(parents=True)
    paths = []
    for i, name in enumerate(TILE_NAMES):
        row, col = divmod(i, 2)
        out = dtm_dir / name
        write_band(out, tile_array(row, col), transform=tile_transform(row, col), crs=layout.epsg)
        paths.append(out)
    build_vrt(paths, dtm_dir / "dtm.vrt")
    build_tile_index(dtm_dir, default_epsg=layout.epsg)

    feature_dir = layout.tpi_dir(AREA, 20)
    feature_dir.mkdir(parents=True)
    fpaths: list[Path] = []
    paths_by_id: dict[str, Path] = {}
    for i, name in enumerate(TILE_NAMES):
        row, col = divmod(i, 2)
        arr = (np.indices(TILE_SHAPE)[0] * 3 - 100 + 30 * row + 60 * col).astype(np.int16)
        out = feature_dir / f"{Path(name).stem}_TPI.tif"
        write_band(out, arr, transform=tile_transform(row, col), crs=layout.epsg, nodata=None)
        fpaths.append(out)
        paths_by_id[Path(name).stem] = out
    build_vrt(fpaths, feature_dir / "tpi.vrt")
    index = load_tile_index(dtm_dir)
    index = index.copy()
    index["path"] = [str(paths_by_id[tile_id]) for tile_id in index["tile_id"]]
    index.to_file(tile_index_path(feature_dir), layer="tiles")
    return layout


def _tiny_config(**overrides: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "feature_list": tuple(FEATURES),
        "img_size": (16, 16),
        "unet_size": 4,
        "batch_size": 4,
        "max_epochs": 1,
        "seed": 0,
        "pred_stride": 8,
        "pred_batch_size": 8,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def model_timestamp(layout_module, annotations) -> str:
    """A trained (1 epoch, tiny) model shared by the predict tests."""
    return train(
        layout_module,
        AREA,
        FEATURES,
        "true.shp",
        "false.shp",
        model_name="predict",
        model_config=_tiny_config(),
    )


def test_predict_e2e_produces_outputs(layout_module, model_timestamp: str) -> None:
    # threshold=0.0 guarantees detections from the untrained-but-valid model.
    result = predict(layout_module, [AREA], model_timestamp, threshold=0.0)
    merged = result[AREA]

    # Detection IR schema.
    validate_detections(merged)
    assert set(COLUMNS) <= set(merged.columns)
    assert merged["class_id"].isin([1]).all()
    assert (merged["score"] >= 0).all() and (merged["score"] <= 1).all()
    assert (merged["area"] > 0).all()

    # Per-tile rasters (uint8 masks, original naming by tile id).
    raster_dir = layout_module.prediction_raster_dir(model_timestamp, AREA)
    rasters = sorted(raster_dir.glob("*.tif"))
    assert len(rasters) == 4
    for path in rasters:
        with rasterio.open(path) as src:
            assert src.dtypes[0] == "uint8"
            assert src.read(1).shape == TILE_SHAPE

    # Merged GeoPackage written next to the predictions.
    merged_path = layout_module.predictions_dir(model_timestamp, AREA) / "merged_shapes_all.gpkg"
    assert merged_path.exists()
    on_disk = gpd.read_file(merged_path)
    assert len(on_disk) == len(merged)


def test_predict_default_threshold_still_writes_rasters(
    layout_module, model_timestamp: str
) -> None:
    result = predict(layout_module, [AREA], model_timestamp)
    raster_dir = layout_module.prediction_raster_dir(model_timestamp, AREA)
    assert len(list(raster_dir.glob("*.tif"))) == 4
    # The IR is valid whether or not anything passed the threshold.
    validate_detections(result[AREA])


def _clear_rasters(layout, model_timestamp: str) -> None:
    """Remove per-tile prediction rasters so tiles re-predict (fresh runs)."""
    for p in layout.prediction_raster_dir(model_timestamp, AREA).glob("*.tif"):
        p.unlink()


def test_batched_equals_single_window(layout_module, model_timestamp: str) -> None:
    _clear_rasters(layout_module, model_timestamp)
    big = predict(layout_module, [AREA], model_timestamp, threshold=0.0, pred_batch_size=64)
    rasters = {
        p.name: rasterio.open(p).read(1)
        for p in layout_module.prediction_raster_dir(model_timestamp, AREA).glob("*.tif")
    }
    assert len(rasters) == 4
    scores_big = big[AREA]["score"].to_numpy().copy()

    # Rerun with batch size 1 over the same (emptied) outputs.
    _clear_rasters(layout_module, model_timestamp)
    single = predict(layout_module, [AREA], model_timestamp, threshold=0.0, pred_batch_size=1)
    for name, arr in rasters.items():
        np.testing.assert_array_equal(
            rasterio.open(layout_module.prediction_raster_dir(model_timestamp, AREA) / name).read(
                1
            ),
            arr,
        )
    np.testing.assert_allclose(
        np.sort(single[AREA]["score"].to_numpy()),
        np.sort(scores_big),
        atol=1e-5,
    )


def test_predict_pred_type_test(layout_module, model_timestamp: str) -> None:
    _clear_rasters(layout_module, model_timestamp)
    result = predict(
        layout_module,
        [AREA],
        model_timestamp,
        pred_type="test",
        test_tiles=["Q"],
        threshold=0.0,
    )
    raster_dir = layout_module.prediction_raster_dir(model_timestamp, AREA)
    # Only tiles matching the test pattern were predicted.
    assert [p.stem for p in sorted(raster_dir.glob("*.tif"))] == ["Q"]
    merged_path = layout_module.predictions_dir(model_timestamp, AREA) / "merged_shapes_test.gpkg"
    assert merged_path.exists()
    validate_detections(result[AREA])


def test_predict_pred_type_test_area(layout_module, model_timestamp: str) -> None:
    """pred_type='test' with a test-area polygon layer (the same convention
    as training) selects exactly the tiles intersecting the polygons."""
    w = TILE_SHAPE[1] * 2.0
    h = TILE_SHAPE[0] * 2.0
    # A polygon strictly inside tile Q (grid position (1, 0): y in
    # [ORIGIN_Y - 2h, ORIGIN_Y - h]).
    write_annotation(
        layout_module,
        "test_area.gpkg",
        [box(ORIGIN_X + 8.0, ORIGIN_Y - 2 * h + 8.0, ORIGIN_X + w - 8.0, ORIGIN_Y - h - 8.0)],
    )
    _clear_rasters(layout_module, model_timestamp)
    result = predict(
        layout_module,
        [AREA],
        model_timestamp,
        pred_type="test",
        test_area="test_area.gpkg",
        threshold=0.0,
    )
    raster_dir = layout_module.prediction_raster_dir(model_timestamp, AREA)
    assert [p.stem for p in sorted(raster_dir.glob("*.tif"))] == ["Q"]
    merged_path = layout_module.predictions_dir(model_timestamp, AREA) / "merged_shapes_test.gpkg"
    assert merged_path.exists()
    validate_detections(result[AREA])


def test_predict_rejects_bad_arguments(layout_module, model_timestamp: str) -> None:
    with pytest.raises(ValueError, match="pred_type"):
        predict(layout_module, [AREA], model_timestamp, pred_type="bogus")
    with pytest.raises(ValueError, match="test_area"):
        predict(layout_module, [AREA], model_timestamp, pred_type="test")
    with pytest.raises(FileNotFoundError, match="not found"):
        predict(layout_module, [AREA], "19990101-000000")


def test_predict_instance_backend_takes_sparse_path(layout_module, monkeypatch) -> None:
    """Instance-segmentation backends (Mask R-CNN) emit the Detection IR
    directly from per-window instances — no score rasters, no stitching."""

    import importlib

    from lidark.config import TASK_INSTANCE_SEGMENTATION, save_model_config

    predict_module = importlib.import_module("lidark.predict")

    model_id = "stub_instances_1"
    model_dir = layout_module.model_dir(model_id)
    model_dir.mkdir(parents=True)
    save_model_config(
        ModelConfig(
            backend="stub_instances",
            feature_list=tuple(FEATURES),
            img_size=(16, 16),
            pred_stride=8,
            pred_batch_size=8,
        ),
        model_dir / "model_config.json",
    )

    class _StubInstanceBackend:
        task = TASK_INSTANCE_SEGMENTATION

        def load(self, model_dir):  # noqa: ARG002
            return None

        def predict_instances(self, tiles):  # noqa: ARG002
            out = []
            for window in np.asarray(tiles):
                mask = np.zeros(window.shape[:2], dtype=np.uint8)
                mask[2:6, 2:6] = 1
                out.append([{"mask": mask, "score": 0.9, "class_id": 1}])
            return out

    monkeypatch.setattr(predict_module, "create_backend", lambda name: _StubInstanceBackend())
    result = predict(layout_module, [AREA], model_id, threshold=0.5)
    ir = result[AREA]
    validate_detections(ir)
    assert len(ir) > 0
    assert set(ir["score"]) == {0.9}
    assert set(ir["class_id"]) == {1}
    assert set(ir["class_name"]) == {"class_1"}

    # The per-tile raster output is the class-id footprint of the instances.
    raster_dir = layout_module.prediction_raster_dir(model_id, AREA)
    rasters = sorted(raster_dir.glob("*.tif"))
    assert len(rasters) == 4  # every tile of the synthetic grid
    with rasterio.open(rasters[0]) as src:
        arr = src.read(1)
    assert int(arr.max()) == 1 and int(arr.min()) == 0

    # Merged GeoPackage written; geometries georeferenced inside the grid.
    merged_path = layout_module.predictions_dir(model_id, AREA) / "merged_shapes_all.gpkg"
    assert merged_path.exists()
    minx, miny, maxx, maxy = ir.total_bounds
    assert minx >= ORIGIN_X and maxy <= ORIGIN_Y

    # Below-threshold instances are dropped by the facade.
    class _LowScoreBackend(_StubInstanceBackend):
        def predict_instances(self, tiles):  # noqa: ARG002
            out = []
            for window in np.asarray(tiles):
                mask = np.zeros(window.shape[:2], dtype=np.uint8)
                mask[2:6, 2:6] = 1
                out.append([{"mask": mask, "score": 0.1, "class_id": 1}])
            return out

    monkeypatch.setattr(predict_module, "create_backend", lambda name: _LowScoreBackend())
    _clear_rasters(layout_module, model_id)
    result = predict(layout_module, [AREA], model_id, threshold=0.5)
    assert len(result[AREA]) == 0


def test_predict_detection_backend_takes_sparse_path(layout_module, monkeypatch) -> None:
    """Detection backends (Faster R-CNN) take the same sparse path: boxes
    arrive as footprint masks and polygonize into box rectangles."""

    import importlib

    from lidark.config import TASK_INSTANCE_DETECTION, save_model_config

    predict_module = importlib.import_module("lidark.predict")

    model_id = "stub_detections_1"
    model_dir = layout_module.model_dir(model_id)
    model_dir.mkdir(parents=True)
    save_model_config(
        ModelConfig(
            backend="stub_detections",
            feature_list=tuple(FEATURES),
            img_size=(16, 16),
            pred_stride=8,
            pred_batch_size=8,
        ),
        model_dir / "model_config.json",
    )

    class _StubDetectionBackend:
        task = TASK_INSTANCE_DETECTION

        def load(self, model_dir):  # noqa: ARG002
            return None

        def predict_instances(self, tiles):  # noqa: ARG002
            out = []
            for window in np.asarray(tiles):
                mask = np.zeros(window.shape[:2], dtype=np.uint8)
                mask[2:6, 2:6] = 1  # a box footprint
                out.append([{"mask": mask, "score": 0.9, "class_id": 1}])
            return out

    monkeypatch.setattr(predict_module, "create_backend", lambda name: _StubDetectionBackend())
    result = predict(layout_module, [AREA], model_id, threshold=0.5)
    ir = result[AREA]
    validate_detections(ir)
    assert len(ir) > 0
    assert set(ir["score"]) == {0.9}
    assert set(ir["class_id"]) == {1}

    # The merged GeoPackage written; geometries georeferenced inside the grid.
    merged_path = layout_module.predictions_dir(model_id, AREA) / "merged_shapes_all.gpkg"
    assert merged_path.exists()
    minx, miny, maxx, maxy = ir.total_bounds
    assert minx >= ORIGIN_X and maxy <= ORIGIN_Y


def test_predict_skips_existing_rasters(
    layout_module, model_timestamp: str, caplog: pytest.LogCaptureFixture
) -> None:
    # Produce all rasters, then rerun: nothing may be recomputed.
    _clear_rasters(layout_module, model_timestamp)
    predict(layout_module, [AREA], model_timestamp, threshold=0.0)
    raster_dir = layout_module.prediction_raster_dir(model_timestamp, AREA)
    before = {p.name for p in raster_dir.glob("*.tif")}
    assert len(before) == 4
    caplog.clear()
    result = predict(layout_module, [AREA], model_timestamp, threshold=0.0)
    assert {p.name for p in raster_dir.glob("*.tif")} == before
    assert any("already exists" in message for message in caplog.messages)
    validate_detections(result[AREA])


def test_predict_tile_failure_logged_and_skipped(
    layout_module, model_timestamp: str, caplog: pytest.LogCaptureFixture
) -> None:
    # Corrupt one feature tile; the other tiles must still predict.
    _clear_rasters(layout_module, model_timestamp)
    feature_dir = layout_module.tpi_dir(AREA, 20)
    victim = feature_dir / "Q_TPI.tif"
    backup = victim.read_bytes()
    try:
        victim.write_bytes(b"not a geotiff at all")
        result = predict(layout_module, [AREA], model_timestamp, threshold=0.0)
        assert any("could not predict tile" in m for m in caplog.messages)
        validate_detections(result[AREA])
        # The corrupt tile produced no raster; the others did.
        rasters = {
            p.stem for p in layout_module.prediction_raster_dir(model_timestamp, AREA).glob("*.tif")
        }
        assert "Q" not in rasters
        assert len(rasters) == 3
    finally:
        victim.write_bytes(backup)


def test_predict_write_tile_shapes(layout_module, model_timestamp: str) -> None:
    # Clear rasters so tiles re-predict with shape writing enabled.
    raster_dir = layout_module.prediction_raster_dir(model_timestamp, AREA)
    for p in raster_dir.glob("*.tif"):
        p.unlink()
    predict(
        layout_module,
        [AREA],
        model_timestamp,
        threshold=0.0,
        write_tile_shape_files=True,
    )
    shapes_dir = layout_module.prediction_shapes_dir(model_timestamp, AREA)
    assert len(list(shapes_dir.glob("*.gpkg"))) == 4


def _dtm_and_tpi_tiles(layout) -> tuple[Path, Path]:
    """Write the DTM and TPI tiles without any product artifacts."""
    from tests.conftest import TILE_NAMES, tile_array, tile_transform

    from lidark.io.rasters import write_band

    dtm_dir = layout.dtm_dir(AREA)
    dtm_dir.mkdir(parents=True)
    for i, name in enumerate(TILE_NAMES):
        row, col = divmod(i, 2)
        write_band(
            dtm_dir / name,
            tile_array(row, col),
            transform=tile_transform(row, col),
            crs=layout.epsg,
        )

    feature_dir = layout.tpi_dir(AREA, 20)
    feature_dir.mkdir(parents=True)
    for i, name in enumerate(TILE_NAMES):
        row, col = divmod(i, 2)
        arr = (np.indices(TILE_SHAPE)[0] * 3 - 100 + 30 * row + 60 * col).astype(np.int16)
        write_band(
            feature_dir / f"{Path(name).stem}_TPI.tif",
            arr,
            transform=tile_transform(row, col),
            crs=layout.epsg,
            nodata=None,
        )
    return dtm_dir, feature_dir


def test_read_tile_features_rebuilds_missing_artifacts(layout) -> None:
    """The feature reader rebuilds missing tile indexes and VRT mosaics."""
    from tests.conftest import tile_array

    from lidark.predict import _read_tile_features

    dtm_dir, feature_dir = _dtm_and_tpi_tiles(layout)
    grid = 2 * TILE_SHAPE[0] * 2.0  # the full 2x2 grid in metres
    footprint = box(ORIGIN_X, ORIGIN_Y - grid, ORIGIN_X + grid, ORIGIN_Y)

    stack, _transform, _crs = _read_tile_features(layout, AREA, tuple(FEATURES), footprint)

    assert stack.shape[:2] == (2 * TILE_SHAPE[0], 2 * TILE_SHAPE[1])
    for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        sub = stack[
            row * TILE_SHAPE[0] : (row + 1) * TILE_SHAPE[0],
            col * TILE_SHAPE[1] : (col + 1) * TILE_SHAPE[1],
            0,
        ]
        assert np.array_equal(sub, tile_array(row, col))
    # The missing artifacts were rebuilt on disk.
    assert (dtm_dir / "dtm.vrt").exists()
    assert (feature_dir / "tpi.vrt").exists()


def test_read_tile_features_falls_back_when_vrt_build_fails(
    layout_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed VRT build only warns: the matching tile rasters are read."""
    from tests.conftest import tile_array

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("gdalbuildvrt failed")

    # import_module returns the real module object; a plain import would
    # yield the like-named predict() function re-exported by lidark/__init__.
    predict_module = importlib.import_module("lidark.predict")
    monkeypatch.setattr(predict_module, "ensure_vrt", _boom)
    from lidark.predict import _read_tile_features

    # Footprint strictly inside tile (0,0): the fallback reads that tile.
    footprint = box(ORIGIN_X + 4.0, ORIGIN_Y - 20.0, ORIGIN_X + 12.0, ORIGIN_Y - 8.0)
    stack, transform, _crs = _read_tile_features(layout_module, AREA, tuple(FEATURES), footprint)

    assert stack.shape == (*TILE_SHAPE, 2)
    assert np.array_equal(stack[..., 0], tile_array(0, 0))
    expected_tpi = (np.indices(TILE_SHAPE)[0] * 3 - 100).astype(np.int16)  # tile (0,0)
    assert np.array_equal(stack[..., 1], expected_tpi)
    assert transform == tile_transform(0, 0)
