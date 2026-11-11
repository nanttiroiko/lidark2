"""Multi-class detection support (plan 3.3, MC1-MC7).

Covers: compact class-id mapping (MC2), class-balanced sampling and
class-weighted loss plumbing (MC3), class-stratified validation split
(MC7), argmax-constrained dense conversion with per-class thresholds
(MC1/MC5), class-name plumbing through config and evaluation (MC4), and
cross-class confusion reporting (MC6). Binary behavior is asserted to be
unchanged wherever a gap touches a shared code path.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import box
from tests.conftest import (
    AREA,
    ORIGIN_X,
    ORIGIN_Y,
    tile_array,
    tile_transform,
    write_annotation,
)

from lidark.cli import _class_spec
from lidark.config import DEFAULT_EPSG, ModelConfig, load_model_config, save_model_config
from lidark.dataset import Dataset, create_dataset
from lidark.evaluate import evaluate
from lidark.io.tiles import build_tile_index
from lidark.predict import _dense_class_masks, predict
from lidark.train import _split, train

IMG = (16, 16)


def _tile_box(row: int, col: int, margin: float = 0.0):
    w = 32 * 2.0
    h = 32 * 2.0
    x0 = ORIGIN_X + col * w
    y_top = ORIGIN_Y - row * h
    return box(x0 + margin, y_top - h + margin, x0 + w - margin, y_top - margin)


# ---------------------------------------------------------------- MC2 (ids)


def test_sparse_label_ids_remap_to_compact_classes(
    layout,
    dtm_product: Path,
    caplog,  # noqa: ARG001
) -> None:
    """Sparse label ids (e.g. {1, 5}) become compact 1..C; the map is
    recorded and the labels are stored with compact ids (MC2)."""
    write_annotation(
        layout,
        "sparse.shp",
        [
            box(ORIGIN_X + 10.0, ORIGIN_Y - 20.0, ORIGIN_X + 30.0, ORIGIN_Y - 8.0),
            box(ORIGIN_X + 10.0, ORIGIN_Y - 50.0, ORIGIN_X + 30.0, ORIGIN_Y - 40.0),
        ],
        class_ids=[1, 5],
    )
    with caplog.at_level(logging.WARNING, logger="lidark.dataset"):
        ds = create_dataset(
            layout, AREA, ["01_DTM"], "sparse.shp", img_size=IMG, dataset_id="mc2", seed=1
        )
    assert ds.num_classes == 2  # observed ids, not max(id) == 5
    assert ds.class_id_map == {1: 1, 5: 2}
    assert ds.meta["class_names"] == ["class_1", "class_2"]
    assert any("non-contiguous" in record.message for record in caplog.records)
    values = set(int(v) for v in np.unique(np.asarray(ds.labels)))
    assert values <= {0, 1, 2}
    assert {1, 2} <= values  # both classes present with compact ids


def test_contiguous_label_ids_keep_identity_map(layout, dtm_product: Path) -> None:
    write_annotation(
        layout,
        "contig.shp",
        [
            box(ORIGIN_X + 10.0, ORIGIN_Y - 20.0, ORIGIN_X + 30.0, ORIGIN_Y - 8.0),
            box(ORIGIN_X + 10.0, ORIGIN_Y - 50.0, ORIGIN_X + 30.0, ORIGIN_Y - 40.0),
        ],
        class_ids=[1, 2],
    )
    ds = create_dataset(
        layout, AREA, ["01_DTM"], "contig.shp", img_size=IMG, dataset_id="mc2b", seed=1
    )
    assert ds.num_classes == 2
    assert ds.class_id_map == {1: 1, 2: 2}


# -------------------------------------------------- MC3 (balance, weights)


def test_balanced_sampling_equalizes_per_class_windows(layout, dtm_product: Path) -> None:
    """Balanced sampling normalizes windows per object per class so every
    class contributes equally (MC3); the default keeps the original
    2-per-object behavior."""
    write_annotation(
        layout,
        "imbalance.shp",
        [
            box(ORIGIN_X + 10.0, ORIGIN_Y - 20.0, ORIGIN_X + 18.0, ORIGIN_Y - 14.0),
            box(ORIGIN_X + 24.0, ORIGIN_Y - 20.0, ORIGIN_X + 32.0, ORIGIN_Y - 14.0),
            box(ORIGIN_X + 10.0, ORIGIN_Y - 40.0, ORIGIN_X + 18.0, ORIGIN_Y - 34.0),
            box(ORIGIN_X + 24.0, ORIGIN_Y - 40.0, ORIGIN_X + 32.0, ORIGIN_Y - 34.0),
        ],
        class_ids=[1, 2, 2, 2],
    )
    plain = create_dataset(
        layout, AREA, ["01_DTM"], "imbalance.shp", img_size=IMG, dataset_id="plain", seed=2
    )
    balanced = create_dataset(
        layout,
        AREA,
        ["01_DTM"],
        "imbalance.shp",
        img_size=IMG,
        class_balanced=True,
        dataset_id="balanced",
        seed=2,
    )
    plain_counts = Counter(c for c in plain.window_classes or [] if c > 0)
    balanced_counts = Counter(c for c in balanced.window_classes or [] if c > 0)
    assert plain_counts == {1: 2, 2: 6}  # original 2-per-object behavior
    # Largest class keeps 2 windows per object; the rare class is oversampled
    # to the same total (3 objects * 2 windows == 1 object * 6 windows).
    assert balanced_counts == {1: 6, 2: 6}
    assert balanced.meta["class_balanced"] is True
    assert plain.meta["class_balanced"] is False


def test_model_config_class_weights_and_map_round_trip(tmp_path: Path) -> None:
    """class_weights and class_id_map persist to model_config.json (MC3/MC2)."""
    config = ModelConfig(
        num_classes=2,
        class_names=("kiln", "pit"),
        class_weights=(1.0, 2.0),
        class_id_map=((1, 1), (5, 2)),
    )
    path = tmp_path / "model_config.json"
    save_model_config(config, path)
    loaded = load_model_config(path)
    assert loaded.num_classes == 2
    assert loaded.class_names == ("kiln", "pit")
    assert loaded.class_weights == (1.0, 2.0)
    assert loaded.class_id_map == ((1, 1), (5, 2))


def test_train_rejects_wrong_length_class_options(
    layout,
    dtm_product: Path,
    tpi_layer: Path,  # noqa: ARG001
) -> None:
    """class_names / class_weights must match the dataset's class count."""
    write_annotation(
        layout,
        "two_classes.shp",
        [
            _tile_box(0, 0, margin=24.0),
            box(ORIGIN_X + 50.0, ORIGIN_Y - 20.0, ORIGIN_X + 70.0, ORIGIN_Y - 8.0),
        ],
        class_ids=[1, 2],
    )
    config = ModelConfig(
        feature_list=("01_DTM", "02_TPI/radius_20"),
        img_size=IMG,
        unet_size=4,
        batch_size=4,
        max_epochs=1,
    )
    with pytest.raises(ValueError, match="class_names"):
        train(
            layout,
            AREA,
            model_name="bad_names",
            true_objects="two_classes.shp",
            model_config=config,
            class_names=("only_one",),
        )
    with pytest.raises(ValueError, match="class_weights"):
        train(
            layout,
            AREA,
            model_name="bad_weights",
            true_objects="two_classes.shp",
            model_config=config,
            class_weights=(1.0,),
        )


def test_train_multiclass_records_classes_in_config(
    layout, dtm_product: Path, tpi_layer: Path
) -> None:
    """A 1-epoch multi-class run stores num_classes, class_names,
    class_weights, and the (identity) class_id_map in model_config.json."""
    write_annotation(
        layout,
        "two_classes.shp",
        [
            _tile_box(0, 0, margin=24.0),
            box(ORIGIN_X + 50.0, ORIGIN_Y - 20.0, ORIGIN_X + 70.0, ORIGIN_Y - 8.0),
        ],
        class_ids=[1, 2],
    )
    model_id = train(
        layout,
        AREA,
        model_name="multiclass_v1",
        true_objects="two_classes.shp",
        model_config=ModelConfig(
            feature_list=("01_DTM", "02_TPI/radius_20"),
            img_size=IMG,
            unet_size=4,
            batch_size=4,
            max_epochs=1,
            pred_stride=8,
        ),
        class_names=("kiln", "pit"),
        class_weights=(1.0, 2.0),
    )
    config = load_model_config(layout.model_dir(model_id) / "model_config.json")
    assert config.num_classes == 2
    assert config.class_names == ("kiln", "pit")
    assert config.class_weights == (1.0, 2.0)
    assert config.class_id_map == ((1, 1), (2, 2))
    log = json.loads((layout.model_dir(model_id) / "training_log.json").read_text(encoding="utf-8"))
    assert log["class names"] == ["kiln", "pit"]
    assert log["class weights"] == [1.0, 2.0]


# ------------------------------------------------------------ MC7 (split)


def _synthetic_dataset(n: int, num_classes: int, window_classes: list[int]) -> Dataset:
    meta = {"num_classes": num_classes, "window_classes": window_classes}
    samples = np.zeros((n, *IMG, 1), dtype=np.int8)
    labels = np.zeros((n, *IMG), dtype=np.uint8)
    return Dataset("synthetic", Path("."), samples, labels, meta)


def test_split_stratifies_both_halves() -> None:
    """Every foreground class with >= 2 windows lands in both halves (MC7)."""
    dataset = _synthetic_dataset(10, 2, [1] * 6 + [2] * 4)
    data = _split(dataset, ModelConfig(val_split=0.2, seed=0))
    assert len(data.val_indices) == 2
    assert len(data.train_indices) == 8
    classes = dataset.window_classes or []
    val_classes = {classes[i] for i in data.val_indices}
    train_classes = {classes[i] for i in data.train_indices}
    assert val_classes == {1, 2}
    assert train_classes == {1, 2}


def test_split_binary_keeps_original_semantics() -> None:
    """Binary datasets keep the exact original shuffle + split (parity)."""
    dataset = _synthetic_dataset(10, 1, [1] * 8 + [0] * 2)
    data = _split(dataset, ModelConfig(val_split=0.2, seed=0))
    rng = np.random.default_rng(0)
    expected = rng.permutation(10)
    assert np.array_equal(np.sort(data.val_indices), np.sort(expected[-2:]))
    assert np.array_equal(np.sort(data.train_indices), np.sort(expected[:-2]))


def test_split_single_window_class_warns(caplog) -> None:
    """A class with one window cannot be stratified; it is left as-is and
    warned about (MC7)."""
    dataset = _synthetic_dataset(10, 2, [1] * 9 + [2])
    with caplog.at_level(logging.WARNING, logger="lidark.train"):
        data = _split(dataset, ModelConfig(val_split=0.2, seed=0))
    assert any("single window" in record.message for record in caplog.records)
    assert len(data.val_indices) == 2
    assert len(data.train_indices) == 8


# ------------------------------------------- MC1/MC5 (masks, thresholds)


def test_dense_class_masks_argmax_constrained() -> None:
    """A pixel belongs to its argmax class only, and only if that class's
    own probability passes its threshold (MC1)."""
    prob = np.zeros((2, 2, 2), dtype=np.float32)
    prob[0] = [[0.9, 0.4], [0.6, 0.55]]  # class 1
    prob[1] = [[0.6, 0.35], [0.7, 0.45]]  # class 2
    masks, raster = _dense_class_masks(prob, [0.5, 0.5])
    # (0,0): argmax class 1, both pass -> class 1.
    # (0,1): argmax class 1 but 0.4 < 0.5 -> background (no class claims it).
    # (1,0): argmax class 2 (0.7 > 0.6), both pass -> class 2 only.
    # (1,1): argmax class 1 (0.55), passes -> class 1.
    assert masks[0].tolist() == [[True, False], [False, True]]
    assert masks[1].tolist() == [[False, False], [True, False]]
    assert raster.tolist() == [[1, 0], [2, 1]]


def test_dense_class_masks_binary_parity() -> None:
    prob = np.array([[[0.2, 0.7], [0.5, 0.9]]], dtype=np.float32)
    masks, raster = _dense_class_masks(prob, [0.5])
    assert masks.shape == (1, 2, 2)
    assert masks[0].tolist() == [[False, True], [False, True]]
    assert raster.tolist() == [[0, 1], [0, 1]]


def test_dense_class_masks_per_class_thresholds() -> None:
    """Per-class thresholds: a pixel passes on its argmax class's own
    threshold (MC5)."""
    prob = np.zeros((2, 1, 2), dtype=np.float32)
    prob[0] = 0.6  # class 1
    prob[1] = 0.75  # class 2 is argmax on the first pixel
    prob[0, 0, 1] = 0.9  # class 1 is argmax on the second pixel
    prob[1, 0, 1] = 0.2
    _, raster = _dense_class_masks(prob, [0.5, 0.9])
    # First pixel: argmax class 2 below its 0.9 threshold -> background;
    # second pixel: argmax class 1 passes 0.5.
    assert raster.tolist() == [[0, 1]]
    _, raster = _dense_class_masks(prob, [0.5, 0.5])
    assert raster.tolist() == [[2, 1]]


class _StubDenseBackend:
    """Dense backend whose class-1 probability is high on the tile's top
    half (pixel value < 26) and class-2 on the bottom half."""

    task = "semantic_segmentation"

    def load(self, model_dir):  # noqa: ARG002
        return None

    def predict_tiles(self, tiles):
        values = np.asarray(tiles, dtype=np.float32)[..., 0]
        top = values < 26.0
        out = np.zeros((*values.shape, 2), dtype=np.float32)
        out[top] = (0.9, 0.1)
        out[~top] = (0.1, 0.9)
        return out


@pytest.fixture()
def novrt_product(layout):
    """A single-tile DTM product with a tile index but no VRT (the VRT
    read path is not needed here)."""
    from lidark.io.rasters import write_band

    dtm_dir = layout.dtm_dir(AREA)
    dtm_dir.mkdir(parents=True, exist_ok=True)
    write_band(
        dtm_dir / "only.tif",
        tile_array(0, 0),
        transform=tile_transform(0, 0),
        crs=layout.epsg,
    )
    build_tile_index(dtm_dir, default_epsg=layout.epsg)
    return dtm_dir


def _stub_predict(monkeypatch, layout, novrt_product, **config_overrides):  # noqa: ARG001
    """Register the stub dense backend and write its model config."""
    import importlib
    from dataclasses import replace

    from lidark.models.base import CONFIG_FILENAME

    predict_module = importlib.import_module("lidark.predict")
    monkeypatch.setattr(predict_module, "create_backend", lambda name: _StubDenseBackend())
    config = replace(
        ModelConfig(
            backend="stub_dense",
            feature_list=("01_DTM",),
            img_size=(16, 16),
            num_classes=2,
            class_names=("kiln", "pit"),
            pred_stride=8,
            pred_batch_size=8,
        ),
        **config_overrides,
    )
    model_id = "stub_dense_1"
    model_dir = layout.model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_model_config(config, model_dir / CONFIG_FILENAME)
    return model_id


def _clear_predictions(layout, model_id: str) -> None:
    import shutil

    shutil.rmtree(layout.predictions_dir(model_id), ignore_errors=True)


def test_predict_multiclass_ir_and_raster_agree(layout, novrt_product, monkeypatch) -> None:
    """The dense multi-class IR and the class-id raster are cut from the
    same argmax decision; classes never overlap in the IR (MC1), and class
    names come from the model config (MC4)."""
    model_id = _stub_predict(monkeypatch, layout, novrt_product)
    _clear_predictions(layout, model_id)
    result = predict(layout, [AREA], model_id, threshold=0.5)
    ir = result[AREA]
    assert set(ir["class_id"]) == {1, 2}
    assert set(ir.loc[ir["class_id"] == 1, "class_name"]) == {"kiln"}
    assert set(ir.loc[ir["class_id"] == 2, "class_name"]) == {"pit"}
    # No cross-class overlap: the two detections share only a boundary.
    geom_1 = ir.loc[ir["class_id"] == 1, "geometry"].iloc[0]
    geom_2 = ir.loc[ir["class_id"] == 2, "geometry"].iloc[0]
    assert geom_1.intersection(geom_2).area == 0.0
    # IR and raster agree: each geometry's centroid pixel carries its id.
    with rasterio.open(layout.prediction_raster_dir(model_id, AREA) / "only.tif") as src:
        raster = src.read(1)
        for _, det in ir.iterrows():
            px, py = det.geometry.centroid.x, det.geometry.centroid.y
            row, col = src.index(px, py)
            assert int(raster[row, col]) == int(det["class_id"])
    # Top half is class 1, bottom half class 2 (value 25 vs 26 boundary).
    # Rows >= 24 stay background: the tiler leaves the bottom edge strip
    # uncovered (original behavior), so assert on covered pixels.
    assert int(raster[0, 0]) == 1
    assert int(raster[20, 0]) == 2


def test_predict_per_class_thresholds(layout, novrt_product, monkeypatch) -> None:
    """A per-class threshold mapping is honored independently (MC5) and is
    equivalent to the scalar when all values match (parity)."""
    model_id = _stub_predict(monkeypatch, layout, novrt_product)
    _clear_predictions(layout, model_id)
    scalar = predict(layout, [AREA], model_id, threshold=0.5)[AREA]
    _clear_predictions(layout, model_id)
    mapped = predict(layout, [AREA], model_id, threshold={1: 0.5, 2: 0.5})[AREA]
    assert len(scalar) == len(mapped)
    assert set(scalar["class_id"]) == set(mapped["class_id"]) == {1, 2}

    # Class 2's 0.9 score sits below 0.95: only class 1 survives.
    _clear_predictions(layout, model_id)
    strict = predict(layout, [AREA], model_id, threshold={1: 0.5, 2: 0.95})[AREA]
    assert set(strict["class_id"]) == {1}

    # A mapping missing a class of the model raises before any tile runs.
    with pytest.raises(ValueError, match="threshold"):
        predict(layout, [AREA], model_id, threshold={1: 0.5})


# ------------------------------------- MC4/MC5/MC6 (evaluate integration)


def _write_named_model_config(layout, model_id: str, **overrides) -> None:
    from dataclasses import replace

    config = replace(
        ModelConfig(
            backend="unet",
            feature_list=("01_DTM",),
            num_classes=2,
            class_names=("kiln", "pit"),
            class_id_map=((1, 1), (5, 2)),
        ),
        **overrides,
    )
    model_dir = layout.model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_model_config(config, model_dir / "model_config.json")


def _write_predictions(layout, model_id: str, rows: list[dict]) -> None:
    frame = gpd.GeoDataFrame(rows, crs=DEFAULT_EPSG)
    frame["area"] = frame.geometry.area
    out = layout.predictions_dir(model_id, AREA) / "merged_shapes_all.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(out)


def _write_eval_annotations(layout, eval_name: str, rows: list[dict]) -> None:
    frame = gpd.GeoDataFrame(rows, crs=DEFAULT_EPSG)
    out_dir = layout.evaluate_annotations_dir(eval_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_file(out_dir / "true.shp")


def test_evaluate_remaps_label_ids_and_uses_class_names(layout) -> None:
    """Labels in the original id space pair with predictions in compact
    ids via class_id_map, and rows carry the model's class names (MC2/MC4)."""
    model_id = "named_1"
    _write_named_model_config(layout, model_id)
    _write_eval_annotations(
        layout,
        "remap",
        [
            {"class_id": 1, "geometry": box(0, 0, 10, 10)},  # original id 1
            {"class_id": 5, "geometry": box(20, 0, 30, 10)},  # original id 5
        ],
    )
    _write_predictions(
        layout,
        model_id,
        [
            {"score": 0.9, "class_id": 1, "class_name": "kiln", "geometry": box(0, 0, 10, 10)},
            {"score": 0.9, "class_id": 2, "class_name": "pit", "geometry": box(20, 0, 30, 10)},
        ],
    )
    report = evaluate(layout, "true.shp", model_id, "remap")
    rows = report.loc[report["class_id"] != "overall"]
    assert rows["class_id"].tolist() == [1, 2]  # compact ids after remap
    assert rows["class_name"].tolist() == ["kiln", "pit"]
    # The class-5 label paired with the compact-2 prediction.
    assert rows.loc[rows["class_id"] == 2, "Fully detected"].iloc[0] == 1
    assert rows.loc[rows["class_id"] == 1, "Fully detected"].iloc[0] == 1


def test_evaluate_unknown_label_id_gets_not_detected_row(layout, caplog) -> None:
    """A label class the model never saw keeps its id and shows up as Not
    detected (truthful, not silently dropped)."""
    model_id = "named_2"
    _write_named_model_config(layout, model_id)
    _write_eval_annotations(
        layout,
        "unknown",
        [
            {"class_id": 1, "geometry": box(0, 0, 10, 10)},
            {"class_id": 7, "geometry": box(20, 0, 30, 10)},  # not in the map
        ],
    )
    _write_predictions(
        layout,
        model_id,
        [{"score": 0.9, "class_id": 1, "class_name": "kiln", "geometry": box(0, 0, 10, 10)}],
    )
    with caplog.at_level(logging.WARNING, logger="lidark.evaluate"):
        report = evaluate(layout, "true.shp", model_id, "unknown")
    assert any("class_id_map" in record.message for record in caplog.records)
    row = report.loc[report["class_id"] == 7].iloc[0]
    assert row["Not detected"] == 1
    assert row["Predicted"] == 0


def test_evaluate_per_class_cuts(layout) -> None:
    """size_cut / prob_cut accept per-class mappings (MC5); a mapping must
    cover every predicted class."""
    model_id = "cuts_1"
    _write_eval_annotations(
        layout,
        "cuts",
        [
            {"class_id": 1, "geometry": box(0, 0, 10, 10)},
            {"class_id": 2, "geometry": box(20, 0, 30, 10)},
        ],
    )
    _write_predictions(
        layout,
        model_id,
        [
            {"score": 0.5, "class_id": 1, "class_name": "class_1", "geometry": box(0, 0, 10, 10)},
            {"score": 0.5, "class_id": 2, "class_name": "class_2", "geometry": box(20, 0, 30, 10)},
        ],
    )
    # Scalar cut (original behavior): both 0.5-score predictions drop at 0.8.
    report = evaluate(layout, "true.shp", model_id, "cuts", prob_cut=0.8)
    assert report.loc[report["class_id"] != "overall", "Predicted"].tolist() == [0, 0]
    # Per-class cut: class 1 kept, class 2 dropped.
    report = evaluate(layout, "true.shp", model_id, "cuts", prob_cut={1: 0.4, 2: 0.9})
    rows = report.loc[report["class_id"] != "overall"]
    assert rows["Predicted"].tolist() == [1, 0]
    assert rows["Fully detected"].tolist() == [1, 0]
    # Per-class size cut: class 2's prediction is small, class 1's is large.
    _write_predictions(
        layout,
        model_id,
        [
            {"score": 0.9, "class_id": 1, "class_name": "class_1", "geometry": box(0, 0, 10, 10)},
            {"score": 0.9, "class_id": 2, "class_name": "class_2", "geometry": box(20, 0, 22, 2)},
        ],
    )
    report = evaluate(layout, "true.shp", model_id, "cuts", size_cut={1: 0.0, 2: 50.0})
    rows = report.loc[report["class_id"] != "overall"]
    assert rows["Predicted"].tolist() == [1, 0]
    # A mapping missing a predicted class raises.
    with pytest.raises(ValueError, match="prob_cut"):
        evaluate(layout, "true.shp", model_id, "cuts", prob_cut={1: 0.4})


def test_evaluate_confusion_report(layout) -> None:
    """The confusion CSV attributes detected labels to the class of their
    best-overlap prediction and groups unlabeled predictions by predicted
    class (MC6)."""
    model_id = "confuse_1"
    _write_eval_annotations(
        layout,
        "confuse",
        [
            {"class_id": 1, "geometry": box(0, 0, 10, 10)},
            {"class_id": 2, "geometry": box(40, 0, 50, 10)},
        ],
    )
    _write_predictions(
        layout,
        model_id,
        [
            # Detected on the class-1 label, but predicted as class 2.
            {"score": 0.9, "class_id": 2, "class_name": "class_2", "geometry": box(0, 0, 10, 10)},
            # Detected on the class-2 label, but predicted as class 1.
            {"score": 0.9, "class_id": 1, "class_name": "class_1", "geometry": box(40, 0, 50, 10)},
            # Inside the AOI (the label bbox) but overlapping no label.
            {"score": 0.9, "class_id": 1, "class_name": "class_1", "geometry": box(20, 0, 30, 10)},
        ],
    )
    evaluate(layout, "true.shp", model_id, "confuse")
    path = layout.reports_dir(model_id) / "confuse_confusion.csv"
    assert path.exists()
    confusion = pd.read_csv(path)
    # The true-class column mixes ids and the "none" marker, so read_csv
    # yields strings there.
    got = {
        (str(row["true_class_id"]), int(row["pred_class_id"])): row["count"]
        for _, row in confusion.iterrows()
    }
    assert got == {("1", 2): 1, ("2", 1): 1, ("none", 1): 1}


# ------------------------------------------------------------------- CLI


def test_cli_class_spec_parsing() -> None:
    """--threshold / --size-cut / --prob-cut accept a scalar or CLASS=VALUE
    tokens (MC5)."""
    assert _class_spec(["0.5"], "--threshold") == 0.5
    assert _class_spec(["1=0.4", "2=0.6"], "--threshold") == {1: 0.4, 2: 0.6}
    with pytest.raises(ValueError, match="CLASS=VALUE"):
        _class_spec(["0.4", "2=0.6"], "--threshold")
    with pytest.raises(ValueError, match="CLASS=VALUE"):
        _class_spec(["just_a_number", "1=0.4"], "--prob-cut")


def test_cli_train_exposes_class_flags() -> None:
    """The train subcommand exposes the multi-class flags."""
    from lidark.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--project-folder",
            "x",
            "--area",
            AREA,
            "--name",
            "m",
            "--true-objects",
            "t.shp",
            "--raster-type",
            "01_DTM",
            "--class-names",
            "kiln",
            "pit",
            "--class-weights",
            "1.0",
            "2.0",
            "--balanced-sampling",
        ]
    )
    assert args.class_names == ["kiln", "pit"]
    assert args.class_weights == [1.0, 2.0]
    assert args.balanced_sampling is True
