"""Tests for lidark.train: full pipeline, checkpoint, log, early stopping."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box
from tests.conftest import (
    AREA,
    AREA2,
    AREA2_ORIGIN_X,
    AREA2_ORIGIN_Y,
    ORIGIN_X,
    ORIGIN_Y,
    TILE_SHAPE,
    write_annotation,
)

from lidark.config import ModelConfig, load_model_config
from lidark.dataset import Dataset, create_dataset
from lidark.train import _split, train

FEATURES = ["01_DTM", "02_TPI/radius_20"]
IMG = (16, 16)


@pytest.fixture()
def annotations(layout) -> None:
    """Labels in every tile so all four tiles contribute windows."""
    w = TILE_SHAPE[1] * 2.0
    h = TILE_SHAPE[0] * 2.0
    geometry = []
    for row in range(2):
        for col in range(2):
            x0 = ORIGIN_X + col * w + 20.0
            y_top = ORIGIN_Y - row * h - 20.0
            geometry.append(box(x0, y_top - 8.0, x0 + 8.0, y_top))
    write_annotation(layout, "true.shp", geometry, class_ids=[1] * 4)
    write_annotation(
        layout, "false.shp", [box(ORIGIN_X + 4, ORIGIN_Y - 60, ORIGIN_X + 12, ORIGIN_Y - 52)]
    )


def _config(**overrides: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "feature_list": tuple(FEATURES),
        "img_size": IMG,
        "unet_size": 4,
        "batch_size": 4,
        "max_epochs": 1,
        "seed": 0,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def test_train_writes_checkpoint_log_and_metadata(
    layout, dtm_product: Path, tpi_layer: Path, annotations
) -> None:
    mid = train(
        layout,
        AREA,
        FEATURES,
        "true.shp",
        "false.shp",
        model_name="tar kilns",
        model_config=_config(),
    )
    # Model id: sanitized user-supplied name + "_" + %Y%m%d-%H%M%S.
    assert re.fullmatch(r"tar_kilns_\d{8}-\d{6}", mid), mid

    model_dir = layout.model_dir(mid)
    assert (model_dir / "model.pt").exists()
    assert (model_dir / "model_config.json").exists()

    # model_config.json round-trips.
    loaded = load_model_config(model_dir / "model_config.json")
    assert loaded.feature_list == tuple(FEATURES)
    assert loaded.img_size == IMG
    assert loaded.num_classes == 1

    # Log (pretty-printed JSON) with the original key set plus backend/classes.
    log = json.loads((layout.model_dir(mid) / "training_log.json").read_text(encoding="utf-8"))
    keys = set(log)
    assert log["log_id"] == mid
    for original_key in (
        "log_id",
        "data_id",
        "epsg",
        "num samples",
        "augmented",
        "features",
        "img size",
        "learning rate",
        "batch size",
        "validation split",
        "max epochs",
        "unet size",
        "dropout",
        "best iou",
    ):
        assert original_key in keys, original_key
    assert "backend" in keys and log["backend"] == "unet"
    # 4 true objects (one per tile) + 1 false object, 2 windows each.
    assert log["num samples"] == 10
    # Full per-epoch history: one series per metric, aligned with epochs run.
    assert "history" in keys
    hist = log["history"]
    assert set(hist) >= {"loss", "iou", "val_iou", "imgs_per_second"}
    assert len(hist["loss"]) == log["epochs run"]
    assert all(len(series) == log["epochs run"] for series in hist.values())


def test_train_with_existing_dataset(
    layout, dtm_product: Path, tpi_layer: Path, annotations
) -> None:
    create_dataset(
        layout,
        AREA,
        FEATURES,
        "true.shp",
        "false.shp",
        img_size=IMG,
        dataset_id="premade",
        seed=0,
    )
    timestamp = train(
        layout, AREA, dataset_id="premade", model_name="premade", model_config=_config()
    )
    model_dir = layout.model_dir(timestamp)
    assert (model_dir / "model.pt").exists()
    log = json.loads(
        (layout.model_dir(timestamp) / "training_log.json").read_text(encoding="utf-8")
    )
    assert log["data_id"] == "premade"


def test_train_requires_inputs(layout) -> None:
    with pytest.raises(ValueError, match="feature_list"):
        train(layout, AREA, model_name="x")
    with pytest.raises(ValueError, match="true_objects"):
        train(layout, AREA, FEATURES, model_name="x", model_config=_config())
    with pytest.raises(ValueError, match="model name"):
        train(layout, AREA, FEATURES, "true.shp", model_name="  ", model_config=_config())


def test_sanitize_model_name() -> None:
    from lidark.train import sanitize_model_name

    assert sanitize_model_name("Tar Kilns v2") == "Tar_Kilns_v2"
    assert sanitize_model_name('weird/name:1? "best"') == "weird_name_1_best"
    assert sanitize_model_name("..leading") == "leading"
    with pytest.raises(ValueError, match="model name"):
        sanitize_model_name("  ?/  ")


def test_early_stopping_honored(layout, dtm_product: Path, tpi_layer: Path) -> None:
    # All-background windows: the true-object label lies outside every tile
    # (no windows from it) and the false object contributes background
    # windows only, so the monitored IoU saturates immediately (0.0 until
    # the model predicts nothing, then 1.0) and EarlyStopping(patience=10)
    # must cut the run far below max_epochs=50.
    write_annotation(
        layout,
        "far.shp",
        [box(ORIGIN_X + 5000.0, ORIGIN_Y - 5010.0, ORIGIN_X + 5010.0, ORIGIN_Y - 5000.0)],
    )
    write_annotation(
        layout, "false.shp", [box(ORIGIN_X + 4, ORIGIN_Y - 60, ORIGIN_X + 12, ORIGIN_Y - 52)]
    )
    timestamp = train(
        layout,
        AREA,
        FEATURES,
        "far.shp",
        "false.shp",
        model_name="earlystop",
        model_config=_config(max_epochs=50),
    )
    log = json.loads(
        (layout.model_dir(timestamp) / "training_log.json").read_text(encoding="utf-8")
    )
    epochs_run = log["epochs run"]
    assert 1 <= epochs_run < 50


def test_train_val_split_window_mode_is_seeded(
    layout, dtm_product: Path, tpi_layer: Path, annotations
) -> None:
    from lidark.dataset import load_dataset

    create_dataset(
        layout,
        AREA,
        FEATURES,
        "true.shp",
        None,
        img_size=IMG,
        dataset_id="splitcheck",
        seed=0,
    )
    ds = load_dataset(layout, "splitcheck")
    config = _config(val_split_mode="window")
    a = _split(ds, config)
    b = _split(ds, config)
    np.testing.assert_array_equal(a.train_indices, b.train_indices)
    n = len(ds.samples)
    assert len(a.train_indices) + len(a.val_indices) == n
    assert len(a.val_indices) == int(n * config.val_split)
    assert not set(a.train_indices.tolist()) & set(a.val_indices.tolist())
    assert a.meta["split"]["mode"] == "window"


def test_train_val_split_group_is_object_disjoint(
    layout, dtm_product: Path, tpi_layer: Path, annotations
) -> None:
    """Group mode (default): whole object groups reach one side, so no
    object appears in both halves; the split is seeded and its summary is
    recorded for the training log."""
    from lidark.dataset import load_dataset

    create_dataset(
        layout,
        AREA,
        FEATURES,
        "true.shp",
        "false.shp",
        img_size=IMG,
        dataset_id="groupcheck",
        seed=0,
    )
    ds = load_dataset(layout, "groupcheck")
    config = _config()  # group mode is the default
    a = _split(ds, config)
    b = _split(ds, config)
    np.testing.assert_array_equal(a.val_indices, b.val_indices)
    n = len(ds.samples)
    assert len(a.train_indices) + len(a.val_indices) == n
    assert len(a.val_indices) > 0
    assert not set(a.train_indices.tolist()) & set(a.val_indices.tolist())
    objects = ds.window_objects or []
    assert len(objects) == n
    train_objects = {objects[i] for i in a.train_indices}
    val_objects = {objects[i] for i in a.val_indices}
    assert not train_objects & val_objects
    # One spatial group per isolated object; the three objects around tiles
    # (0,0) and (1,0) sit within the window diagonal (hypot(16,16) * 2 m)
    # of each other and share a group.
    info = a.meta["split"]
    assert info["mode"] == "group"
    assert info["num_groups"] == 3
    assert info["cross_side_window_overlaps"] == 0
    assert info["num_train"] == len(a.train_indices)
    assert info["num_val"] == len(a.val_indices)


def test_train_val_split_group_keeps_clusters_together(
    layout, dtm_product: Path, tpi_layer: Path
) -> None:
    """Three objects a few metres apart (one cluster) plus a lone object:
    the cluster's windows all land on one side, the loner on the other."""
    from lidark.dataset import load_dataset

    cluster = [
        box(ORIGIN_X + 20.0, ORIGIN_Y - 20.0, ORIGIN_X + 28.0, ORIGIN_Y - 12.0),
        box(ORIGIN_X + 32.0, ORIGIN_Y - 24.0, ORIGIN_X + 40.0, ORIGIN_Y - 16.0),
        box(ORIGIN_X + 26.0, ORIGIN_Y - 36.0, ORIGIN_X + 34.0, ORIGIN_Y - 28.0),
    ]
    loner = box(ORIGIN_X + 84.0, ORIGIN_Y - 84.0, ORIGIN_X + 92.0, ORIGIN_Y - 76.0)
    write_annotation(layout, "cluster.shp", cluster + [loner], class_ids=[1, 1, 1, 1])
    create_dataset(
        layout,
        AREA,
        FEATURES,
        "cluster.shp",
        None,
        img_size=IMG,
        dataset_id="clustercheck",
        seed=0,
    )
    ds = load_dataset(layout, "clustercheck")
    data = _split(ds, _config(val_split=0.5))
    objects = ds.window_objects or []
    sides: dict[str, set[int]] = {}
    for i in range(len(ds.samples)):
        side = 0 if i in set(data.train_indices.tolist()) else 1
        sides.setdefault(objects[i], set()).add(side)
    # No object on both sides...
    assert all(len(side_set) == 1 for side_set in sides.values())
    # ...and the cluster shares one side, the loner the other.
    cluster_sides = {next(iter(sides[f"true_{k}"])) for k in range(3)}
    assert len(cluster_sides) == 1
    assert sides["true_3"] != cluster_sides
    assert len(data.val_indices) > 0 and len(data.train_indices) > 0


def test_train_val_split_group_stratifies_classes(
    layout, dtm_product: Path, tpi_layer: Path
) -> None:
    """One object per tile (four spatial groups), two classes: every class
    keeps windows on both sides and reaches its validation share."""
    from lidark.dataset import load_dataset

    w = TILE_SHAPE[1] * 2.0
    h = TILE_SHAPE[0] * 2.0
    geometry = []
    for row in range(2):
        for col in range(2):
            x0 = ORIGIN_X + col * w + 20.0
            y_top = ORIGIN_Y - row * h - 20.0
            geometry.append(box(x0, y_top - 8.0, x0 + 8.0, y_top))
    write_annotation(layout, "twoclass.shp", geometry, class_ids=[1, 1, 2, 2])
    create_dataset(
        layout,
        AREA,
        FEATURES,
        "twoclass.shp",
        None,
        img_size=IMG,
        dataset_id="stratgroup",
        seed=0,
    )
    ds = load_dataset(layout, "stratgroup")
    data = _split(ds, _config(val_split=0.5))
    classes = ds.window_classes or []
    val_classes = {classes[i] for i in data.val_indices}
    train_classes = {classes[i] for i in data.train_indices}
    assert val_classes == {1, 2}
    assert train_classes == {1, 2}
    # One group per class in validation: 4 of the 8 windows.
    assert len(data.val_indices) == 4


def test_train_val_split_val_area_holds_out_region(
    layout, dtm_product: Path, tpi_layer: Path, annotations
) -> None:
    """--val-area forces every window whose footprint intersects the polygon
    layer into validation (whole group), none into training."""
    from lidark.dataset import load_dataset

    write_annotation(
        layout,
        "valarea.gpkg",
        [box(ORIGIN_X + 64.0, ORIGIN_Y - 128.0, ORIGIN_X + 128.0, ORIGIN_Y - 64.0)],
    )
    create_dataset(
        layout,
        AREA,
        FEATURES,
        "true.shp",
        None,
        img_size=IMG,
        dataset_id="valareacheck",
        seed=0,
    )
    ds = load_dataset(layout, "valareacheck")
    data = _split(ds, _config(val_area="valarea.gpkg"), layout)
    objects = ds.window_objects or []
    val_objects = {objects[i] for i in data.val_indices}
    train_objects = {objects[i] for i in data.train_indices}
    # true_3 sits in tile (1,1), entirely inside the held-out region; its
    # windows (and only those, at a 0.2 split) are validation.
    assert val_objects == {"true_3"}
    assert "true_3" not in train_objects
    assert data.meta["split"]["val_area"] == "valarea.gpkg"


def test_train_val_split_per_class_fractions(layout, dtm_product: Path, tpi_layer: Path) -> None:
    """Per-class validation fractions: a class pinned to 0.0 keeps all its
    windows in training while the other class is split."""
    from lidark.dataset import load_dataset

    w = TILE_SHAPE[1] * 2.0
    h = TILE_SHAPE[0] * 2.0
    geometry = []
    for row in range(2):
        for col in range(2):
            x0 = ORIGIN_X + col * w + 20.0
            y_top = ORIGIN_Y - row * h - 20.0
            geometry.append(box(x0, y_top - 8.0, x0 + 8.0, y_top))
    write_annotation(layout, "twoclass.shp", geometry, class_ids=[1, 1, 2, 2])
    create_dataset(
        layout,
        AREA,
        FEATURES,
        "twoclass.shp",
        None,
        img_size=IMG,
        dataset_id="perclass",
        seed=0,
    )
    ds = load_dataset(layout, "perclass")
    data = _split(ds, _config(val_split=0.5, val_per_class=((1, 0.5), (2, 0.0))))
    classes = ds.window_classes or []
    assert not any(classes[i] == 2 for i in data.val_indices)
    assert any(classes[i] == 2 for i in data.train_indices)
    assert {classes[i] for i in data.val_indices} == {1}


def test_split_legacy_dataset_falls_back_to_window(caplog) -> None:
    """A dataset without object provenance (created before group splits)
    warns and falls back to the original window-level split."""
    meta = {"num_classes": 1, "window_classes": [1] * 8 + [0] * 2}
    samples = np.zeros((10, *IMG, 1), dtype=np.int8)
    labels = np.zeros((10, *IMG), dtype=np.uint8)
    dataset = Dataset("legacy", Path("."), samples, labels, meta)
    with caplog.at_level(logging.WARNING, logger="lidark.train"):
        data = _split(dataset, ModelConfig(val_split=0.2, seed=0))
    assert any("provenance" in record.message for record in caplog.records)
    rng = np.random.default_rng(0)
    expected = rng.permutation(10)
    assert np.array_equal(np.sort(data.val_indices), np.sort(expected[-2:]))
    assert data.meta["split"]["legacy_fallback"] is True


def test_train_multi_area(
    layout,
    dtm_product: Path,
    tpi_layer: Path,
    dtm_product2: Path,
    tpi_layer2: Path,
) -> None:
    """One model trained on several areas: windows from every area land in
    one dataset and one model id."""
    w = TILE_SHAPE[1] * 2.0
    h = TILE_SHAPE[0] * 2.0
    geometry = []
    for row in range(2):
        for col in range(2):
            x0 = ORIGIN_X + col * w + 20.0
            y_top = ORIGIN_Y - row * h - 20.0
            geometry.append(box(x0, y_top - 8.0, x0 + 8.0, y_top))
    # One true object strictly inside the second area's tile (0,0).
    geometry.append(
        box(
            AREA2_ORIGIN_X + 20.0,
            AREA2_ORIGIN_Y - 40.0,
            AREA2_ORIGIN_X + 28.0,
            AREA2_ORIGIN_Y - 32.0,
        )
    )
    write_annotation(layout, "true.shp", geometry, class_ids=[1] * 5)
    write_annotation(
        layout, "false.shp", [box(ORIGIN_X + 4, ORIGIN_Y - 60, ORIGIN_X + 12, ORIGIN_Y - 52)]
    )
    mid = train(
        layout,
        [AREA, AREA2],
        FEATURES,
        "true.shp",
        "false.shp",
        model_name="multi_area",
        model_config=_config(),
    )
    log = json.loads((layout.model_dir(mid) / "training_log.json").read_text(encoding="utf-8"))
    # 4 AREA true + 1 false + 1 AREA2 true objects, 2 windows each.
    assert log["num samples"] == 12
