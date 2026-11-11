"""Fine-tuning / warm-start from an existing model (plan 3.2).

Covers the plan's acceptance tests: a warm-started run's epoch-1 loss is
below a scratch run's on the same dataset; the base model's files are
unchanged afterwards; the compatibility guards raise the documented
errors; and the new model records ``base_model_id`` provenance in its
config and training log, with this run's hyperparameters (not the base's).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box
from tests.conftest import (
    AREA,
    ORIGIN_X,
    ORIGIN_Y,
    TILE_SHAPE,
    tile_array,
    tile_transform,
    write_annotation,
)

from lidark.config import DEFAULT_EPSG, ModelConfig, ProjectLayout, save_model_config
from lidark.dataset import create_dataset
from lidark.models.base import CONFIG_FILENAME
from lidark.train import LOG_FILENAME, train

FEATURES = ("01_DTM", "02_TPI/radius_20")
IMG = (16, 16)
FINE_TUNE_LR = 0.0001


def _tiny_config(**overrides: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "feature_list": FEATURES,
        "img_size": IMG,
        "unet_size": 4,
        "batch_size": 4,
        "max_epochs": 1,
        "seed": 0,
        "pred_stride": 8,
        "pred_batch_size": 8,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def _history(layout: ProjectLayout, model_id: str) -> dict:
    log = json.loads((layout.model_dir(model_id) / LOG_FILENAME).read_text(encoding="utf-8"))
    return log["history"]


def _dir_digest(path: Path) -> str:
    """Stable digest of a model directory: names, order, and contents."""
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            digest.update(str(file.relative_to(path)).encode("utf-8"))
            digest.update(file.read_bytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def layout_module(tmp_path_factory) -> ProjectLayout:
    """Module-scoped layout + feature products shared by the fine-tune tests."""
    from tests.conftest import TILE_NAMES

    from lidark.io.rasters import write_band
    from lidark.io.tiles import LAYER_NAME, build_tile_index, load_tile_index, tile_index_path
    from lidark.io.vrt import build_vrt

    layout = ProjectLayout(
        project_folder=tmp_path_factory.mktemp("finetune_project"), epsg=DEFAULT_EPSG
    )
    dtm_dir = layout.dtm_dir(AREA)
    dtm_dir.mkdir(parents=True)
    paths: list[Path] = []
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
    index = load_tile_index(dtm_dir).copy()
    index["path"] = [str(paths_by_id[tile_id]) for tile_id in index["tile_id"]]
    index.to_file(tile_index_path(feature_dir), layer=LAYER_NAME)
    return layout


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
def dataset_id(layout_module, annotations) -> str:  # noqa: ARG001
    """One dataset shared by the scratch and fine-tuned runs."""
    ds = create_dataset(
        layout_module,
        AREA,
        list(FEATURES),
        "true.shp",
        "false.shp",
        img_size=IMG,
        dataset_id="finetune_d1",
        seed=0,
    )
    return ds.id


@pytest.fixture(scope="module")
def base_model_id(layout_module, dataset_id: str) -> str:
    """The scratch-trained base model (plan 3.2's starting point)."""
    return train(
        layout_module,
        AREA,
        dataset_id=dataset_id,
        model_name="base_v1",
        model_config=_tiny_config(),
    )


@pytest.fixture(scope="module")
def tuned_model_id(layout_module, dataset_id: str, base_model_id: str) -> str:
    """A fine-tuned run warm-started from the base model, with this run's
    (lower) learning rate rather than the base's."""
    return train(
        layout_module,
        AREA,
        dataset_id=dataset_id,
        model_name="tuned_v2",
        model_config=_tiny_config(learning_rate=FINE_TUNE_LR),
        base_model_id=base_model_id,
    )


def test_warm_start_epoch1_loss_below_scratch(
    layout_module, base_model_id: str, tuned_model_id: str
) -> None:
    """Epoch-1 loss of the warm-started run is below the scratch run's on
    the same dataset (the plan's acceptance test)."""
    scratch_loss = _history(layout_module, base_model_id)["loss"][0]
    tuned_loss = _history(layout_module, tuned_model_id)["loss"][0]
    assert tuned_loss < scratch_loss


def test_base_model_unchanged_after_fine_tune(layout_module, base_model_id: str) -> None:
    """The base model's files are byte-identical after a fine-tuned run."""
    before = _dir_digest(layout_module.model_dir(base_model_id))
    train(
        layout_module,
        AREA,
        dataset_id=_dataset_of(layout_module, base_model_id),
        model_name="tuned_check",
        model_config=_tiny_config(),
        base_model_id=base_model_id,
    )
    assert _dir_digest(layout_module.model_dir(base_model_id)) == before


def _dataset_of(layout: ProjectLayout, model_id: str) -> str:
    log = json.loads((layout.model_dir(model_id) / LOG_FILENAME).read_text(encoding="utf-8"))
    return str(log["data_id"])


def test_fine_tune_provenance_and_hyperparameters(
    layout_module, base_model_id: str, tuned_model_id: str
) -> None:
    """The new model gets a fresh id, records ``base_model_id`` in its
    config and log, and keeps **this run's** hyperparameters (the lower
    fine-tune learning rate, not the base's)."""
    assert tuned_model_id != base_model_id
    assert layout_module.model_dir(tuned_model_id) != layout_module.model_dir(base_model_id)

    from lidark.config import load_model_config

    config = load_model_config(layout_module.model_dir(tuned_model_id) / CONFIG_FILENAME)
    assert config.base_model_id == base_model_id
    assert config.learning_rate == FINE_TUNE_LR  # the new run's, not the base's

    log = json.loads(
        (layout_module.model_dir(tuned_model_id) / LOG_FILENAME).read_text(encoding="utf-8")
    )
    assert log["base model"] == base_model_id
    assert log["learning rate"] == FINE_TUNE_LR

    # The scratch-trained base records no lineage.
    base_config = load_model_config(layout_module.model_dir(base_model_id) / CONFIG_FILENAME)
    assert base_config.base_model_id is None


def _write_base_config(layout: ProjectLayout, model_id: str, **overrides: object) -> str:
    """A base model dir whose config mismatches this run in one field."""
    model_dir = layout.model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_model_config(_tiny_config(**overrides), model_dir / CONFIG_FILENAME)
    return model_id


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"backend": "maskrcnn"}, "backend"),
        ({"img_size": (32, 32)}, "img_size"),
        ({"feature_list": ("01_DTM",)}, "feature_list length"),
        ({"num_classes": 2}, "num_classes"),
        ({"unet_size": 8}, "unet_size"),
    ],
)
def test_warm_start_guards_raise(
    layout_module, dataset_id: str, overrides: dict, field: str
) -> None:
    """Each incompatible base model raises the documented guard error."""
    bad_base = _write_base_config(layout_module, "bad_base", **overrides)
    with pytest.raises(ValueError, match=field):
        train(
            layout_module,
            AREA,
            dataset_id=dataset_id,
            model_name="guarded",
            model_config=_tiny_config(),
            base_model_id=bad_base,
        )


def test_warm_start_missing_base_model_raises(layout_module, dataset_id: str) -> None:
    with pytest.raises(FileNotFoundError, match="no_such_model"):
        train(
            layout_module,
            AREA,
            dataset_id=dataset_id,
            model_name="guarded",
            model_config=_tiny_config(),
            base_model_id="no_such_model",
        )


def test_cli_base_model_flag() -> None:
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
            "--base-model",
            "base_v1_20260101-120000",
        ]
    )
    assert args.base_model == "base_v1_20260101-120000"
