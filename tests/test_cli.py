"""Tests for lidark.cli: parser, error handling, and the four commands e2e."""

from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import box
from tests.conftest import AREA, ORIGIN_X, ORIGIN_Y, TILE_SHAPE, write_annotation

from lidark.cli import build_parser, main

FEATURES = ["01_DTM", "02_TPI/radius_20"]


# --------------------------------------------------------------------- parser


def test_parser_train_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--project-folder",
            "x",
            "--area",
            "a",
            "--name",
            "n",
            "--true-objects",
            "t.shp",
            "--raster-type",
            "f1",
            "f2",
        ]
    )
    assert args.command == "train"
    assert args.name == "n"
    assert args.backend == "unet"
    assert args.max_epochs == 200
    assert args.batch_size == 32
    assert args.learning_rate == 0.001
    assert args.img_size == [512, 512]
    assert args.pred_stride == 128
    assert args.no_augment is False
    assert args.val_split == 0.2
    assert args.val_split_mode == "group"
    assert args.val_group_buffer is None
    assert args.val_area is None
    assert args.val_per_class is None


def test_parser_train_val_split_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--project-folder",
            "x",
            "--area",
            "a",
            "--name",
            "n",
            "--true-objects",
            "t.shp",
            "--raster-type",
            "f1",
            "--val-split-mode",
            "window",
            "--val-split",
            "0.3",
            "--val-group-buffer",
            "250",
            "--val-area",
            "holdout.gpkg",
            "--val-per-class",
            "1=0.3",
            "2=0.1",
        ]
    )
    assert args.val_split_mode == "window"
    assert args.val_split == 0.3
    assert args.val_group_buffer == 250.0
    assert args.val_area == "holdout.gpkg"
    assert args.val_per_class == ["1=0.3", "2=0.1"]
    from lidark.cli import _per_class_fractions

    assert _per_class_fractions(args.val_per_class) == ((1, 0.3), (2, 0.1))


def test_parser_train_multiple_areas() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--project-folder",
            "x",
            "--area",
            "a",
            "b",
            "--name",
            "n",
            "--true-objects",
            "t.shp",
            "--raster-type",
            "f1",
        ]
    )
    assert args.area == ["a", "b"]


def test_parser_predict_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--project-folder",
            "x",
            "--areas",
            "A",
            "B",
            "--model-timestamp",
            "ts",
            "--pred-type",
            "test",
            "--test-tile",
            "Q",
            "chunk",
            "--test-area",
            "ta.gpkg",
            "--write-tile-shapes",
        ]
    )
    assert args.areas == ["A", "B"]
    assert args.pred_type == "test"
    assert args.test_tile == ["Q", "chunk"]
    assert args.test_area == "ta.gpkg"
    assert args.write_tile_shapes is True


def test_parser_epsg_default_from_config() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["create-map", "--project-folder", "x", "--area", "a", "--map-type", "TPI"]
    )
    assert args.epsg == "EPSG:3067"


def test_cli_errors_return_nonzero(tmp_path: Path) -> None:
    # Bad map type (validated in the facade, reported by the CLI).
    rc = main(
        [
            "create-map",
            "--project-folder",
            str(tmp_path),
            "--area",
            "X",
            "--map-type",
            "MDHS",
        ]
    )
    assert rc == 1
    # Missing required arguments: argparse raises SystemExit(2).
    with pytest.raises(SystemExit) as exc:
        main(["create-map", "--project-folder", str(tmp_path)])
    assert exc.value.code == 2


# ------------------------------------------------------------------ e2e chain


@pytest.fixture()
def annotations(layout) -> None:
    """Training labels in every tile; evaluation labels in one corner."""
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
        layout,
        "false.shp",
        [box(ORIGIN_X + 4, ORIGIN_Y - 60, ORIGIN_X + 12, ORIGIN_Y - 52)],
    )


def test_cli_all_four_commands_end_to_end(
    layout,
    dtm_product: Path,
    tpi_layer: Path,
    annotations,
    capsys: pytest.CaptureFixture,
) -> None:
    folder = str(layout.project_folder)
    common = ["--project-folder", folder]

    # 1. create-map: TPI radius 4 from the DTM product.
    rc = main(
        [
            "create-map",
            *common,
            "--area",
            AREA,
            "--map-type",
            "TPI",
            "--buffer",
            "4",
            "--cores",
            "1",
        ]
    )
    assert rc == 0, capsys.readouterr().out
    assert (layout.tpi_dir(AREA, 4) / "tpi.vrt").exists()
    assert (layout.tpi_dir(AREA, 4) / "tiles.gpkg").exists()

    # 2. train.
    rc = main(
        [
            "train",
            *common,
            "--area",
            AREA,
            "--name",
            "cli_test",
            "--true-objects",
            "true.shp",
            "--false-objects",
            "false.shp",
            "--raster-type",
            *FEATURES,
            "--backend",
            "unet",
            "--max-epochs",
            "1",
            "--img-size",
            "16",
            "16",
            "--unet-size",
            "4",
            "--batch-size",
            "4",
            "--seed",
            "0",
            "--pred-stride",
            "8",
            "--pred-batch-size",
            "8",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "model trained:" in out
    timestamp = out.strip().splitlines()[-1].split(": ")[-1].strip()
    model_dir = layout.model_dir(timestamp)
    assert (model_dir / "model.pt").exists()
    assert (model_dir / "model_config.json").exists()
    assert (model_dir / "training_log.json").exists()

    # 3. predict (threshold 0 so the untrained model still detects).
    rc = main(
        [
            "predict",
            *common,
            "--areas",
            AREA,
            "--model-id",
            timestamp,
            "--pred-type",
            "all",
            "--threshold",
            "0.0",
        ]
    )
    assert rc == 0
    merged = layout.predictions_dir(timestamp, AREA) / "merged_shapes_all.gpkg"
    assert merged.exists()
    assert len(list(layout.prediction_raster_dir(timestamp, AREA).glob("*.tif"))) == 4

    # 4. evaluate.
    eval_dir = layout.evaluate_annotations_dir("smoke")
    eval_dir.mkdir(parents=True)
    import geopandas as gpd

    from lidark.config import DEFAULT_EPSG

    gpd.GeoDataFrame(
        geometry=[box(ORIGIN_X + 18.0, ORIGIN_Y - 28.0, ORIGIN_X + 30.0, ORIGIN_Y - 12.0)],
        crs=DEFAULT_EPSG,
    ).to_file(eval_dir / "evaluate_true.shp")
    rc = main(
        [
            "evaluate",
            *common,
            "--model-id",
            timestamp,
            "--eval-name",
            "smoke",
            "--true-objects",
            "evaluate_true.shp",
            "--prob-cut",
            "0.0",
            "--size-cut",
            "0.0",
        ]
    )
    assert rc == 0
    report = layout.reports_dir(timestamp) / "smoke.csv"
    assert report.exists()
    assert "report written:" in capsys.readouterr().out


def test_cli_train_missing_args_returns_one(tmp_path: Path) -> None:
    rc = main(
        [
            "train",
            "--project-folder",
            str(tmp_path),
            "--area",
            "X",
            "--name",
            "n",
        ]
    )
    assert rc == 1


def test_cli_predict_missing_model_returns_one(layout, tmp_path: Path) -> None:
    rc = main(
        [
            "predict",
            "--project-folder",
            str(layout.project_folder),
            "--areas",
            AREA,
            "--model-id",
            "19990101-000000",
        ]
    )
    assert rc == 1


# --------------------------------------------------------- project folder cfg


def _isolate_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the settings store at a temporary home; returns the store dir."""
    from lidark import user_config

    home = tmp_path / "home"
    monkeypatch.setattr(user_config, "CONFIG_DIR", home)
    return home


def test_parser_project_folder_is_optional() -> None:
    parser = build_parser()
    args = parser.parse_args(["create-map", "--area", "a", "--map-type", "TPI"])
    assert args.project_folder is None


def test_user_config_project_folder_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lidark import user_config

    _isolate_user_config(tmp_path, monkeypatch)
    assert user_config.get_project_folder() is None
    stored = user_config.set_project_folder(tmp_path)
    assert user_config.get_project_folder() == stored
    assert stored == tmp_path.resolve()


def test_set_project_folder_rejects_missing_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lidark import user_config

    _isolate_user_config(tmp_path, monkeypatch)
    with pytest.raises(NotADirectoryError, match="does not exist"):
        user_config.set_project_folder(tmp_path / "nope")


def test_cli_set_project_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from lidark import user_config

    _isolate_user_config(tmp_path, monkeypatch)
    rc = main(["set-project-folder", str(tmp_path)])
    assert rc == 0
    assert "project folder set" in capsys.readouterr().out
    assert user_config.get_project_folder() == tmp_path.resolve()


def test_cli_set_project_folder_missing_dir_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_user_config(tmp_path, monkeypatch)
    rc = main(["set-project-folder", str(tmp_path / "nope")])
    assert rc == 1


def _capture_create_map_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Route create-map at a stub facade; returns the seen arguments."""
    import lidark.maps.api as maps_api

    seen: dict[str, object] = {}

    def _fake_create_map(project_folder, area, map_type, **kwargs):  # noqa: ARG001
        seen["folder"] = project_folder
        return []

    monkeypatch.setattr(maps_api, "create_map", _fake_create_map)
    return seen


def test_cli_uses_stored_project_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lidark import user_config

    _isolate_user_config(tmp_path, monkeypatch)
    user_config.set_project_folder(tmp_path)
    seen = _capture_create_map_folder(monkeypatch)
    rc = main(["create-map", "--area", "a", "--map-type", "DTM"])
    assert rc == 0
    assert seen["folder"] == tmp_path.resolve()


def test_cli_project_folder_flag_overrides_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lidark import user_config

    _isolate_user_config(tmp_path, monkeypatch)
    user_config.set_project_folder(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    seen = _capture_create_map_folder(monkeypatch)
    rc = main(["create-map", "--area", "a", "--map-type", "DTM", "--project-folder", str(other)])
    assert rc == 0
    # The stored folder is untouched by the one-run override.
    assert seen["folder"] == other
    assert user_config.get_project_folder() == tmp_path.resolve()


def test_cli_without_any_project_folder_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_user_config(tmp_path, monkeypatch)
    rc = main(["create-map", "--area", "a", "--map-type", "DTM"])
    assert rc == 1
