"""Tests for lidark.config: layout conventions, EPSG default, model config."""

from pathlib import Path

from lidark.config import (
    DEFAULT_EPSG,
    ModelConfig,
    ProjectLayout,
    load_model_config,
    save_model_config,
)


def test_layout_resolves_lidar_paths(tmp_path: Path) -> None:
    layout = ProjectLayout(project_folder=tmp_path)
    assert layout.lidar_root == tmp_path / "Lidar"
    assert layout.lidar_area("Naljanka") == tmp_path / "Lidar" / "Naljanka"
    assert layout.originals_dir("Naljanka") == tmp_path / "Lidar" / "Naljanka" / "00_Original"
    assert layout.dtm_dir("Naljanka") == tmp_path / "Lidar" / "Naljanka" / "01_DTM"
    assert (
        layout.tpi_dir("Naljanka", 20) == tmp_path / "Lidar" / "Naljanka" / "02_TPI" / "radius_20"
    )


def test_layout_resolves_deep_learning_paths(tmp_path: Path) -> None:
    layout = ProjectLayout(project_folder=tmp_path)
    root = tmp_path / "DeepLearning"
    assert layout.deep_learning_root == root
    assert layout.train_annotations_dir == root / "train_annotations"
    assert layout.evaluate_annotations_dir("test1") == root / "evaluate_annotations" / "test1"
    assert layout.samples_dir == root / "samples"
    assert layout.labels_dir == root / "labels"
    assert layout.model_root == root / "model"
    assert layout.model_dir("20260101-120000") == root / "model" / "20260101-120000"
    assert (
        layout.predictions_dir("20260101-120000")
        == root / "model" / "20260101-120000" / "predictions"
    )
    assert (
        layout.prediction_raster_dir("20260101-120000", "Naljanka")
        == root / "model" / "20260101-120000" / "predictions" / "Naljanka" / "raster"
    )
    assert (
        layout.prediction_shapes_dir("20260101-120000", "Naljanka")
        == root / "model" / "20260101-120000" / "predictions" / "Naljanka" / "shapes"
    )
    assert layout.reports_root == root / "reports"
    assert layout.reports_dir("20260101-120000") == root / "reports" / "20260101-120000"


def test_default_epsg_lives_in_config() -> None:
    assert DEFAULT_EPSG == "EPSG:3067"
    assert ProjectLayout(project_folder=Path(".")).epsg == DEFAULT_EPSG


def test_epsg_override(tmp_path: Path) -> None:
    layout = ProjectLayout(project_folder=tmp_path, epsg="EPSG:3067")
    assert layout.epsg == "EPSG:3067"
    other = ProjectLayout(project_folder=tmp_path, epsg="EPSG:25832")
    assert other.epsg == "EPSG:25832"


def test_project_folder_accepts_str() -> None:
    layout = ProjectLayout(project_folder="C:/project")  # noqa: UP025 - deliberate
    assert isinstance(layout.project_folder, Path)


def test_model_config_defaults() -> None:
    config = ModelConfig()
    assert config.backend == "unet"
    assert config.num_classes == 1
    assert config.class_names is None
    assert config.img_size == (512, 512)
    assert config.learning_rate == 0.001
    assert config.batch_size == 32
    assert config.val_split == 0.2
    assert config.val_split_mode == "group"
    assert config.val_group_buffer is None
    assert config.val_area is None
    assert config.val_per_class is None
    assert config.max_epochs == 200
    assert config.unet_size == 16
    assert config.dropout == 0.0
    assert config.pred_stride == 128
    assert config.pred_batch_size == 32
    assert config.mixed_precision is False
    assert config.jit_compile is False


def test_model_config_val_split_round_trip(tmp_path: Path) -> None:
    config = ModelConfig(
        val_split=0.3,
        val_split_mode="window",
        val_group_buffer=250.0,
        val_area="holdout.gpkg",
        val_per_class=((1, 0.3), (2, 0.1)),
    )
    path = save_model_config(config, tmp_path / "model_config.json")
    loaded = load_model_config(path)
    assert loaded.val_split == 0.3
    assert loaded.val_split_mode == "window"
    assert loaded.val_group_buffer == 250.0
    assert loaded.val_area == "holdout.gpkg"
    assert loaded.val_per_class == ((1, 0.3), (2, 0.1))


def test_model_config_rejects_invalid_split_settings() -> None:
    for kwargs in (
        {"val_split_mode": "tile"},
        {"val_split": 1.0},
        {"val_split": -0.1},
        {"val_per_class": ((0, 0.5),)},
        {"val_per_class": ((1, 1.5),)},
    ):
        try:
            ModelConfig(**kwargs)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_model_config_class_name() -> None:
    config = ModelConfig()
    assert config.class_name(1) == "class_1"
    named = ModelConfig(num_classes=2, class_names=("tar_kiln", "pit"))
    assert named.class_name(1) == "tar_kiln"
    assert named.class_name(2) == "pit"
    try:
        named.class_name(3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range class id")
    try:
        config.class_name(0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for class_id < 1")


def test_model_config_max_samples_per_epoch() -> None:
    assert ModelConfig().max_samples_per_epoch is None
    config = ModelConfig(max_samples_per_epoch=100)
    assert config.max_samples_per_epoch == 100
    try:
        ModelConfig(max_samples_per_epoch=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for max_samples_per_epoch=0")
