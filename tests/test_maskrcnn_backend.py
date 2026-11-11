"""Tests for the Mask R-CNN backend (require torch + torchvision; skip otherwise)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lidark.config import TASK_INSTANCE_SEGMENTATION, load_model_config
from lidark.models.base import ModelBackend, TrainingData
from lidark.models.maskrcnn.backend import WEIGHTS_FILENAME, MaskRcnnBackend

pytest.importorskip("torch", reason="torch not installed (lidark[torch])")
pytest.importorskip("torchvision", reason="torchvision not installed (lidark[torch])")

FEATURES = ("01_DTM", "02_TPI/radius_20")


def _tiny_config(**overrides: object):
    from lidark.config import ModelConfig

    defaults: dict[str, object] = {
        "feature_list": FEATURES,
        "img_size": (16, 16),
        "batch_size": 2,
        "max_epochs": 1,
        "seed": 0,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def _tiny_training_data(num_classes: int = 1, n: int = 6) -> TrainingData:
    rng = np.random.default_rng(1)
    samples = rng.uniform(-100, 100, size=(n, 16, 16, 2)).astype(np.int8)
    labels = np.zeros((n, 16, 16), dtype=np.uint8)
    labels[:, 4:8, 4:8] = 1
    if num_classes > 1:
        labels[:, 10:12, 10:12] = 2
    return TrainingData(
        samples=samples,
        labels=labels,
        train_indices=np.arange(n - 2),
        val_indices=np.arange(n - 2, n),
        num_classes=num_classes,
        meta={"dataset_id": "test", "num_classes": num_classes},
    )


def test_build_matches_config() -> None:
    backend = MaskRcnnBackend()
    config = _tiny_config()
    backend.build(config)
    assert backend.task == TASK_INSTANCE_SEGMENTATION
    assert isinstance(backend, ModelBackend)  # runtime protocol conformance
    assert backend._model is not None


def test_build_requires_feature_list() -> None:
    backend = MaskRcnnBackend()
    with pytest.raises(ValueError, match="feature_list"):
        backend.build(_tiny_config(feature_list=()))


def test_fit_writes_checkpoint_and_history(tmp_path: Path) -> None:
    backend = MaskRcnnBackend()
    backend.build(_tiny_config())
    history = backend.fit(_tiny_training_data(), tmp_path, augment=False)
    assert (tmp_path / WEIGHTS_FILENAME).exists()
    assert history["loss"]
    # Mask mAP validation series (AP50 and AP75 both reported).
    assert len(history["val_map50"]) == len(history["loss"])
    assert len(history["val_map75"]) == len(history["loss"])
    assert len(history["val_map"]) == len(history["loss"])
    # The monitored metric follows the configured IoU threshold.
    assert history["val_map"][0] == history["val_map50"][0]
    assert history["imgs_per_second"]


def test_fit_monitored_metric_is_ap75_when_configured(tmp_path: Path) -> None:
    backend = MaskRcnnBackend()
    backend.build(_tiny_config(mask_ap_iou=0.75))
    history = backend.fit(_tiny_training_data(), tmp_path, augment=False)
    assert history["val_map"][0] == history["val_map75"][0]


def test_save_load_round_trip(tmp_path: Path) -> None:
    from lidark.models.base import CONFIG_FILENAME

    backend = MaskRcnnBackend()
    backend.build(_tiny_config())
    backend.fit(_tiny_training_data(), tmp_path, augment=False)
    backend.save(tmp_path)
    assert (tmp_path / CONFIG_FILENAME).exists()

    loaded = MaskRcnnBackend()
    loaded.load(tmp_path)
    assert loaded._config is not None
    assert loaded._config.feature_list == tuple(FEATURES)
    # Inference works after the round trip.
    windows = np.zeros((2, 16, 16, 2), dtype=np.float32)
    results = loaded.predict_instances(windows)
    assert len(results) == 2


def test_load_missing_weights_raises(tmp_path: Path) -> None:
    from lidark.models.base import CONFIG_FILENAME

    backend = MaskRcnnBackend()
    backend.build(_tiny_config())
    backend.save(tmp_path)
    assert (tmp_path / CONFIG_FILENAME).exists()
    with pytest.raises(FileNotFoundError, match="weights"):
        MaskRcnnBackend().load(tmp_path)


def test_predict_instances_schema() -> None:
    backend = MaskRcnnBackend()
    backend.build(_tiny_config())
    windows = np.zeros((2, 16, 16, 2), dtype=np.float32)
    results = backend.predict_instances(windows)
    assert len(results) == 2
    for instances in results:
        for instance in instances:
            assert set(instance) == {"mask", "score", "class_id"}
            assert instance["mask"].shape == (16, 16)
            assert instance["mask"].dtype == np.uint8
            assert 0.0 <= instance["score"] <= 1.0
            assert instance["class_id"] >= 1


def test_predict_tiles_not_available() -> None:
    backend = MaskRcnnBackend()
    backend.build(_tiny_config())
    with pytest.raises(NotImplementedError, match="predict_instances"):
        backend.predict_tiles(np.zeros((1, 16, 16, 2), dtype=np.float32))


def test_model_config_records_mask_ap_iou(tmp_path: Path) -> None:
    from lidark.models.base import CONFIG_FILENAME

    backend = MaskRcnnBackend()
    backend.build(_tiny_config(mask_ap_iou=0.75))
    backend.save(tmp_path)
    config = load_model_config(tmp_path / CONFIG_FILENAME)
    assert config.mask_ap_iou == 0.75


def test_fit_caps_training_windows_per_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_samples_per_epoch`` caps the training batches per epoch
    (validation loops over windows separately and is never capped)."""
    import lidark.models.maskrcnn.backend as maskrcnn_module

    processed: list[int] = []
    original = maskrcnn_module._batch_to_torch

    def counting(*args: object, **kwargs: object) -> object:
        processed.append(len(args[2]))  # third positional arg is the batch
        return original(*args, **kwargs)

    monkeypatch.setattr(maskrcnn_module, "_batch_to_torch", counting)

    # _tiny_training_data(): 4 training + 2 validation windows.
    backend = MaskRcnnBackend()
    backend.build(_tiny_config(max_samples_per_epoch=2, batch_size=2, max_epochs=1))
    history = backend.fit(_tiny_training_data(), tmp_path, augment=False)
    assert sum(processed) == 2  # the capped training batches
    assert history["loss"]

    # Uncapped run: all 4 training windows.
    processed.clear()
    backend = MaskRcnnBackend()
    backend.build(_tiny_config(batch_size=2, max_epochs=1))
    backend.fit(_tiny_training_data(), tmp_path, augment=False)
    assert sum(processed) == 4
