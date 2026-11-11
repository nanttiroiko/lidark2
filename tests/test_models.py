"""Tests for the models package: registry, protocol, U-Net, IoU, backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lidark.config import TASK_SEMANTIC_SEGMENTATION, ModelConfig, load_model_config
from lidark.models import create_backend, known_backends
from lidark.models.base import ModelBackend, TrainingData
from lidark.models.registry import register

FEATURES = ("01_DTM", "02_TPI/radius_20")


def _tiny_config(**overrides: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "feature_list": FEATURES,
        "img_size": (16, 16),
        "unet_size": 4,
        "batch_size": 4,
        "max_epochs": 1,
        "seed": 0,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def _tiny_training_data(num_classes: int = 1, n: int = 8) -> TrainingData:
    rng = np.random.default_rng(1)
    samples = rng.uniform(-100, 100, size=(n, 16, 16, 2)).astype(np.int8)
    labels = np.zeros((n, 16, 16), dtype=np.uint8)
    if num_classes == 1:
        labels[:, 4:8, 4:8] = 1
    else:
        labels[:, 4:6, 4:6] = 1
        labels[:, 6:8, 6:8] = min(num_classes, 2)
    indices = np.arange(n)
    return TrainingData(
        samples=samples,
        labels=labels,
        train_indices=indices[:6],
        val_indices=indices[6:],
        num_classes=num_classes,
        meta={"dataset_id": "test", "num_classes": num_classes},
    )


# --------------------------------------------------------------------- registry


def test_known_backends_includes_unet() -> None:
    assert "unet" in known_backends()


def test_known_backends_includes_maskrcnn() -> None:
    # The maskrcnn factory imports torch lazily, so listing and creating the
    # backend work without lidark[torch] installed.
    assert "maskrcnn" in known_backends()


def test_known_backends_includes_fasterrcnn() -> None:
    # The fasterrcnn factory imports torch lazily, so listing and creating
    # the backend work without lidark[torch] installed.
    assert "fasterrcnn" in known_backends()


def test_create_backend_resolves_default() -> None:
    backend = create_backend("unet")
    assert backend.task == TASK_SEMANTIC_SEGMENTATION
    assert isinstance(backend, ModelBackend)  # runtime protocol conformance


def test_create_backend_resolves_maskrcnn() -> None:
    from lidark.config import TASK_INSTANCE_SEGMENTATION

    backend = create_backend("maskrcnn")
    assert backend.task == TASK_INSTANCE_SEGMENTATION
    assert isinstance(backend, ModelBackend)  # runtime protocol conformance


def test_create_backend_resolves_fasterrcnn() -> None:
    from lidark.config import TASK_INSTANCE_DETECTION

    backend = create_backend("fasterrcnn")
    assert backend.task == TASK_INSTANCE_DETECTION
    assert isinstance(backend, ModelBackend)  # runtime protocol conformance


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="known backends"):
        create_backend("yolo")


def test_register_custom_backend() -> None:
    class _Fake:
        task = TASK_SEMANTIC_SEGMENTATION

        def build(self, config: ModelConfig) -> None:
            return None

    register("fake", _Fake)
    assert "fake" in known_backends()
    assert isinstance(create_backend("fake"), _Fake)


# ------------------------------------------------------------------ architecture

torch = pytest.importorskip("torch", reason="torch not installed")


def test_unet_topology_matches_original() -> None:
    from lidark.models.unet.architecture import get_unet_model

    model = get_unet_model((64, 64, 2), 1, "relu", 0.0, 8)
    out = model(torch.zeros(1, 2, 64, 64))
    assert out.shape == (1, 1, 64, 64)  # logits, channels-first
    # 4 Add-skip decoders (upscale convolutions) and 4 pooling encoders.
    for name in ("decoder1", "decoder2", "decoder3", "decoder4"):
        block = getattr(model, name)
        assert isinstance(block.conv_up, torch.nn.ConvTranspose2d)
        assert block.conv_up.stride == (2, 2)
    for name in ("encoder1", "encoder2", "encoder3", "encoder4"):
        assert isinstance(getattr(model, name).down_pool, torch.nn.MaxPool2d)
    # Original block naming: encoder1_c1_conv / b1_c2_conv / decoder4_conv_up.
    named = dict(model.named_modules())
    assert "encoder1.block.c1.conv" in named
    assert "b1.c2.conv" in named
    assert "decoder4.conv_up" in named
    # Conv2D(use_bias=False) parity: the BN learns the bias.
    assert model.encoder1.block.c1.conv.bias is None
    assert model.head.kernel_size == (1, 1)


def test_unet_multiclass_head() -> None:
    from lidark.models.unet.architecture import get_unet_model

    model = get_unet_model((64, 64, 2), 3, "relu", 0.0, 8)
    out = model(torch.zeros(1, 2, 64, 64))
    assert out.shape == (1, 3, 64, 64)  # C+1 channels


# ----------------------------------------------------------------------- metrics


def test_iou_perfect_binary_prediction() -> None:
    from lidark.models.unet.metrics import IoU

    metric = IoU(num_classes=1)
    labels = np.zeros((2, 8, 8), dtype=np.int64)
    labels[:, 2:4, 2:4] = 1
    logits = torch.full((2, 1, 8, 8), -20.0)
    logits[:, 0, 2:4, 2:4] = 20.0
    metric.update(logits, labels)
    assert metric.result() == pytest.approx(1.0)


def test_iou_multiclass_perfect_prediction() -> None:
    from lidark.models.unet.metrics import IoU

    metric = IoU(num_classes=2)
    labels = np.zeros((1, 8, 8), dtype=np.int64)
    labels[0, 1:3, 1:3] = 1
    labels[0, 5:7, 5:7] = 2
    logits = torch.full((1, 3, 8, 8), -20.0)
    logits[0, 1, 1:3, 1:3] = 20.0
    logits[0, 2, 5:7, 5:7] = 20.0
    metric.update(logits, labels)
    assert metric.result() == pytest.approx(1.0)


def test_iou_empty_union_scores_one() -> None:
    from lidark.models.unet.metrics import IoU

    metric = IoU(num_classes=1)
    metric.update(torch.full((1, 1, 4, 4), -20.0), np.zeros((1, 4, 4), dtype=np.int64))
    assert metric.result() == pytest.approx(1.0)


def test_iou_accumulates_across_batches_and_resets() -> None:
    from lidark.models.unet.metrics import IoU

    metric = IoU(num_classes=1)
    # Batch 1: TP=1, FN=1 -> intersect=1, union=2.
    labels1 = np.ones((1, 2, 1), dtype=np.int64)
    logits1 = torch.tensor([[[[20.0], [-20.0]]]])  # only the first pixel predicted
    metric.update(logits1, labels1)
    assert metric.result() == pytest.approx(0.5)
    # Batch 2: a perfect single pixel -> accumulated 2/3.
    metric.update(torch.full((1, 1, 1, 1), 20.0), np.ones((1, 1, 1), dtype=np.int64))
    assert metric.result() == pytest.approx(2.0 / 3.0)
    metric.reset()
    assert metric.result() == pytest.approx(1.0)


# ----------------------------------------------------------------------- backend


def test_backend_build_and_predict_shapes() -> None:
    backend = create_backend("unet")
    backend.build(_tiny_config())
    tiles = np.zeros((2, 16, 16, 2), dtype=np.int8)
    scores = backend.predict_tiles(tiles)
    assert scores.shape == (2, 16, 16)
    assert scores.dtype == np.float32
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_backend_multiclass_predict_shape() -> None:
    backend = create_backend("unet")
    backend.build(_tiny_config(num_classes=2))
    tiles = np.zeros((2, 16, 16, 2), dtype=np.int8)
    scores = backend.predict_tiles(tiles)
    assert scores.shape == (2, 16, 16, 2)  # per-class maps, background dropped


def test_backend_batched_equals_single_window() -> None:
    backend = create_backend("unet")
    backend.build(_tiny_config())
    rng = np.random.default_rng(5)
    tiles = rng.uniform(-100, 100, size=(5, 16, 16, 2)).astype(np.int8)
    batched = backend.predict_tiles(tiles)
    for i in range(5):
        single = backend.predict_tiles(tiles[i : i + 1])
        # float32 convolutions are batch-order sensitive in the last ulps
        np.testing.assert_allclose(batched[i], single[0], atol=1e-5)


def test_backend_build_requires_feature_list() -> None:
    backend = create_backend("unet")
    with pytest.raises(ValueError, match="feature_list"):
        backend.build(ModelConfig(img_size=(16, 16), unet_size=4))


def test_backend_trains_one_epoch(tmp_path: Path) -> None:
    backend = create_backend("unet")
    backend.build(_tiny_config())
    history = backend.fit(_tiny_training_data(num_classes=1), tmp_path, augment=True)
    assert len(history["loss"]) >= 1
    assert "iou" in history
    assert len(history["imgs_per_second"]) == len(history["loss"])
    # Best checkpoint written.
    assert (tmp_path / "model.pt").exists()


def test_backend_save_load_round_trip(tmp_path: Path) -> None:
    backend = create_backend("unet")
    config = _tiny_config()
    backend.build(config)
    backend.fit(_tiny_training_data(), tmp_path, augment=False)
    backend.save(tmp_path)

    assert (tmp_path / "model_config.json").exists()
    loaded_config = load_model_config(tmp_path / "model_config.json")
    assert loaded_config == config

    restored = create_backend("unet")
    restored.load(tmp_path)
    assert restored.task == TASK_SEMANTIC_SEGMENTATION
    tiles = np.zeros((1, 16, 16, 2), dtype=np.int8)
    assert restored.predict_tiles(tiles).shape == (1, 16, 16)


def test_backend_predict_requires_build() -> None:
    backend = create_backend("unet")
    with pytest.raises(RuntimeError, match="build"):
        backend.predict_tiles(np.zeros((1, 16, 16, 2), dtype=np.int8))


def test_unet_fit_caps_training_windows_per_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_samples_per_epoch`` caps the training windows per epoch: a
    fresh seeded subset each epoch, validation never capped."""
    import lidark.models.unet.backend as unet_module

    processed: list[int] = []
    original = unet_module._batch_to_torch

    def counting(samples: object, labels: object, batch: object, augment: object, device: object):
        processed.append(len(batch))  # type: ignore[arg-type]
        return original(samples, labels, batch, augment, device)

    monkeypatch.setattr(unet_module, "_batch_to_torch", counting)

    data = _tiny_training_data(n=8)  # 6 train + 2 val windows
    backend = create_backend("unet")
    backend.build(_tiny_config(max_samples_per_epoch=4, batch_size=2, max_epochs=2))
    history = backend.fit(data, tmp_path)

    # Per epoch: 4 (capped) training windows + 2 uncapped validation windows.
    assert sum(processed) == (4 + 2) * 2
    assert len(history["loss"]) == 2

    # A cap larger than the training set is a no-op.
    processed.clear()
    backend = create_backend("unet")
    backend.build(_tiny_config(max_samples_per_epoch=100, batch_size=2, max_epochs=1))
    backend.fit(data, tmp_path)
    assert sum(processed) == 6 + 2
