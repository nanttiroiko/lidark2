"""Mask R-CNN instance segmentation backend (torchvision, via ``lidark[torch]``).

Implements the :class:`~lidark.models.base.ModelBackend` protocol with
``task = instance_segmentation``. The model is torchvision's
``maskrcnn_resnet50_fpn`` (no pretrained weights — nothing is downloaded)
with the stem convolution replaced for ``len(feature_list)`` input channels
and the internal transform pinned to the training window size.

Instance targets derive from the dataset's per-pixel class maps
(:func:`~lidark.models.maskrcnn.metrics.instances_from_label`:
4-connected components; touching same-class objects merge, because the class
maps carry no instance identity). Training validates with **mask mAP** —
AP50 by default, AP75 via ``ModelConfig.mask_ap_iou`` (CLI
``--mask-ap-iou``) — and the best ``val_map`` checkpoint is written to
``<model_dir>/model.pt``.

All torch/torchvision imports are lazy: registering or merely listing the
backend never requires the framework; only ``build``/``fit``/inference do.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from tqdm import tqdm

from lidark.config import (
    TASK_INSTANCE_SEGMENTATION,
    ModelConfig,
    load_model_config,
    save_model_config,
)
from lidark.logging import get_logger
from lidark.models.base import CONFIG_FILENAME, TrainingData
from lidark.models.maskrcnn.metrics import (
    AP50,
    AP75,
    Instance,
    instances_from_label,
    mask_average_precision,
    mask_box,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lidark._types import Array

LOGGER = get_logger("models.maskrcnn")

# Torch-native weights file inside the model directory.
WEIGHTS_FILENAME = "model.pt"


class MaskRcnnBackend:
    """torchvision Mask R-CNN implementing the ModelBackend protocol."""

    task = TASK_INSTANCE_SEGMENTATION

    def __init__(self) -> None:
        self._config: ModelConfig | None = None
        self._model: Any = None

    # ------------------------------------------------------------------ build

    def build(self, config: ModelConfig) -> None:
        """Construct Mask R-CNN for the config's classes and input layers."""
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision.models.detection.transform import GeneralizedRCNNTransform

        if config.num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {config.num_classes}")
        if not config.feature_list:
            raise ValueError("ModelConfig.feature_list must name the input layers")
        height, width = config.img_size
        channels = len(config.feature_list)
        # No pretrained weights: the ImageNet backbone would be downloaded
        # only for its 3-channel stem to be replaced anyway.
        model = maskrcnn_resnet50_fpn(
            num_classes=config.num_classes + 1,  # + background
            weights=None,
            weights_backbone=None,
        )
        model.backbone.body["conv1"] = torch.nn.Conv2d(
            channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Transform pinned to the window size with identity normalization
        # (the feature layers keep their physical scale).
        model.transform = GeneralizedRCNNTransform(
            min_size=(min(height, width),),
            max_size=max(height, width),
            image_mean=[0.0] * channels,
            image_std=[1.0] * channels,
        )
        self._config = config
        self._model = model

    # ------------------------------------------------------------------- fit

    def fit(
        self, data: TrainingData, model_dir: Path, *, augment: bool = True
    ) -> dict[str, Sequence[float]]:
        """Train Mask R-CNN; validate with mask mAP (AP50 and AP75).

        The monitored metric is the mask mAP at ``ModelConfig.mask_ap_iou``
        (AP50 default, AP75 optional); the best checkpoint (``model.pt``)
        and early stopping (after ``ModelConfig.early_stop_patience`` stale
        epochs, default 10) follow it, mirroring the U-Net backend's
        ``val_iou`` behaviour. Returns the metric history.
        """
        import torch

        if self._model is None or self._config is None:
            raise RuntimeError("call build() before fit()")
        config = self._config
        torch.manual_seed(config.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self._model.to(device)
        scaler: Any = None
        if config.mixed_precision and device == "cuda":
            scaler = torch.amp.GradScaler("cuda")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        samples, labels = data.samples, data.labels
        train_indices = list(np.asarray(data.train_indices))
        val_indices = list(np.asarray(data.val_indices))
        monitor = "val_map" if val_indices else "loss"
        maximize = monitor == "val_map"
        history: dict[str, list[float]] = {
            "loss": [],
            "val_map50": [],
            "val_map75": [],
            "val_map": [],
            "imgs_per_second": [],
        }
        best_metric = -float("inf")
        stale = 0

        model_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(config.max_epochs):
            start = time.perf_counter()
            epoch_loss = self._train_epoch(
                model,
                optimizer,
                scaler,
                samples,
                labels,
                train_indices,
                config,
                epoch,
                augment,
                device,
            )
            history["loss"].append(epoch_loss)
            history["imgs_per_second"].append(
                max(len(train_indices), 1) / max(time.perf_counter() - start, 1e-9)
            )
            if val_indices:
                map50, map75 = self._validate(
                    model, samples, labels, val_indices, config, epoch, device
                )
                monitored = map50 if config.mask_ap_iou <= AP50 else map75
                history["val_map50"].append(map50)
                history["val_map75"].append(map75)
                history["val_map"].append(monitored)
                LOGGER.info(
                    "epoch %d: loss %.4f, mask mAP50 %.4f, mAP75 %.4f",
                    epoch + 1,
                    epoch_loss,
                    map50,
                    map75,
                )
            metric = history[monitor][-1]
            improved = metric > best_metric if maximize else metric < best_metric
            if improved:
                best_metric = metric
                stale = 0
                torch.save(model.state_dict(), model_dir / WEIGHTS_FILENAME)
            else:
                stale += 1
                if stale >= config.early_stop_patience:
                    LOGGER.info("early stopping after %d stale epochs", stale)
                    break
        return cast("dict[str, Sequence[float]]", history)

    def _train_epoch(
        self,
        model: Any,
        optimizer: Any,
        scaler: Any,
        samples: Array,
        labels: Array,
        indices: list[int],
        config: ModelConfig,
        epoch: int,
        augment: bool,
        device: str,
    ) -> float:
        """One training epoch over shuffled batches; mean batch loss."""
        import torch

        rng = np.random.default_rng(config.seed + epoch)
        order = rng.permutation(indices)
        # Per-epoch sample cap: the permutation is fresh each epoch, so the
        # prefix is a fresh seeded subset (validation is never capped).
        if config.max_samples_per_epoch is not None:
            order = order[: min(int(config.max_samples_per_epoch), len(order))]
        model.train()
        losses: list[float] = []
        batch_size = max(config.batch_size, 1)
        batches = tqdm(
            range(0, len(order), batch_size),
            desc=f"epoch {epoch + 1}/{config.max_epochs} train",
            unit="batch",
            disable=None,  # bars on terminals, silent under pytest/redirects
        )
        for start in batches:
            batch = order[start : start + batch_size]
            images, targets = _batch_to_torch(samples, labels, batch, augment, device)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
            batches.set_postfix(loss=f"{losses[-1]:.4f}")
        return float(np.mean(losses)) if losses else 0.0

    def _validate(
        self,
        model: Any,
        samples: Array,
        labels: Array,
        indices: list[int],
        config: ModelConfig,
        epoch: int,
        device: str,
    ) -> tuple[float, float]:
        """Mask mAP50 / mAP75 over the validation windows."""
        import torch

        model.eval()
        gts: list[list[Instance]] = []
        preds: list[list[Instance]] = []
        with torch.inference_mode():
            for i in tqdm(
                indices,
                desc=f"epoch {epoch + 1}/{config.max_epochs} val",
                unit="window",
                disable=None,
            ):
                image = _image_to_torch(samples[i], device)
                output = model([image])[0]
                pred_instances = [
                    Instance(
                        mask=np.asarray(inst["mask"], dtype=bool),
                        class_id=int(inst["class_id"]),
                        score=float(inst["score"]),
                    )
                    for inst in _output_to_instances(output)
                ]
                gts.append(instances_from_label(np.asarray(labels[i])))
                preds.append(pred_instances)
        map50 = mask_average_precision(
            gts, preds, num_classes=config.num_classes, iou_threshold=AP50
        )
        map75 = mask_average_precision(
            gts, preds, num_classes=config.num_classes, iou_threshold=AP75
        )
        model.train()
        return map50, map75

    # ------------------------------------------------------------- inference

    def predict_tiles(self, tiles: Array) -> Array:
        """Not available: dense tile scoring is the semantic-segmentation
        path. Instance backends emit detections through
        :meth:`predict_instances`, consumed by the sparse predict path."""
        raise NotImplementedError(
            "MaskRcnnBackend produces instances, not dense score rasters; use predict_instances()"
        )

    def predict_instances(self, tiles: Array) -> list[list[dict[str, Any]]]:
        """Instance detections per window of a ``(B, H, W, F)`` batch.

        Returns one list per window; each detection is a dict with
        ``mask`` ((H, W) uint8, window pixel coordinates), ``score``
        (float), and ``class_id`` (int >= 1). Background class 0 is
        dropped; masks are binarized at 0.5.
        """
        import torch

        if self._model is None or self._config is None:
            raise RuntimeError("call build() or load() before predict_instances()")
        tiles = np.asarray(tiles)
        if tiles.ndim != 4:
            raise ValueError(f"expected (B, H, W, F) tiles, got shape {tiles.shape}")
        device = next(self._model.parameters()).device
        self._model.eval()
        results: list[list[dict[str, Any]]] = []
        with torch.inference_mode():
            for window in tiles:
                image = _image_to_torch(window, device)
                output = self._model([image])[0]
                results.append(_output_to_instances(output))
        return results

    # -------------------------------------------------------------- persist

    def save(self, model_dir: Path) -> None:
        """Write ``model_config.json``; the trained weights stay in the
        best-on-``val_map`` checkpoint written by :meth:`fit`."""
        if self._config is None:
            raise RuntimeError("nothing to save: build() or load() first")
        save_model_config(self._config, Path(model_dir) / CONFIG_FILENAME)

    def load(self, model_dir: Path) -> None:
        """Restore config + weights (``model.pt``) from the model dir.

        A backend already ``build()``-ed for a new run keeps its current
        config and takes only the weights — the fine-tuning warm start
        (plan 3.2): hyperparameters come from the new run, weights from
        the base model."""
        import torch

        model_dir = Path(model_dir)
        if self._config is None:
            config = load_model_config(model_dir / CONFIG_FILENAME)
            self.build(config)
        weights = model_dir / WEIGHTS_FILENAME
        if not weights.exists():
            raise FileNotFoundError(f"model weights not found: {weights}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(weights, map_location=device, weights_only=True)
        self._model.load_state_dict(state)


# ------------------------------------------------------------------- helpers


def _image_to_torch(window: Array, device: str) -> Any:
    """(H, W, F) window -> (F, H, W) float tensor on the device."""
    import torch

    return torch.as_tensor(np.asarray(window), dtype=torch.float32, device=device).permute(2, 0, 1)


def _targets_for_label(label: Array, device: str) -> dict[str, Any]:
    """torchvision target dict from one per-pixel class map."""
    import torch

    instances = instances_from_label(np.asarray(label))
    boxes = torch.as_tensor([mask_box(i.mask) for i in instances], dtype=torch.float32)
    labels = torch.as_tensor([i.class_id for i in instances], dtype=torch.int64)
    masks = (
        torch.as_tensor(np.stack([i.mask for i in instances]), dtype=torch.uint8)
        if instances
        else torch.zeros((0, *np.asarray(label).shape), dtype=torch.uint8)
    )
    return {
        "boxes": boxes.to(device),
        "labels": labels.to(device),
        "masks": masks.to(device),
    }


def _batch_to_torch(
    samples: Array,
    labels: Array,
    batch: Array,
    augment: bool,
    device: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """One batch of windows -> torchvision images + instance targets.

    With ``augment``, the image and label map are flipped together
    (p = 0.5 per axis) before the instance targets are derived.
    """
    import torch

    images: list[Any] = []
    targets: list[dict[str, Any]] = []
    for i in batch:
        window = np.asarray(samples[i], dtype=np.float32)
        label = np.asarray(labels[i])
        if augment:
            for axis in (0, 1):
                if float(torch.rand(1)) > 0.5:
                    window = np.flip(window, axis=axis)
                    label = np.flip(label, axis=axis)
        images.append(torch.as_tensor(window, device=device).permute(2, 0, 1))
        targets.append(_targets_for_label(np.ascontiguousarray(label), device))
    return images, targets


def _output_to_instances(output: dict[str, Any]) -> list[dict[str, Any]]:
    """torchvision output -> instance dicts (mask, score, class_id).

    Background class 0 is dropped; masks are binarized at 0.5.
    """
    detections: list[dict[str, Any]] = []
    for raw_mask, score, label in zip(
        output["masks"], output["scores"], output["labels"], strict=False
    ):
        class_id = int(label)
        if class_id < 1:
            continue
        mask = (np.asarray(raw_mask[0].detach().cpu()) >= 0.5).astype(np.uint8)
        if not mask.any():
            continue
        detections.append({"mask": mask, "score": float(score), "class_id": class_id})
    return detections


__all__ = ["MaskRcnnBackend", "WEIGHTS_FILENAME"]
