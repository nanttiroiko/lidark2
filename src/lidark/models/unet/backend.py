"""Default U-Net backend: build, train (memmap streaming), persist, infer.

Training streams per-index windows straight from the dataset memmap through a
batched index loop (7, P5): RAM is bounded by the batch size and nothing is
copied into RAM beyond it. Flip augmentation (original: materialized H/V
copies) happens on the fly with p=0.5 per axis - the same flip family,
without the 4x dataset duplication. The best checkpoint (``val_iou``) is
written to ``<model_dir>/model.pt``; per-epoch throughput (img/s) is
returned with the history so the log can surface regressions (7, P6).

The torch head emits logits; losses are the numerically stable
``BCEWithLogitsLoss`` / ``CrossEntropyLoss`` and :meth:`predict_tiles`
applies sigmoid/softmax explicitly. History keys (``loss``, ``iou``,
``val_loss``, ``val_iou``, ``imgs_per_second``) match the former Keras
backend so the training log is unchanged.

All torch imports are lazy: registering or merely listing the backend never
requires the framework; only ``build``/``fit``/inference do.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from lidark.config import (
    TASK_SEMANTIC_SEGMENTATION,
    ModelConfig,
    save_model_config,
)
from lidark.logging import get_logger
from lidark.models.base import CONFIG_FILENAME, MODEL_FILENAME, TrainingData
from lidark.models.unet.metrics import IoU

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lidark._types import Array

LOGGER = get_logger("models.unet")


class UnetBackend:
    """torch U-Net implementing the :class:`ModelBackend` protocol."""

    task = TASK_SEMANTIC_SEGMENTATION

    def __init__(self) -> None:
        self._config: ModelConfig | None = None
        self._model: Any = None

    # ------------------------------------------------------------------ build

    def build(self, config: ModelConfig) -> None:
        """Construct the U-Net for the config's classes and input layers."""
        from lidark.models.unet.architecture import get_unet_model

        if config.num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {config.num_classes}")
        if not config.feature_list:
            raise ValueError("ModelConfig.feature_list must name the input layers")
        if config.class_weights is not None and len(config.class_weights) != config.num_classes:
            raise ValueError(
                f"class_weights has {len(config.class_weights)} entries, "
                f"num_classes is {config.num_classes}"
            )
        out_channels = 1 if config.num_classes == 1 else config.num_classes + 1
        model = get_unet_model(
            (*config.img_size, len(config.feature_list)),
            out_channels,
            "relu",
            config.dropout,
            config.unet_size,
        )
        self._config = config
        self._model = model

    # ------------------------------------------------------------------- fit

    def fit(
        self, data: TrainingData, model_dir: Path, *, augment: bool = True
    ) -> dict[str, Sequence[float]]:
        """Train the U-Net; validate with IoU.

        The best-on-``val_iou`` (or ``iou`` without a validation split)
        checkpoint (``model.pt``) and early stopping (after
        ``ModelConfig.early_stop_patience`` stale epochs, default 10)
        follow the monitored metric, mirroring the former Keras callbacks.
        Returns the metric history.
        """
        import torch

        if self._model is None or self._config is None:
            raise RuntimeError("call build() before fit()")
        config = self._config
        torch.manual_seed(config.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self._model.to(device)
        # torch.compile wraps (not copies) the module; forwards go through
        # the wrapper, but checkpoints keep the raw module's prefix-free
        # state dict so load() can rebuild a plain UNet.
        forward_model = torch.compile(model) if config.jit_compile else model
        # Class weights (plan 3.3/MC3): binary = pos_weight on the single
        # sigmoid channel; multi-class = CrossEntropyLoss weight with the
        # background fixed at 1.0. Unweighted reproduces the original loss.
        if config.num_classes == 1:
            pos_weight = (
                torch.tensor(float(config.class_weights[0]), device=device)
                if config.class_weights
                else None
            )
            loss_fn: Any = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            weight = (
                torch.tensor([1.0, *(float(w) for w in config.class_weights)], device=device)
                if config.class_weights
                else None
            )
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scaler: Any = None
        if config.mixed_precision and device == "cuda":
            scaler = torch.amp.GradScaler("cuda")

        samples, labels = data.samples, data.labels
        train_indices = list(np.asarray(data.train_indices))
        val_indices = list(np.asarray(data.val_indices))
        has_val = len(val_indices) > 0
        history: dict[str, list[float]] = {
            "loss": [],
            "iou": [],
            "imgs_per_second": [],
        }
        if has_val:
            history["val_loss"] = []
            history["val_iou"] = []
        monitor = "val_iou" if has_val else "iou"
        best_metric = -float("inf")
        stale = 0

        model_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(config.max_epochs):
            start = time.perf_counter()
            train_loss, train_iou = self._run_epoch(
                forward_model,
                loss_fn,
                optimizer,
                scaler,
                samples,
                labels,
                train_indices,
                config,
                epoch,
                augment,
                device,
                training=True,
            )
            history["loss"].append(train_loss)
            history["iou"].append(train_iou)
            history["imgs_per_second"].append(
                max(len(train_indices), 1) / max(time.perf_counter() - start, 1e-9)
            )
            if has_val:
                val_loss, val_iou = self._run_epoch(
                    forward_model,
                    loss_fn,
                    None,
                    None,
                    samples,
                    labels,
                    val_indices,
                    config,
                    epoch,
                    False,
                    device,
                    training=False,
                )
                history["val_loss"].append(val_loss)
                history["val_iou"].append(val_iou)
                LOGGER.info(
                    "epoch %d: loss %.4f, iou %.4f, val_loss %.4f, val_iou %.4f",
                    epoch + 1,
                    train_loss,
                    train_iou,
                    val_loss,
                    val_iou,
                )
            else:
                LOGGER.info("epoch %d: loss %.4f, iou %.4f", epoch + 1, train_loss, train_iou)
            metric = history[monitor][-1]
            if metric > best_metric:
                best_metric = metric
                stale = 0
                torch.save(model.state_dict(), Path(model_dir) / MODEL_FILENAME)
            else:
                stale += 1
                if stale >= config.early_stop_patience:
                    LOGGER.info("early stopping after %d stale epochs", stale)
                    break
        result: dict[str, Sequence[float]] = dict(history)
        return result

    def _run_epoch(
        self,
        model: Any,
        loss_fn: Any,
        optimizer: Any,
        scaler: Any,
        samples: Array,
        labels: Array,
        indices: list[int],
        config: ModelConfig,
        epoch: int,
        augment: bool,
        device: str,
        *,
        training: bool,
    ) -> tuple[float, float]:
        """One pass over shuffled (train) or ordered (val) index batches.

        Returns the mean batch loss and the epoch's accumulated IoU. Nothing
        beyond the batch is read from the memmap (7, P5); flip augmentation
        is applied per window with p = 0.5 per axis.
        """
        import torch

        batch_size = max(config.batch_size, 1)
        if training:
            rng = np.random.default_rng(config.seed + epoch)
            order = np.asarray(rng.permutation(indices))
            # Per-epoch sample cap: the permutation is fresh each epoch,
            # so the prefix is a fresh seeded subset (validation uncapped).
            cap = config.max_samples_per_epoch
            if cap is not None:
                order = order[: min(int(cap), len(order))]
            model.train()
            # A trailing single-sample batch leaves BatchNorm2d with one
            # value per channel at the 1x1 bottleneck (no variance
            # estimate); drop it. Validation is unaffected (running stats).
            if len(order) % batch_size == 1:
                order = order[:-1]
        else:
            order = np.asarray(indices)
            model.eval()
        metric = IoU(num_classes=config.num_classes)
        losses: list[float] = []
        phase = "train" if training else "val"
        batches = tqdm(
            range(0, len(order), batch_size),
            desc=f"epoch {epoch + 1}/{config.max_epochs} {phase}",
            unit="batch",
            disable=None,  # bars on terminals, silent under pytest/redirects
        )
        for start in batches:
            batch = order[start : start + batch_size]
            images, targets = _batch_to_torch(samples, labels, batch, augment and training, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # BCEWithLogitsLoss broadcasts against the (B, 1, H, W) logits;
            # the (B, H, W) label map needs a channel axis to match.
            loss_target = targets.unsqueeze(1).float() if config.num_classes == 1 else targets
            with torch.set_grad_enabled(training):
                if scaler is not None and training:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(images)
                        loss = loss_fn(logits, loss_target)
                else:
                    logits = model(images)
                    loss = loss_fn(logits, loss_target)
                if training:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
            metric.update(logits.detach(), targets)
            losses.append(float(loss.detach()))
            batches.set_postfix(loss=f"{losses[-1]:.4f}")
        mean_loss = float(np.mean(losses)) if losses else 0.0
        return mean_loss, metric.result()

    # ------------------------------------------------------------- inference

    def predict_tiles(self, tiles: Array) -> Array:
        """Score a ``(B, H, W, F)`` batch.

        Returns:
            ``(B, H, W)`` foreground probabilities for single-class models;
            ``(B, H, W, C)`` per-class probability maps for multi-class
            models (background channel dropped).
        """
        import torch

        if self._model is None or self._config is None:
            raise RuntimeError("call build() or load() before predict_tiles()")
        config = self._config
        tiles = np.asarray(tiles)
        if tiles.ndim != 4:
            raise ValueError(f"expected (B, H, W, F) tiles, got shape {tiles.shape}")
        device = next(self._model.parameters()).device
        self._model.eval()
        with torch.inference_mode():
            images = torch.as_tensor(tiles, dtype=torch.float32, device=device)
            images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
            if config.mixed_precision and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = self._model(images)
            else:
                logits = self._model(images)
            if config.num_classes == 1:
                probs = torch.sigmoid(logits[:, 0])  # (B, H, W)
                return probs.cpu().numpy().astype(np.float32)
            probs = torch.softmax(logits.float(), dim=1)
            probs = probs.permute(0, 2, 3, 1)  # (B, H, W, C + 1)
            return probs[..., 1:].cpu().numpy().astype(np.float32)  # drop background

    # -------------------------------------------------------------- persist

    def save(self, model_dir: Path) -> None:
        """Write ``model_config.json``; the trained weights stay in the
        best-on-``val_iou`` checkpoint written by :meth:`fit`."""
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

        from lidark.config import load_model_config
        from lidark.models.unet.architecture import get_unet_model

        model_dir = Path(model_dir)
        weights = model_dir / MODEL_FILENAME
        if not weights.exists():
            raise FileNotFoundError(f"model weights not found: {weights}")
        if self._config is None:
            self._config = load_model_config(model_dir / CONFIG_FILENAME)
        config = self._config
        out_channels = 1 if config.num_classes == 1 else config.num_classes + 1
        model = get_unet_model(
            (*config.img_size, len(config.feature_list)),
            out_channels,
            "relu",
            config.dropout,
            config.unet_size,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(weights, map_location=device, weights_only=True)
        model.load_state_dict(state)
        self._model = model.to(device)


def _batch_to_torch(
    samples: Array,
    labels: Array,
    batch: Array,
    augment: bool,
    device: str,
) -> tuple[Any, Any]:
    """One batch of windows -> (B, F, H, W) float images + (B, H, W) labels.

    Windows stream straight from the memmap and are cast to float32 here -
    torch does not auto-cast int8 inputs. With ``augment``, the image and
    label map are flipped together (p = 0.5 per axis).
    """
    import torch

    images = np.asarray(samples[list(batch)], dtype=np.float32)
    targets = np.asarray(labels[list(batch)])
    if augment:
        for i in range(len(images)):
            for axis in (0, 1):  # H (up/down), W (left/right)
                if float(torch.rand(1)) > 0.5:
                    images[i] = np.flip(images[i], axis=axis)
                    targets[i] = np.flip(targets[i], axis=axis)
    images_t = torch.as_tensor(np.ascontiguousarray(images), device=device)
    targets_t = torch.as_tensor(np.ascontiguousarray(targets), dtype=torch.long, device=device)
    return images_t.permute(0, 3, 1, 2), targets_t


__all__ = ["UnetBackend"]
