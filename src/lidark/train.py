"""Training facade: dataset, split, backend, checkpoint, and log.

Replaces the original ``Training`` class. Resolves the backend through the
registry (default ``"unet"``), so no torch import happens here; splits the
dataset object-disjointly by default (group mode: whole spatial object
groups — never a single object twice — reach one side, class-stratified;
the original seeded window shuffle + ``val_split`` semantics remain
available as window mode); writes the best checkpoint,
``model_config.json``, and the key/value training log - including the full
per-epoch history (``loss``, ``iou``, ``val_iou``, ...) - as pretty-printed
JSON (``<model_dir>/training_log.json``, next to the model itself).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree

from lidark.config import ModelConfig, load_model_config
from lidark.dataset import Dataset, create_dataset, load_dataset
from lidark.io.vectors import read_polygon_union
from lidark.logging import get_logger
from lidark.models.base import CONFIG_FILENAME, TrainingData
from lidark.models.registry import create_backend

LOGGER = get_logger("train")

# Training log lives inside the model directory, next to the model itself.
LOG_FILENAME = "training_log.json"

# Characters that cannot appear in a model directory name (Windows- and
# POSIX-invalid path characters plus whitespace).
_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\s]+')


def model_id(model_name: str, timestamp: str) -> str:
    """Model id: sanitized user-supplied name + underscore + timestamp."""
    return f"{sanitize_model_name(model_name)}_{timestamp}"


def sanitize_model_name(name: str) -> str:
    """Make a user-supplied model name safe as a path segment.

    Invalid filename characters and whitespace become ``_``; leading/trailing
    dots and underscores are stripped.

    Raises:
        ValueError: If the name is empty after sanitizing.
    """
    cleaned = _INVALID_NAME_CHARS.sub("_", name.strip()).strip("._")
    if not cleaned:
        raise ValueError(f"model name {name!r} is empty after sanitizing")
    return cleaned


def train(
    layout: Any,
    area: str | Sequence[str],
    feature_list: list[str] | tuple[str, ...] | None = None,
    true_objects: str | None = None,
    false_objects: str | None = None,
    *,
    model_name: str,
    dataset_id: str | None = None,
    test_area: str | None = None,
    model_config: ModelConfig | None = None,
    class_names: Sequence[str] | None = None,
    class_weights: Sequence[float] | None = None,
    class_balanced: bool = False,
    base_model_id: str | None = None,
    early_stop_patience: int | None = None,
    augment: bool = True,
) -> str:
    """Train a model and return its id.

    The model id is the **user-supplied name plus the timestamp**
    (``<name>_<YYYYMMDD-HHMMSS>``), naming
    ``DeepLearning/model/<name>_<timestamp>/`` and the log file — the name
    makes models memorable, the timestamp keeps them unique.

    Either create a dataset (``true_objects`` + ``feature_list`` given, or a
    config carrying ``feature_list``) or load one via ``dataset_id``
    (the original ``fetch_data`` / ``data_timestamp`` pair).

    Args:
        layout: :class:`~lidark.config.ProjectLayout`.
        area: Project area name, or several names to train **one** model on
            all of their tiles (the label shapefiles are shared and matched
            spatially per area; every area must contain the feature layers
            and all areas must share one pixel size —
            :func:`~lidark.dataset.create_dataset` raises ``ValueError``
            otherwise).
        feature_list: Feature layers (required when creating the dataset).
        true_objects: True-object shapefile (dataset creation).
        false_objects: Optional false-object shapefile.
        model_name: User-supplied model name (required); sanitized into the
            model id as ``<name>_<timestamp>``.
        dataset_id: Load an existing dataset instead of creating one.
        test_area: Optional test-area polygon layer (e.g. a ``.gpkg`` file
            under ``DeepLearning/train_annotations/``); tiles whose footprint
            intersects it are excluded from dataset creation (the held-out
            test region).
        model_config: Hyperparameters and backend selection; defaults to a
            :class:`ModelConfig` with parity defaults.
        class_names: Human-readable class names, one per class in compact-id
            order (plan 3.3/MC4); stored in the model config and used by
            predict/evaluate output.
        class_weights: Loss weight per class in compact-id order
            (plan 3.3/MC3); ``None`` = unweighted (original behavior).
        class_balanced: Class-balanced window sampling when creating the
            dataset (plan 3.3/MC3); ignored when ``dataset_id`` is given.
        base_model_id: Fine-tune / warm-start from an existing model
            (plan 3.2): the base model's **weights** are the starting
            point; the new run's hyperparameters (learning rate, epochs,
            ...) and dataset apply. The fine-tuned model always gets a
            fresh id and directory — the base model is never modified —
            and records ``base_model_id`` in its config and training log.
            The base model's backend, input shape (``img_size`` x
            ``len(feature_list)``), and ``num_classes`` must match this
            run; a mismatch raises ``ValueError``. A lower
            ``learning_rate`` than scratch training is typically wanted.
        early_stop_patience: Early stopping: stop after this many
            consecutive epochs without improvement of the monitored metric
            (default 10, the original Keras EarlyStopping patience; a value
            >= ``max_epochs`` disables early stopping). ``None`` keeps the
            ``model_config`` value.
        augment: On-the-fly flip augmentation (original default: True).

    Returns:
        The model id (``<name>_<YYYYMMDD-HHMMSS>``), naming
        ``DeepLearning/model/<name>_<timestamp>/`` and the log file.

    Raises:
        ValueError: If ``model_name`` is empty after sanitizing, or the
            base model is incompatible with this run (plan 3.2 guards).
        FileNotFoundError: If the base model does not exist.
    """
    name = sanitize_model_name(model_name)
    config = model_config if model_config is not None else ModelConfig()
    if feature_list is not None:
        config = ModelConfig(
            **{
                **_config_fields(config),
                "feature_list": tuple(feature_list),
            }
        )
    if not config.feature_list:
        raise ValueError("feature_list is required (argument or model_config)")
    if early_stop_patience is not None:
        config = ModelConfig(
            **{
                **_config_fields(config),
                "early_stop_patience": int(early_stop_patience),
            }
        )

    if dataset_id is not None:
        dataset = load_dataset(layout, dataset_id)
    else:
        if true_objects is None:
            raise ValueError("true_objects is required when creating a dataset")
        dataset = create_dataset(
            layout,
            area,
            config.feature_list,
            true_objects,
            false_objects,
            test_area=test_area,
            img_size=config.img_size,
            class_balanced=class_balanced,
            seed=config.seed,
        )
    if config.num_classes != dataset.num_classes:
        LOGGER.info(
            "num_classes adjusted to the dataset: %s -> %s",
            config.num_classes,
            dataset.num_classes,
        )
    weights = tuple(float(w) for w in class_weights) if class_weights is not None else None
    if weights is not None and len(weights) != dataset.num_classes:
        raise ValueError(
            f"class_weights has {len(weights)} entries, dataset has {dataset.num_classes} classes"
        )
    names = tuple(class_names) if class_names is not None else config.class_names
    if names is not None and len(names) != dataset.num_classes:
        raise ValueError(
            f"class_names has {len(names)} entries, dataset has {dataset.num_classes} classes"
        )
    class_id_map = tuple(sorted(dataset.class_id_map.items()))
    config = ModelConfig(
        **{
            **_config_fields(config),
            "num_classes": dataset.num_classes,
            "class_names": names,
            "class_weights": weights if weights is not None else config.class_weights,
            "class_id_map": class_id_map,
        }
    )

    # Fine-tuning / warm start (plan 3.2): resolve after the config is final
    # (num_classes adjusted to the dataset) so the guards compare what will
    # actually be trained.
    base_model_id = base_model_id or config.base_model_id
    if base_model_id is not None:
        base_config = _load_base_config(layout, base_model_id)
        _check_warm_start(config, base_config, base_model_id)
        config = ModelConfig(
            **{
                **_config_fields(config),
                "base_model_id": base_model_id,
            }
        )

    data = _split(dataset, config, layout)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mid = model_id(name, timestamp)
    model_dir = layout.model_dir(mid)
    model_dir.mkdir(parents=True, exist_ok=True)

    backend = create_backend(config.backend)
    backend.build(config)
    if base_model_id is not None:
        # Warm start: the backend keeps the config it was just built with
        # (this run's hyperparameters) and takes only the weights from the
        # base model directory (plan 3.2: load() + fit(), no protocol growth).
        LOGGER.info("warm-starting %s from base model %s", mid, base_model_id)
        backend.load(layout.model_dir(base_model_id))
    history = backend.fit(data, model_dir, augment=augment)
    backend.save(model_dir)

    _write_log(layout, model_dir, config, dataset, history, augment, data.meta.get("split"))
    LOGGER.info("model %s trained (backend %s)", mid, config.backend)
    return mid


def _load_base_config(layout: Any, base_model_id: str) -> ModelConfig:
    """Read the base model's ``model_config.json`` (plan 3.2).

    Raises:
        FileNotFoundError: If the base model or its config is missing.
    """
    config_path = layout.model_dir(base_model_id) / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"base model {base_model_id!r} not found ({config_path})")
    return load_model_config(config_path)


def _check_warm_start(config: ModelConfig, base: ModelConfig, base_model_id: str) -> None:
    """Plan 3.2 compatibility guards — checked, not assumed.

    The base model's backend, input shape (``img_size`` x
    ``len(feature_list)``), and ``num_classes`` must match this run; the
    U-Net additionally needs the same ``unet_size`` (its architecture
    parameter — a mismatch would otherwise surface as a cryptic state-dict
    error inside torch). Differing feature-layer *names* with the same
    count load cleanly but change the meaning of the inputs, so they are
    warned about, not rejected (the plan guards the shape only).

    Raises:
        ValueError: On any mismatch.
    """
    checks: list[tuple[str, Any, Any]] = [
        ("backend", base.backend, config.backend),
        ("img_size", tuple(base.img_size), tuple(config.img_size)),
        ("feature_list length", len(base.feature_list), len(config.feature_list)),
        ("num_classes", base.num_classes, config.num_classes),
    ]
    if config.backend == "unet":
        checks.append(("unet_size", base.unet_size, config.unet_size))
    mismatches = [
        f"{field}: base {base_value!r} vs this run {new_value!r}"
        for field, base_value, new_value in checks
        if base_value != new_value
    ]
    if mismatches:
        raise ValueError(
            f"base model {base_model_id!r} cannot be warm-started for this run: "
            + "; ".join(mismatches)
        )
    if tuple(base.feature_list) != tuple(config.feature_list):
        LOGGER.warning(
            "feature layers differ from base model %s (%s vs %s); the warm "
            "start assumes the same data products",
            base_model_id,
            base.feature_list,
            config.feature_list,
        )


def _split(dataset: Dataset, config: ModelConfig, layout: Any = None) -> TrainingData:
    """Split a dataset into training and validation windows.

    ``group`` (default): object-disjoint, class-stratified split — label
    objects within ``val_group_buffer`` metres (default: the training
    window diagonal) form one group and whole groups are assigned to one
    side, so the same object — or a cluster of neighbouring objects —
    never appears in both halves. ``window``: the original seeded
    shuffle + ``val_split`` tail cut. Legacy datasets without object
    provenance fall back to ``window`` with a warning.
    """
    if config.val_split_mode == "group":
        if dataset.window_objects is not None:
            return _split_grouped(dataset, config, layout)
        LOGGER.warning(
            "dataset %s records no object provenance (created before group "
            "splits); falling back to the window-level split, which cannot "
            "keep an object out of both halves — re-create the dataset to "
            "enable group mode",
            dataset.id,
        )
        return _split_window(dataset, config, legacy_fallback=True)
    return _split_window(dataset, config)


def _split_window(
    dataset: Dataset, config: ModelConfig, *, legacy_fallback: bool = False
) -> TrainingData:
    """Original shuffle + split semantics, seeded (fixes the original's
    unseeded ``random.shuffle``). Multi-class datasets additionally get a
    class-stratified split (plan 3.3/MC7): every foreground class with at
    least two windows appears in both halves, so per-class val IoU stays
    meaningful. Binary datasets keep the exact original split."""
    n = len(dataset.samples)
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(n)
    split = int(n * config.val_split)
    if split <= 0 or split >= n:
        val_list: list[int] = []
        train_list = [int(i) for i in indices]
    else:
        val_list = [int(i) for i in indices[-split:]]
        train_list = [int(i) for i in indices[:-split]]
        if dataset.num_classes > 1:
            val_list, train_list = _stratify(dataset, val_list, train_list)
    info = _split_info("window", n, len(train_list), len(val_list), config)
    info["legacy_fallback"] = legacy_fallback
    return _training_data(dataset, train_list, val_list, info)


def _split_grouped(dataset: Dataset, config: ModelConfig, layout: Any) -> TrainingData:
    """Object-disjoint, class-stratified split over spatial object groups.

    Objects within ``val_group_buffer`` metres (default: the training
    window diagonal, which also keeps a window placed around one group
    from containing another group's pixels) are merged into one group;
    groups are assigned whole, in seeded order, until every class reaches
    its validation share, so neither an object nor a cluster of objects
    can appear in both halves. ``val_area`` forces every window whose
    footprint intersects the polygon layer into validation first (with
    its whole group), and ``val_per_class`` overrides the validation
    fraction per class. Classes with at least two groups are guaranteed
    windows on both sides (plan 3.3/MC7 at group level).
    """
    n = len(dataset.samples)
    if n == 0:
        return _training_data(dataset, [], [], _split_info("group", 0, 0, 0, config))
    classes = _window_classes(dataset)
    window_objects = dataset.window_objects or []
    objects = dataset.objects or {}
    window_bounds = dataset.window_bounds or []
    pixel_size = dataset.pixel_size
    if pixel_size is None:
        raise ValueError(
            f"dataset {dataset.id!r} records no pixel_size; the group split "
            "needs it to derive the default group buffer"
        )
    height, width = (int(v) for v in dataset.meta["img_size"])
    buffer_m = config.val_group_buffer
    if buffer_m is None:
        buffer_m = math.hypot(height, width) * pixel_size
    elif buffer_m < 0:
        raise ValueError(f"val_group_buffer must be >= 0, got {buffer_m}")

    groups = _object_groups(objects, buffer_m)
    window_groups = [groups[oid] for oid in window_objects]
    group_ids = sorted(set(window_groups))
    group_classes: dict[int, Counter[int]] = {gid: Counter() for gid in group_ids}
    for gid, cls in zip(window_groups, classes, strict=True):
        group_classes[gid][cls] += 1
    group_sizes = {gid: sum(group_classes[gid].values()) for gid in group_ids}
    total_per_class: Counter[int] = Counter(classes)

    # Validation share per class (0 = false-object background windows): the
    # global fraction unless per-class overrides are given.
    shares: dict[int, float] = {c: config.val_split for c in range(dataset.num_classes + 1)}
    for cls, fraction in config.val_per_class or ():
        shares[int(cls)] = float(fraction)
    targets: dict[int, int] = {}
    for cls, share in shares.items():
        total = total_per_class.get(cls, 0)
        targets[cls] = max(1, int(round(total * share))) if share > 0.0 and total else 0

    # Val-area hold-out: windows whose footprint intersects the polygon
    # layer go to validation, together with their whole group.
    forced: set[int] = set()
    if config.val_area:
        if layout is None:
            raise ValueError("val_area requires the project layout")
        val_geom = read_polygon_union(layout.train_annotations_dir / config.val_area)
        for i, gid in enumerate(window_groups):
            if box(*window_bounds[i]).intersects(val_geom):
                forced.add(gid)

    val_groups: set[int] = set(forced)
    val_counts: Counter[int] = Counter()
    for gid in forced:
        val_counts.update(group_classes[gid])
    if config.val_split > 0.0:
        order = np.random.default_rng(config.seed).permutation(len(group_ids))
        for pos in order:
            gid = group_ids[int(pos)]
            if gid in val_groups:
                continue
            if all(val_counts[cls] >= targets.get(cls, 0) for cls in targets):
                break
            if any(
                group_classes[gid][cls] > 0 and val_counts[cls] < targets[cls] for cls in targets
            ):
                val_groups.add(gid)
                val_counts.update(group_classes[gid])

    # Both-halves guarantee at group level (MC7): every foreground class
    # with at least two groups keeps windows on both sides. Groups held
    # out by val_area are never moved back — the hold-out wins.
    train_counts: Counter[int] = Counter({cls: total_per_class[cls] for cls in total_per_class})
    train_counts.subtract(val_counts)
    for cls in range(1, dataset.num_classes + 1):
        gids = [gid for gid in group_ids if group_classes[gid][cls] > 0]
        if len(gids) < 2:
            if total_per_class[cls] > 0:
                LOGGER.warning(
                    "class %d has a single object group; split not stratified for it", cls
                )
            continue
        if shares.get(cls, 0.0) <= 0.0:
            continue  # the user asked for no validation windows of this class
        if val_counts[cls] == 0:
            donor = min(
                (gid for gid in gids if gid not in val_groups),
                key=lambda gid: group_sizes[gid],
            )
            val_groups.add(donor)
            val_counts.update(group_classes[donor])
            train_counts.subtract(group_classes[donor])
        elif train_counts[cls] == 0:
            movable = [gid for gid in gids if gid in val_groups and gid not in forced]
            if not movable:
                LOGGER.warning(
                    "class %d stays validation-only: all its groups lie in the validation area",
                    cls,
                )
                continue
            donor = min(movable, key=lambda gid: group_sizes[gid])
            val_groups.remove(donor)
            val_counts.subtract(group_classes[donor])
            train_counts.update(group_classes[donor])

    val_list = [i for i in range(n) if window_groups[i] in val_groups]
    train_list = [i for i in range(n) if window_groups[i] not in val_groups]
    if not train_list and val_list:
        # A single group swallowed everything: train on all windows instead
        # of crashing on an empty training set.
        LOGGER.warning(
            "group split: all %d windows form a single group; training on all "
            "windows without validation",
            n,
        )
        train_list, val_list = list(range(n)), []

    info = _split_info("group", n, len(train_list), len(val_list), config)
    info["group_buffer_m"] = buffer_m
    info["num_groups"] = len(group_ids)
    info["num_val_groups"] = len(val_groups)
    info["val_area"] = config.val_area
    info["per_class_train"] = {str(cls): train_counts[cls] for cls in sorted(total_per_class)}
    info["per_class_val"] = {str(cls): val_counts[cls] for cls in sorted(total_per_class)}
    overlaps = _cross_side_overlaps(objects, groups, window_bounds, window_groups, val_groups)
    info["cross_side_window_overlaps"] = overlaps
    if overlaps:
        LOGGER.warning(
            "%d window(s) still contain objects from the other half of the "
            "split (group buffer %.1f m is smaller than a window); raise "
            "--val-group-buffer or accept the overlap",
            overlaps,
            buffer_m,
        )
    return _training_data(dataset, train_list, val_list, info)


def _split_info(
    mode: str, n: int, num_train: int, num_val: int, config: ModelConfig
) -> dict[str, Any]:
    """Shared split summary recorded in the training log (``meta["split"]``)."""
    return {
        "mode": mode,
        "seed": config.seed,
        "num_windows": n,
        "num_train": num_train,
        "num_val": num_val,
        "val_fraction": (num_val / n) if n else 0.0,
    }


def _training_data(
    dataset: Dataset, train_list: list[int], val_list: list[int], info: dict[str, Any]
) -> TrainingData:
    """Assemble :class:`TrainingData` with the split summary attached to a
    copy of the metadata (``meta["split"]``, read by the training log)."""
    meta = dict(dataset.meta)
    meta["split"] = info
    return TrainingData(
        samples=dataset.samples,
        labels=dataset.labels,
        train_indices=np.asarray(train_list, dtype=np.int64),
        val_indices=np.asarray(val_list, dtype=np.int64),
        num_classes=dataset.num_classes,
        meta=meta,
    )


def _object_groups(objects: dict[str, dict[str, Any]], buffer_m: float) -> dict[str, int]:
    """Connected-component grouping of label objects: objects whose bounds
    lie within ``buffer_m`` metres of each other share one split group, so
    clustered objects always land on the same side of the split.

    Returns:
        Object id -> group id (compact, assigned in sorted-object order).
    """
    ids = sorted(objects)
    parent = list(range(len(ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(len(ids)), key=lambda i: objects[ids[i]]["bounds"][0])
    for pos, i in enumerate(order):
        bounds_i = objects[ids[i]]["bounds"]
        for j in order[pos + 1 :]:
            bounds_j = objects[ids[j]]["bounds"]
            if bounds_j[0] - bounds_i[2] > buffer_m:
                break  # minx-sorted: everything further is out of reach
            root_i, root_j = find(i), find(j)
            if root_i != root_j and _box_distance(bounds_i, bounds_j) <= buffer_m:
                parent[root_j] = root_i
    roots: dict[int, int] = {}
    groups: dict[str, int] = {}
    for i, oid in enumerate(ids):
        root = find(i)
        if root not in roots:
            roots[root] = len(roots)
        groups[oid] = roots[root]
    return groups


def _box_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two (minx, miny, maxx, maxy) boxes."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def _cross_side_overlaps(
    objects: dict[str, dict[str, Any]],
    groups: dict[str, int],
    window_bounds: list[list[float]],
    window_groups: list[int],
    val_groups: set[int],
) -> int:
    """Safety net (group mode): windows whose footprint still contains
    objects from the other side of the split.

    Only possible when the group buffer is smaller than a training window;
    with the default buffer (the window diagonal) a window placed around
    one group cannot reach another group's objects.
    """
    if not window_bounds:
        return 0
    train_boxes = [
        box(*info["bounds"]) for oid, info in objects.items() if groups[oid] not in val_groups
    ]
    val_boxes = [box(*info["bounds"]) for oid, info in objects.items() if groups[oid] in val_groups]
    hits = 0
    if train_boxes:
        tree = STRtree(train_boxes)
        hits += sum(
            len(tree.query(box(*window_bounds[i])))
            for i, gid in enumerate(window_groups)
            if gid in val_groups
        )
    if val_boxes:
        tree = STRtree(val_boxes)
        hits += sum(
            len(tree.query(box(*window_bounds[i])))
            for i, gid in enumerate(window_groups)
            if gid not in val_groups
        )
    return int(hits)


def _stratify(dataset: Dataset, val: list[int], train: list[int]) -> tuple[list[int], list[int]]:
    """Greedy class-stratification of the split halves (plan 3.3/MC7).

    For every foreground class missing from one half, one window of that
    class moves over from the other half, evicting a window of the half's
    most common class so the split sizes stay exact. Classes with a single
    window cannot appear in both halves and are left as-is (warned).
    """
    classes = _window_classes(dataset)
    val_classes: Counter[int] = Counter(classes[i] for i in val)
    train_classes: Counter[int] = Counter(classes[i] for i in train)
    for cls in sorted(set(val_classes) | set(train_classes)):
        if cls == 0:  # background windows need no stratification
            continue
        total = val_classes[cls] + train_classes[cls]
        if total < 2:
            LOGGER.warning("class %d has a single window; split not stratified for it", cls)
            continue
        if val_classes[cls] == 0:
            donor = next(i for i in train if classes[i] == cls)
            evicted = max(val, key=lambda i: val_classes[classes[i]])
            train.remove(donor)
            train.append(evicted)
            val.remove(evicted)
            val.append(donor)
            train_classes[cls] -= 1
            val_classes[cls] += 1
            val_classes[classes[evicted]] -= 1
            train_classes[classes[evicted]] += 1
        elif train_classes[cls] == 0:
            donor = next(i for i in val if classes[i] == cls)
            evicted = max(train, key=lambda i: train_classes[classes[i]])
            val.remove(donor)
            val.append(evicted)
            train.remove(evicted)
            train.append(donor)
            val_classes[cls] -= 1
            train_classes[cls] += 1
            train_classes[classes[evicted]] -= 1
            val_classes[classes[evicted]] += 1
    return val, train


def _window_classes(dataset: Dataset) -> list[int]:
    """Placing class per window: the recorded metadata when present, else
    the max class id of each label window (legacy datasets, MC7)."""
    recorded = dataset.window_classes
    if recorded is not None:
        return recorded
    return [int(np.asarray(dataset.labels[i]).max()) for i in range(len(dataset.labels))]


def _config_fields(config: ModelConfig) -> dict[str, Any]:
    return {
        field.name: getattr(config, field.name) for field in config.__dataclass_fields__.values()
    }


def _per_epoch_history(history: dict[str, Any]) -> dict[str, list[float]]:
    """Per-epoch metric series of the fit: one list per metric name.

    Contains whatever the backend reported per epoch — e.g. ``loss``, ``iou``,
    ``val_loss``, ``val_iou``, and ``imgs_per_second`` — in fit order.
    """
    series: dict[str, list[float]] = {}
    for key, values in history.items():
        if isinstance(values, (list, tuple)):
            series[key] = [float(v) for v in values]
    return series


def _write_log(
    layout: Any,
    model_dir: Path,
    config: ModelConfig,
    dataset: Dataset,
    history: dict[str, Any],
    augment: bool,
    split_info: dict[str, Any] | None = None,
) -> None:
    """The original ``create_log`` rows (plus backend/classes and the full
    per-epoch history), as a pretty-printed JSON file written **into the
    model directory** (``<model_dir>/training_log.json``) so the log lives
    next to the model it belongs to."""
    rows: dict[str, Any] = {
        "log_id": model_dir.name,
        "data_id": dataset.id,
        "epsg": layout.epsg,
        "num samples": len(dataset.samples),
        "augmented": augment,
        "features": list(config.feature_list),
        "img size": list(config.img_size),
        "learning rate": config.learning_rate,
        "batch size": config.batch_size,
        "max samples per epoch": config.max_samples_per_epoch,
        "validation split": config.val_split,
        "validation split mode": (split_info or {}).get("mode", config.val_split_mode),
        "validation split details": split_info,
        "max epochs": config.max_epochs,
        "early stop patience": config.early_stop_patience,
        "unet size": config.unet_size,
        "dropout": config.dropout,
        "best iou": max(history.get("val_iou", [0.0]) or [0.0]),
        "best mAP": max(history.get("val_map", [0.0]) or [0.0]),
        "epochs run": len(history.get("loss", [])),
        "history": _per_epoch_history(history),
        "backend": config.backend,
        "num classes": config.num_classes,
        "class names": list(config.class_names) if config.class_names else None,
        "class weights": list(config.class_weights) if config.class_weights else None,
        "base model": config.base_model_id,
        "seed": config.seed,
    }
    rates = history.get("imgs_per_second")
    rows["imgs per second (mean)"] = float(np.mean(rates)) if rates else None
    payload = json.dumps(rows, indent=2)
    (Path(model_dir) / LOG_FILENAME).write_text(f"{payload}\n", encoding="utf-8")


__all__ = ["train", "model_id", "sanitize_model_name", "LOG_FILENAME"]
