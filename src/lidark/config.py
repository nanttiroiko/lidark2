"""Project layout, CRS defaults, and model configuration.

:class:`ProjectLayout` is the single place that knows the project folder
convention (MODERNIZATION_PLAN.md, section 1). All other modules resolve paths
through it and receive :class:`pathlib.Path` objects; no path is built by string
concatenation elsewhere.

The default CRS lives here and only here: :data:`DEFAULT_EPSG` is the default of
``ProjectLayout.epsg`` (overridable per project/area). No other module
hardcodes an EPSG; CRS is read from raster/vector metadata when present, with
the layout value as fallback and as the CRS written to new outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_EPSG = "EPSG:3067"


@dataclass(frozen=True)
class ProjectLayout:
    """Path conventions of a LIDARK project folder.

    Attributes:
        project_folder: Root of the project; contains ``Lidar/`` and
            ``DeepLearning/``.
        epsg: Default CRS of the project. Fallback when data carries no CRS
            metadata, and the CRS written to new outputs.
    """

    project_folder: Path
    epsg: str = DEFAULT_EPSG

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_folder", Path(self.project_folder))

    # --- Lidar products -------------------------------------------------

    @property
    def lidar_root(self) -> Path:
        return self.project_folder / "Lidar"

    def lidar_area(self, area: str) -> Path:
        return self.lidar_root / area

    def originals_dir(self, area: str) -> Path:
        return self.lidar_area(area) / "00_Original"

    def dtm_dir(self, area: str) -> Path:
        return self.lidar_area(area) / "01_DTM"

    def tpi_dir(self, area: str, radius: float) -> Path:
        label = int(radius) if float(radius).is_integer() else radius
        return self.lidar_area(area) / "02_TPI" / f"radius_{label}"

    # --- DeepLearning ----------------------------------------------------

    @property
    def deep_learning_root(self) -> Path:
        return self.project_folder / "DeepLearning"

    @property
    def train_annotations_dir(self) -> Path:
        return self.deep_learning_root / "train_annotations"

    def evaluate_annotations_dir(self, eval_name: str) -> Path:
        return self.deep_learning_root / "evaluate_annotations" / eval_name

    @property
    def samples_dir(self) -> Path:
        return self.deep_learning_root / "samples"

    @property
    def labels_dir(self) -> Path:
        return self.deep_learning_root / "labels"

    @property
    def model_root(self) -> Path:
        return self.deep_learning_root / "model"

    def model_dir(self, model_id: str) -> Path:
        """Directory of one model (``DeepLearning/model/<name>_<timestamp>/``)."""
        return self.model_root / model_id

    def predictions_dir(self, model_id: str, area: str | None = None) -> Path:
        path = self.model_dir(model_id) / "predictions"
        return path / area if area is not None else path

    def prediction_raster_dir(self, model_id: str, area: str) -> Path:
        return self.predictions_dir(model_id, area) / "raster"

    def prediction_shapes_dir(self, model_id: str, area: str) -> Path:
        return self.predictions_dir(model_id, area) / "shapes"

    @property
    def reports_root(self) -> Path:
        return self.deep_learning_root / "reports"

    def reports_dir(self, model_id: str) -> Path:
        return self.reports_root / model_id


@dataclass(frozen=True)
class ModelConfig:
    """ML backend selection, classes, and hyperparameters (Phases 3-5).

    Defaults reproduce the original training hyperparameters. ``backend``
    selects a ModelBackend through the registry (default ``"unet"``).
    ``class_names`` maps class ids (1..num_classes) to labels; ``None``
    derives ``class_<id>``. ``pred_stride`` is the window stride used at
    prediction time (the original used window/4 = 128 for a 512 window).
    ``feature_list`` records which data maps the model consumes; it is
    written to ``model_config.json`` and read back by ``predict()``.
    """

    backend: str = "unet"
    num_classes: int = 1
    class_names: tuple[str, ...] | None = None
    feature_list: tuple[str, ...] = ()
    img_size: tuple[int, int] = (512, 512)
    learning_rate: float = 0.001
    batch_size: int = 32
    val_split: float = 0.2
    max_epochs: int = 200
    # Cap on the training windows processed per epoch; None = all windows
    # (original behavior). Each epoch draws a fresh seeded random subset
    # of the training indices (validation is never capped) — useful for
    # very large datasets or quick iteration runs.
    max_samples_per_epoch: int | None = None
    # Early stopping: training stops after this many consecutive epochs
    # without improvement of the monitored metric (10 = the original Keras
    # EarlyStopping patience); a value >= ``max_epochs`` disables it.
    early_stop_patience: int = 10
    unet_size: int = 16
    dropout: float = 0.0
    seed: int = 0
    pred_stride: int = 128
    pred_batch_size: int = 32
    mixed_precision: bool = False
    jit_compile: bool = False
    # Mask R-CNN: IoU threshold of the monitored mask mAP metric
    # (0.5 = AP50 default, 0.75 = AP75).
    mask_ap_iou: float = 0.5
    # Faster R-CNN: IoU threshold of the monitored box mAP metric
    # (0.5 = PASCAL VOC mAP@0.5 default, 0.75 = AP75).
    det_ap_iou: float = 0.5
    # Loss weight per class (compact ids 1..num_classes, plan 3.3/MC3);
    # None = unweighted (original behavior).
    class_weights: tuple[float, ...] | None = None
    # (original label id -> compact id) pairs (plan 3.3/MC2); None = identity.
    # Labels with sparse ids (e.g. {1, 5}) are remapped to 1..C at dataset
    # creation; this map travels with the model so predictions (compact ids)
    # can be traced back to the user's original ids.
    class_id_map: tuple[tuple[int, int], ...] | None = None
    # Id of the model this one was warm-started from (plan 3.2); None for a
    # scratch-trained model. Provenance only — recorded in
    # ``model_config.json`` and the training log, never used at inference.
    base_model_id: str | None = None
    # Validation-split strategy. ``"group"`` (default): object-disjoint,
    # class-stratified split — label objects within ``val_group_buffer``
    # metres of each other form one group and whole groups are assigned to
    # one side, so an object (or a cluster of objects) never appears in both
    # halves. ``"window"``: the original seeded window-level split (parity
    # with the old program; objects may overlap the halves).
    val_split_mode: str = "group"
    # Metres within which label objects share a split group (group mode).
    # None = the training-window diagonal, which also keeps a window placed
    # around one group from containing another group's pixels.
    val_group_buffer: float | None = None
    # Optional polygon layer under ``DeepLearning/train_annotations/``
    # (group mode): windows whose footprint intersects it are validation
    # only — a spatial hold-out, mirroring ``test_area``.
    val_area: str | None = None
    # Per-class validation fractions (compact class ids, group mode);
    # unlisted classes use ``val_split``. None = one fraction for all.
    val_per_class: tuple[tuple[int, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.val_split_mode not in ("group", "window"):
            raise ValueError(
                f"val_split_mode must be 'group' or 'window', got {self.val_split_mode!r}"
            )
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError(f"val_split must be in [0, 1), got {self.val_split}")
        if self.max_samples_per_epoch is not None and self.max_samples_per_epoch < 1:
            raise ValueError(
                f"max_samples_per_epoch must be >= 1, got {self.max_samples_per_epoch}"
            )
        if self.early_stop_patience < 1:
            raise ValueError(f"early_stop_patience must be >= 1, got {self.early_stop_patience}")
        for cls, fraction in self.val_per_class or ():
            if int(cls) < 1:
                raise ValueError(f"val_per_class class ids must be >= 1, got {cls}")
            if not 0.0 <= float(fraction) < 1.0:
                raise ValueError(f"val_per_class fractions must be in [0, 1), got {fraction}")

    def class_name(self, class_id: int) -> str:
        """Label of a class id (>= 1)."""
        if class_id < 1:
            raise ValueError(f"class_id must be >= 1, got {class_id}")
        if self.class_names is None:
            return f"class_{class_id}"
        if class_id > len(self.class_names):
            raise ValueError(f"no class_name for class_id {class_id}")
        return self.class_names[class_id - 1]


def save_model_config(config: ModelConfig, path: Path | str) -> Path:
    """Write a model config to ``model_config.json`` (tuples become lists)."""
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": config.backend,
        "task": TASK_SEMANTIC_SEGMENTATION,
        "num_classes": config.num_classes,
        "class_names": list(config.class_names) if config.class_names else None,
        "feature_list": list(config.feature_list),
        "img_size": list(config.img_size),
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "val_split": config.val_split,
        "max_epochs": config.max_epochs,
        "max_samples_per_epoch": config.max_samples_per_epoch,
        "early_stop_patience": config.early_stop_patience,
        "unet_size": config.unet_size,
        "dropout": config.dropout,
        "seed": config.seed,
        "pred_stride": config.pred_stride,
        "pred_batch_size": config.pred_batch_size,
        "mixed_precision": config.mixed_precision,
        "jit_compile": config.jit_compile,
        "mask_ap_iou": config.mask_ap_iou,
        "det_ap_iou": config.det_ap_iou,
        "class_weights": list(config.class_weights) if config.class_weights else None,
        "class_id_map": (
            [list(pair) for pair in config.class_id_map] if config.class_id_map else None
        ),
        "base_model_id": config.base_model_id,
        "val_split_mode": config.val_split_mode,
        "val_group_buffer": config.val_group_buffer,
        "val_area": config.val_area,
        "val_per_class": (
            [[int(cls), float(fraction)] for cls, fraction in config.val_per_class]
            if config.val_per_class
            else None
        ),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_model_config(path: Path | str) -> ModelConfig:
    """Read a ``model_config.json`` back into a :class:`ModelConfig`."""
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    class_names = payload.get("class_names")
    class_weights = payload.get("class_weights")
    class_id_map = payload.get("class_id_map")
    val_per_class = payload.get("val_per_class")
    return ModelConfig(
        backend=payload.get("backend", "unet"),
        num_classes=payload.get("num_classes", 1),
        class_names=tuple(class_names) if class_names else None,
        feature_list=tuple(payload.get("feature_list", ())),
        img_size=tuple(payload.get("img_size", (512, 512))),
        learning_rate=payload.get("learning_rate", 0.001),
        batch_size=payload.get("batch_size", 32),
        val_split=payload.get("val_split", 0.2),
        max_epochs=payload.get("max_epochs", 200),
        max_samples_per_epoch=payload.get("max_samples_per_epoch"),
        early_stop_patience=payload.get("early_stop_patience", 10),
        unet_size=payload.get("unet_size", 16),
        dropout=payload.get("dropout", 0.0),
        seed=payload.get("seed", 0),
        pred_stride=payload.get("pred_stride", 128),
        pred_batch_size=payload.get("pred_batch_size", 32),
        mixed_precision=payload.get("mixed_precision", False),
        jit_compile=payload.get("jit_compile", False),
        mask_ap_iou=payload.get("mask_ap_iou", 0.5),
        det_ap_iou=payload.get("det_ap_iou", 0.5),
        class_weights=tuple(float(w) for w in class_weights) if class_weights else None,
        class_id_map=(
            tuple((int(pair[0]), int(pair[1])) for pair in class_id_map) if class_id_map else None
        ),
        base_model_id=payload.get("base_model_id"),
        val_split_mode=payload.get("val_split_mode", "group"),
        val_group_buffer=payload.get("val_group_buffer"),
        val_area=payload.get("val_area"),
        val_per_class=(
            tuple((int(pair[0]), float(pair[1])) for pair in val_per_class)
            if val_per_class
            else None
        ),
    )


# Task of a model backend (section 3.1 of the plan).
TASK_SEMANTIC_SEGMENTATION = "semantic_segmentation"
TASK_INSTANCE_DETECTION = "instance_detection"
TASK_INSTANCE_SEGMENTATION = "instance_segmentation"
