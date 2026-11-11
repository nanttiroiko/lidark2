"""Evaluation facade: detection quality against evaluation annotations.

Replaces the original ``Evaluate`` class with the same six counters -
``Predicted / Fully detected (>= 0.9) / Partly detected / Not detected /
Detected with no label / False detected`` - computed per class and summed
into an overall row, so the original report is exactly the single-class
case. Evaluation consumes the Detection IR, so it works for any backend.

Candidate pairing is done with ``sjoin`` (STRtree) and the pairwise geometry
intersections run only on candidate pairs (7, P11). Two documented
deviations from the original, per the compatibility note: fully/partly
detected are counted **per label** (the original counted candidate pairs,
double-counting labels touched by several predictions), and false detections
are counted **per prediction** (the original counted pairs).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from lidark._types import ClassSpec
from lidark.config import ModelConfig, ProjectLayout, load_model_config
from lidark.logging import get_logger
from lidark.models.base import CONFIG_FILENAME

LOGGER = get_logger("evaluate")

COUNTERS: tuple[str, ...] = (
    "Predicted",
    "Fully detected",
    "Partly detected",
    "Not detected",
    "Detected with no label",
    "False detected",
)
FULL_RATIO = 0.9


def evaluate(
    layout: ProjectLayout,
    true_objects: str,
    model_id: str,
    eval_name: str,
    *,
    false_objects: str | None = None,
    pred_type: str = "all",
    eval_aoi: str | None = None,
    size_cut: ClassSpec = 1.0,
    prob_cut: ClassSpec = 0.8,
    full_ratio: float = FULL_RATIO,
) -> pd.DataFrame:
    """Evaluate a prediction against evaluation annotations.

    Reads the merged Detection IR from
    ``DeepLearning/model/<ts>/predictions/*/merged_shapes_<pred_type>.gpkg``,
    the labels from ``DeepLearning/evaluate_annotations/<eval_name>/``, clips
    the predictions to the AOI (from ``eval_aoi`` or the label bounding box),
    applies the ``size_cut``/``prob_cut`` filters and the ``buffer(0)``
    geometry repair, computes the six counters per class plus an overall row,
    and writes the report to
    ``DeepLearning/reports/<model_id>/<eval_name>.csv``.

    Args:
        layout: Project layout.
        true_objects: True-label shapefile (optional ``class_id`` column,
            default class 1).
        model_id: The model id (``<name>_<timestamp>``) whose predictions
            are evaluated.
        eval_name: The evaluation set name (annotations subfolder and report
            name).
        false_objects: Optional false-object shapefile; predictions
            overlapping them count as ``False detected``.
        pred_type: Which merged shapes to read (``all``/``test``).
        eval_aoi: Optional AOI shapefile; defaults to the label bounding box.
        size_cut: Drop predictions smaller than this area (m2; original
            1), or a ``{class_id: area}`` mapping for per-class cuts
            (plan 3.3/MC5).
        prob_cut: Drop predictions scoring below this (original 0.8), or a
            ``{class_id: score}`` mapping for per-class cuts.
        full_ratio: Overlap ratio above which a label counts as fully
            detected (original 0.9).

    Returns:
        The report: one row per class plus an ``overall`` row, with the six
        counters as columns. A confusion report (plan 3.3/MC6) is written
        next to it as ``<eval_name>_confusion.csv``.

    Raises:
        FileNotFoundError: If annotations or merged predictions are missing.
    """
    annotations = layout.evaluate_annotations_dir(eval_name)
    true = _read_labels(annotations / true_objects, default_class=1)
    false = _read_labels(annotations / false_objects, default_class=1) if false_objects else None
    # Class names and the compact-id map come from the model's config
    # (plan 3.3/MC4/MC2): label ids live in the user's original id space,
    # predictions carry the model's compact ids.
    model_config = _load_model_config(layout, model_id)
    class_map = (
        dict(model_config.class_id_map) if model_config and model_config.class_id_map else {}
    )
    true = _remap_label_classes(true, class_map)
    aoi = _read_aoi(annotations, eval_aoi, true, false)

    # All areas' merged Detection IR.
    pattern = f"merged_shapes_{pred_type}.gpkg"
    pred_files = sorted(layout.predictions_dir(model_id).glob(f"*/{pattern}"))
    if not pred_files:
        raise FileNotFoundError(
            f"no merged prediction shapes ({pattern}) under {layout.predictions_dir(model_id)}"
        )
    preds = gpd.GeoDataFrame(
        pd.concat([gpd.read_file(path) for path in pred_files], ignore_index=True)
    )
    preds.geometry = preds.geometry.buffer(0)
    true.geometry = true.geometry.buffer(0)
    if false is not None:
        false.geometry = false.geometry.buffer(0)

    preds = gpd.clip(preds, aoi)
    preds = preds.loc[_cut_mask(preds, preds.geometry.area, size_cut, "size_cut")]
    preds = preds.loc[_cut_mask(preds, preds["score"], prob_cut, "prob_cut")]

    all_labels = [true]
    if false is not None:
        all_labels.append(false)

    rows: list[dict[str, Any]] = []
    for class_id in sorted(true["class_id"].unique()):
        true_c = true.loc[true["class_id"] == class_id].reset_index(drop=True)
        preds_c = preds.loc[preds["class_id"] == class_id].reset_index(drop=True)
        rows.append(
            {
                "class_id": class_id,
                "class_name": _class_name(model_config, int(class_id)),
                **_count_stats(preds_c, true_c, all_labels, false, full_ratio),
            }
        )
    overall = {
        "class_id": "overall",
        "class_name": "overall",
        **{counter: sum(row[counter] for row in rows) for counter in COUNTERS},
    }
    report = pd.DataFrame([*rows, overall])

    report_dir = layout.reports_dir(model_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{eval_name}.csv"
    report.to_csv(report_path, index=False)
    _write_confusion(layout, model_id, eval_name, preds, true, model_config)
    LOGGER.info("evaluation report written: %s", report_path)
    return report


def _count_stats(
    preds: gpd.GeoDataFrame,
    true: gpd.GeoDataFrame,
    all_labels: Sequence[gpd.GeoDataFrame],
    false: gpd.GeoDataFrame | None,
    full_ratio: float,
) -> dict[str, int]:
    """The six counters for one class (candidates via sjoin, 7/P11)."""
    num_predicted = len(preds)
    if num_predicted == 0:
        return {
            "Predicted": 0,
            "Fully detected": 0,
            "Partly detected": 0,
            "Not detected": len(true),
            "Detected with no label": 0,
            "False detected": 0,
        }

    # Fully / partly / not detected: best overlap ratio per true label.
    pairs = _pair_overlap(preds, true)
    true_areas = true.geometry.area.to_numpy()
    best: dict[int, float] = {}
    for _d_idx, t_idx, area in pairs:
        best[t_idx] = max(best.get(t_idx, 0.0), area)
    fully = partly = 0
    for t_idx, area in best.items():
        ratio = area / max(true_areas[t_idx], 1e-12)
        if ratio >= full_ratio:
            fully += 1
        else:
            partly += 1
    not_detected = len(true) - fully - partly

    # Labeled = overlapping (area > 0) any true label of any class or any
    # false object; boundary-only touch counts as unlabeled, as in the
    # original difference-based ratio.
    labeled: set[int] = set()
    for labels_frame in all_labels:
        for d_idx, _t_idx, area in _pair_overlap(preds, labels_frame):
            if area > 0:
                labeled.add(d_idx)
    not_labeled = num_predicted - len(labeled)

    false_detected = 0
    if false is not None:
        false_dets = {d_idx for d_idx, _t, area in _pair_overlap(preds, false) if area > 0}
        false_detected = len(false_dets)

    return {
        "Predicted": num_predicted,
        "Fully detected": fully,
        "Partly detected": partly,
        "Not detected": not_detected,
        "Detected with no label": not_labeled,
        "False detected": false_detected,
    }


def _pair_overlap(
    preds: gpd.GeoDataFrame, labels: gpd.GeoDataFrame
) -> list[tuple[int, int, float]]:
    """Candidate (pred_idx, label_idx, intersection_area) pairs via sjoin."""
    if len(labels) == 0:
        return []
    left = preds.reset_index(drop=True)
    right = labels.reset_index(drop=True)
    if len(left) == 0:
        return []
    joined = gpd.sjoin(left, right, predicate="intersects", how="inner")
    pairs: list[tuple[int, int, float]] = []
    for left_pos, row in joined.iterrows():
        right_pos = int(row["index_right"])
        area = float(left.geometry.iloc[left_pos].intersection(right.geometry.iloc[right_pos]).area)
        pairs.append((int(left_pos), right_pos, area))
    return pairs


def _load_model_config(layout: ProjectLayout, model_id: str) -> ModelConfig | None:
    """The model's config, or ``None`` when it is missing (legacy models)."""
    path = layout.model_dir(model_id) / CONFIG_FILENAME
    if not path.exists():
        return None
    return load_model_config(path)


def _remap_label_classes(frame: gpd.GeoDataFrame, class_map: dict[int, int]) -> gpd.GeoDataFrame:
    """Map original label ids to the model's compact ids (plan 3.3/MC2).

    Unknown ids (classes the model never saw) keep their id: they simply
    pair with no predictions and surface as ``Not detected`` rows."""
    if not class_map:
        return frame
    ids = frame["class_id"].astype(int)
    unknown = sorted(set(ids.unique()) - set(class_map))
    if unknown:
        LOGGER.warning("class ids %s not in the model's class_id_map; kept as-is", unknown)
    return frame.assign(class_id=ids.map(lambda c: class_map.get(int(c), int(c))))


def _class_name(model_config: ModelConfig | None, class_id: int) -> str:
    """Class label from the model config (plan 3.3/MC4); ``class_<id>`` for
    legacy models or ids outside the config's names."""
    if model_config is not None:
        try:
            return model_config.class_name(class_id)
        except ValueError:
            pass
    return f"class_{class_id}"


def _cut_mask(preds: gpd.GeoDataFrame, values: pd.Series, cut: ClassSpec, kind: str) -> pd.Series:
    """Per-class cut mask (plan 3.3/MC5): a scalar applies to every class
    (original behavior); a mapping must cover every predicted class id."""
    if isinstance(cut, Mapping):
        classes = preds["class_id"].astype(int)
        missing = sorted(set(classes.unique()) - {int(c) for c in cut})
        if missing:
            raise ValueError(f"no {kind} for class_id(s) {missing}")
        limits = classes.map(lambda c: float(cut[int(c)]))
        return values >= limits
    return values >= float(cut)


def _write_confusion(
    layout: ProjectLayout,
    model_id: str,
    eval_name: str,
    preds: gpd.GeoDataFrame,
    true: gpd.GeoDataFrame,
    model_config: ModelConfig | None,
) -> None:
    """Confusion report (plan 3.3/MC6): per (true class, class of the
    best-overlap prediction) counts for detected labels, plus predictions
    overlapping no true label grouped by predicted class (true side
    ``none``). Written next to the main report as
    ``<eval_name>_confusion.csv``."""
    counts: Counter[tuple[Any, int]] = Counter()
    pairs = _pair_overlap(preds, true)
    best: dict[int, tuple[float, int]] = {}
    for d_idx, t_idx, area in pairs:
        if area > 0 and (t_idx not in best or area > best[t_idx][0]):
            best[t_idx] = (area, d_idx)
    for t_idx, (_area, d_idx) in best.items():
        true_id = int(true["class_id"].iloc[t_idx])
        pred_id = int(preds["class_id"].iloc[d_idx])
        counts[(true_id, pred_id)] += 1
    detected = {d_idx for d_idx, _t_idx, area in pairs if area > 0}
    for d_idx in range(len(preds)):
        if d_idx not in detected:
            counts[("none", int(preds["class_id"].iloc[d_idx]))] += 1
    rows = []
    for (true_id, pred_id), count in sorted(
        counts.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])
    ):
        rows.append(
            {
                "true_class_id": true_id,
                "true_class_name": (
                    "none" if true_id == "none" else _class_name(model_config, int(true_id))
                ),
                "pred_class_id": pred_id,
                "pred_class_name": _class_name(model_config, int(pred_id)),
                "count": count,
            }
        )
    columns = [
        "true_class_id",
        "true_class_name",
        "pred_class_id",
        "pred_class_name",
        "count",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    out = layout.reports_dir(model_id) / f"{eval_name}_confusion.csv"
    frame.to_csv(out, index=False)
    LOGGER.info("confusion report written: %s", out)


def _read_labels(path: Path, *, default_class: int) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"label file not found: {path}")
    frame = gpd.read_file(path)
    if "class_id" in frame.columns:
        frame = frame.assign(class_id=frame["class_id"].astype(int))
    else:
        frame = frame.assign(class_id=default_class)
    return frame.reset_index(drop=True)[["class_id", "geometry"]]


def _read_aoi(
    annotations: Path,
    eval_aoi: str | None,
    true: gpd.GeoDataFrame,
    false: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame:
    """AOI from file or the bounding box of all labels (original default)."""
    if eval_aoi is not None:
        aoi_path = annotations / eval_aoi
        if not aoi_path.exists():
            raise FileNotFoundError(f"AOI file not found: {aoi_path}")
        frame = gpd.read_file(aoi_path)
        return gpd.GeoDataFrame(geometry=[frame.union_all()], crs=frame.crs)
    frames = [true, false] if false is not None else [true]
    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=true.crs)
    bounds = combined.total_bounds
    return gpd.GeoDataFrame(geometry=[box(*bounds)], crs=true.crs)


__all__ = ["evaluate", "COUNTERS", "FULL_RATIO"]
