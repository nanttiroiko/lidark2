"""Detection IR: the interchange schema consumed by merge and evaluation.

Every ML backend converts its output (dense score rasters today, boxes/masks
for future detector backends) into this schema. It is a
:class:`geopandas.GeoDataFrame` with the columns in :data:`COLUMNS` plus a
geometry column: ``score`` (float, detection probability), ``class_id``
(int, >= 1), ``class_name`` (str), and ``area`` (float, computed from the
geometry). Downstream vector merge and evaluation accept nothing else.
"""

from __future__ import annotations

from collections.abc import Iterable

import geopandas as gpd
import numpy as np
from rasterio.crs import CRS
from shapely.geometry.base import BaseGeometry

COLUMNS: tuple[str, ...] = ("score", "class_id", "class_name", "area")


def make_detections(
    geometry: Iterable[BaseGeometry],
    scores: float | Iterable[float],
    class_ids: int | Iterable[int] = 1,
    class_names: str | Iterable[str] | None = None,
    *,
    crs: CRS | str | None = None,
) -> gpd.GeoDataFrame:
    """Build a Detection IR frame; ``area`` is computed from the geometries.

    Args:
        geometry: Detection geometries.
        scores: One score per detection, or a scalar broadcast to all.
        class_ids: Class ids (>= 1), or a scalar broadcast to all.
        class_names: Labels, a scalar, or ``None`` to derive ``class_<id>``.
        crs: CRS of the geometries.

    Returns:
        GeoDataFrame with columns ``score, class_id, class_name, area`` and
        geometry.
    """
    geoms = list(geometry)
    n = len(geoms)
    score_arr = np.broadcast_to(np.asarray(scores, dtype=np.float64), (n,)).copy()
    class_arr = np.broadcast_to(np.asarray(class_ids, dtype=np.int64), (n,)).copy()
    if np.any(class_arr < 1):
        raise ValueError("class_id must be >= 1")
    if class_names is None:
        name_arr = np.asarray([f"class_{int(c)}" for c in class_arr])
    elif isinstance(class_names, str):
        name_arr = np.broadcast_to(np.asarray(class_names), (n,)).copy()
    else:
        name_arr = np.broadcast_to(np.asarray(class_names, dtype=object), (n,)).copy()
    detections = gpd.GeoDataFrame(
        {
            "score": score_arr,
            "class_id": class_arr,
            "class_name": name_arr,
            "geometry": geoms,
        },
        crs=crs,
    )
    detections["area"] = detections.geometry.area
    return detections[[*COLUMNS, "geometry"]]


def validate_detections(detections: gpd.GeoDataFrame) -> None:
    """Validate the Detection IR schema.

    Raises:
        ValueError: Listing every schema violation found.
    """
    problems: list[str] = []
    for column in (*COLUMNS, "geometry"):
        if column not in detections.columns:
            problems.append(f"missing column {column!r}")
    if len(detections) == 0:
        if problems:
            raise ValueError("invalid Detection IR: " + "; ".join(problems))
        return
    if "class_id" in detections.columns and np.any(detections["class_id"].to_numpy() < 1):
        problems.append("class_id must be >= 1")
    if "score" in detections.columns:
        scores = detections["score"].to_numpy()
        if np.any(scores < 0) or np.any(scores > 1):
            problems.append("score must be within [0, 1]")
    if problems:
        raise ValueError("invalid Detection IR: " + "; ".join(problems))


def empty_detections(crs: CRS | str | None = None) -> gpd.GeoDataFrame:
    """Return an empty Detection IR frame with the schema columns in place."""
    detections = gpd.GeoDataFrame(
        {
            "score": np.empty(0, dtype=np.float64),
            "class_id": np.empty(0, dtype=np.int64),
            "class_name": np.empty(0, dtype=object),
            "area": np.empty(0, dtype=np.float64),
            "geometry": [],
        },
        crs=crs,
    )
    return detections


def is_empty(detections: gpd.GeoDataFrame) -> bool:
    """True when the frame carries no detections."""
    return len(detections) == 0


__all__ = [
    "COLUMNS",
    "empty_detections",
    "is_empty",
    "make_detections",
    "validate_detections",
]
