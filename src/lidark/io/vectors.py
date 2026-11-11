"""Vector helpers: polygonization, one-pass polygon scoring, overlap merging.

These functions replace the two most fragile parts of the original code
(MODERNIZATION_PLAN.md B4, B5): per-polygon zonal statistics and the broken
cross-mapsheet merge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.windows
import scipy.ndimage
import shapely
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry.base import BaseGeometry

from lidark._types import Array
from lidark.io.detections import COLUMNS

# How polygon_scores() handles polygons that share area (plan section 7.1).
OverlapStrategy: TypeAlias = Literal["independent", "exclusive", "priority"]
_OVERLAP_STRATEGIES: tuple[str, ...] = ("independent", "exclusive", "priority")


def read_polygon_union(path: Path | str) -> BaseGeometry:
    """Union of all geometries in a polygon layer (e.g. a ``.gpkg`` file).

    Used by the **test-area** feature: training excludes tiles whose
    footprint intersects the polygons; prediction selects them for
    ``pred_type="test"``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the layer contains no geometries.
    """
    out_path = Path(path)
    if not out_path.exists():
        raise FileNotFoundError(f"polygon layer not found: {out_path}")
    frame = gpd.read_file(out_path)
    if len(frame) == 0:
        raise ValueError(f"polygon layer {out_path.name!r} contains no geometries")
    return frame.geometry.union_all()


def polygonize_mask(
    mask: Array,
    *,
    transform: Affine,
    crs: CRS | str | None,
    value: int = 1,
    connectivity: int = 4,
    simplify_tolerance: float = 0.25,
) -> gpd.GeoDataFrame:
    """Polygonize the cells of ``mask`` equal to ``value``.

    Args:
        mask: 2-D integer array (e.g. a thresholded detection mask).
        transform: Affine transform of ``mask``.
        crs: CRS written to the output frame.
        value: Cell value to polygonize (original: 1).
        connectivity: 4- or 8-connected regions (original: 4).
        simplify_tolerance: Geometry simplification tolerance in world units
            (original: 0.25); ``0`` disables simplification.

    Returns:
        GeoDataFrame of one (simplified) polygon per connected region.
    """
    shapes = cast(
        "list[tuple[dict[str, object], int]]",
        list(rasterio.features.shapes(mask, connectivity=connectivity, transform=transform)),
    )
    geoms = [shapely.geometry.shape(geom) for geom, val in shapes if val == value]
    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    polygons = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    if simplify_tolerance > 0:
        polygons.geometry = polygons.geometry.simplify(simplify_tolerance)
    return polygons


def polygon_scores(
    polygons: gpd.GeoDataFrame,
    values: Array,
    transform: Affine,
    overlap: OverlapStrategy = "independent",
) -> Array:
    """Mean of ``values`` inside each polygon.

    Replaces the original per-polygon ``rasterstats.zonal_stats`` loop
    (O(polygons x pixels)) with one rasterization plus one ``bincount``
    reduction (O(pixels)) — exact when the polygons are disjoint, which is the
    common case for dense-segmentation detections.

    Overlapping polygons (multi-class regions, detector backends, user
    labels) are handled per the ``overlap`` strategy (plan section 7.1):

    - ``"independent"`` (default): each polygon is scored over its **full
      geometry**; contested pixels count for every covering polygon — the
      original ``zonal_stats`` semantics. A detection's confidence reflects
      its whole footprint.
    - ``"priority"``: one-pass behavior — the polygon rasterized last claims
      contested pixels; earlier polygons' means exclude them. Fast but
      arbitrary; only meaningful for provably disjoint inputs.
    - ``"exclusive"``: contested pixels are assigned to the nearest
      uncontested region (Euclidean distance transform), giving every
      detection a well-defined "own" region. A polygon fully covered by
      others receives no pixels and scores ``NaN``.

    Adaptive dispatch: a cheap STRtree self-join detects whether any polygon
    pair shares area; disjoint inputs never leave the one-pass path.

    Args:
        polygons: Frame whose geometries define the regions.
        values: 2-D array sampled per pixel (e.g. the probability raster).
        transform: Affine transform of ``values``.
        overlap: Strategy for overlapping polygons (see above).

    Returns:
        Array of one mean per polygon; ``NaN`` for polygons that do not
        intersect the raster (or, under ``"exclusive"``, for polygons with
        no own pixels).

    Raises:
        ValueError: If ``overlap`` is not a known strategy.
    """
    if overlap not in _OVERLAP_STRATEGIES:
        raise ValueError(f"overlap must be one of {_OVERLAP_STRATEGIES}, got {overlap!r}")
    n = len(polygons)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if overlap == "priority" or not _has_area_overlap(polygons):
        return _one_pass_scores(polygons, values, transform)
    if overlap == "independent":
        return _independent_scores(polygons, values, transform)
    return _exclusive_scores(polygons, values, transform)


def _one_pass_scores(polygons: gpd.GeoDataFrame, values: Array, transform: Affine) -> Array:
    """Region-id rasterization + bincount; later polygons claim contested pixels."""
    n = len(polygons)
    region_ids = rasterio.features.rasterize(
        ((geom, i + 1) for i, geom in enumerate(polygons.geometry)),
        out_shape=values.shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )
    return _means_per_region(region_ids, values, n)


def _independent_scores(polygons: gpd.GeoDataFrame, values: Array, transform: Affine) -> Array:
    """Per-polygon means over bounding-box windows (full-footprint semantics)."""
    scores = np.full(len(polygons), np.nan)
    vals = np.asarray(values, dtype=np.float64)
    for i, geom in enumerate(polygons.geometry):
        window = _geometry_window(geom.bounds, transform, values.shape)
        if window is None:
            continue
        rows, cols, sub_transform = window
        mask = rasterio.features.rasterize(
            [(geom, 1)],
            out_shape=(rows.stop - rows.start, cols.stop - cols.start),
            transform=sub_transform,
            fill=0,
            dtype="uint8",
        )
        covered = mask == 1
        if covered.any():
            scores[i] = vals[rows, cols][covered].mean()
    return scores


def _exclusive_scores(polygons: gpd.GeoDataFrame, values: Array, transform: Affine) -> Array:
    """Contested pixels take the region id of the nearest uncontested pixel.

    Coverage is accumulated per polygon in bounding-box windows
    (O(sum of polygon areas)); one distance transform assigns contested
    pixels. Fully covered polygons end up with no pixels (NaN).
    """
    n = len(polygons)
    shape = values.shape
    region_ids = rasterio.features.rasterize(
        ((geom, i + 1) for i, geom in enumerate(polygons.geometry)),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )
    coverage = np.zeros(shape, dtype=np.int32)
    for geom in polygons.geometry:
        window = _geometry_window(geom.bounds, transform, shape)
        if window is None:
            continue
        rows, cols, sub_transform = window
        mask = rasterio.features.rasterize(
            [(geom, 1)],
            out_shape=(rows.stop - rows.start, cols.stop - cols.start),
            transform=sub_transform,
            fill=0,
            dtype="uint8",
        )
        coverage[rows, cols] += mask
    contested = coverage >= 2
    seeds = np.where(contested, 0, region_ids)
    # For each background pixel (0), distance_transform_edt returns the
    # indices of the nearest foreground (owned) pixel.
    _, (row_idx, col_idx) = scipy.ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    assigned = np.where(contested, seeds[row_idx, col_idx], region_ids)
    return _means_per_region(assigned.astype(np.int32), values, n)


def _means_per_region(region_ids: Array, values: Array, n: int) -> Array:
    """Mean of ``values`` per region id (1..n); NaN for absent regions."""
    flat_ids = region_ids.ravel()
    flat_vals = np.asarray(values, dtype=np.float64).ravel()
    valid = flat_ids > 0
    sums = np.bincount(flat_ids[valid], weights=flat_vals[valid], minlength=n + 1)
    counts = np.bincount(flat_ids[valid], minlength=n + 1)
    scores = np.full(n, np.nan)
    hit = counts[1:] > 0
    scores[hit] = sums[1:][hit] / counts[1:][hit]
    return scores


def _geometry_window(
    bounds: tuple[float, float, float, float],
    transform: Affine,
    shape: tuple[int, ...],
) -> tuple[slice, slice, Affine] | None:
    """Row/col slices covering a geometry's bounds, plus the window transform.

    Returns ``None`` when the bounds fall entirely outside the raster.
    """
    window = rasterio.windows.from_bounds(*bounds, transform=transform)
    row0 = max(int(np.floor(window.row_off)), 0)
    col0 = max(int(np.floor(window.col_off)), 0)
    row1 = min(int(np.ceil(window.row_off + window.height)), shape[0])
    col1 = min(int(np.ceil(window.col_off + window.width)), shape[1])
    if row0 >= row1 or col0 >= col1:
        return None
    sub_transform = transform * Affine.translation(col0, row0)
    return slice(row0, row1), slice(col0, col1), sub_transform


def _has_area_overlap(polygons: gpd.GeoDataFrame) -> bool:
    """True if any polygon pair shares area (touching boundaries do not count)."""
    pairs = np.asarray(polygons.sindex.query(polygons.geometry, predicate="intersects"))
    if pairs.ndim != 2:
        return False
    for left, right in pairs.T:
        i, j = int(left), int(right)
        if i == j:
            continue
        if polygons.geometry.iloc[i].intersection(polygons.geometry.iloc[j]).area > 0:
            return True
    return False


def merge_overlapping(detections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge overlapping detections, within each class.

    Reimplements the original cross-mapsheet merge with modern geopandas:
    union of all geometries (per class) -> overlay with the original
    detections -> dissolve per union piece. Detections from different classes
    never merge. Attribute values come from the first overlapping detection
    (the original ``dissolve`` default); ``area`` is recomputed.

    Args:
        detections: Detection IR frame.

    Returns:
        Detection IR frame with overlaps dissolved per class.
    """
    if len(detections) == 0:
        return detections.copy()
    merged: list[gpd.GeoDataFrame] = []
    for _, group in detections.groupby("class_id"):
        group = group.reset_index(drop=True)
        union_geom = group.geometry.union_all()
        parts = shapely.get_parts(union_geom)
        pieces = gpd.GeoDataFrame(
            {"geo_id": np.arange(len(parts))}, geometry=parts, crs=detections.crs
        )
        intersections = gpd.overlay(group, pieces, how="intersection")
        dissolved = intersections.dissolve(by="geo_id", aggfunc="first").reset_index()
        dissolved["area"] = dissolved.geometry.area
        merged.append(dissolved)
    out = gpd.GeoDataFrame(pd.concat(merged, ignore_index=True), crs=detections.crs)
    ordered = [c for c in (*COLUMNS, "geometry") if c in out.columns]
    return out[ordered]
