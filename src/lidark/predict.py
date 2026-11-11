"""Prediction facade: dense tile-and-stitch inference into the Detection IR.

Replaces the original ``Predict`` class. The backend is loaded through the
registry from the model's ``model_config.json`` (``feature_list`` read from
it unless overridden); tiles come from the feature product's tile index;
windows are processed in batches (7, P8) and blended by per-pixel maximum
(the original max-probability stitching); the thresholded masks are
polygonized (connectivity 4, 0.25 simplify), scored with the one-pass
``polygon_scores`` (7, P9, ``independent`` default), converted to the
Detection IR, and merged per area with ``merge_overlapping``. Per-tile
Detection IR accumulates in memory - per-tile GeoPackages only on request
(7, P10). Per-tile failures are logged and skipped, not swallowed.

Instance backends (Mask R-CNN, Faster R-CNN) take the **sparse path**: no
score rasters or stitching - each window's instances/detections are
converted straight into the Detection IR (masks polygonized in window
coordinates, so the sliding-window duplicates resolve in the area merge;
Faster R-CNN's boxes arrive as box-footprint masks and polygonize into
the box rectangles), and the per-tile raster output becomes the class-id
footprint of the predicted instances.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from tqdm import tqdm

from lidark._types import Array, ClassSpec
from lidark.config import (
    TASK_INSTANCE_DETECTION,
    TASK_INSTANCE_SEGMENTATION,
    ProjectLayout,
    load_model_config,
)
from lidark.io.detections import empty_detections, make_detections, validate_detections
from lidark.io.products import ensure_tile_index, ensure_vrt
from lidark.io.rasters import write_band
from lidark.io.tiling import WindowTiler
from lidark.io.vectors import (
    OverlapStrategy,
    merge_overlapping,
    polygon_scores,
    polygonize_mask,
    read_polygon_union,
)
from lidark.io.vrt import open_vrt
from lidark.logging import get_logger
from lidark.models.base import CONFIG_FILENAME
from lidark.models.registry import create_backend

if TYPE_CHECKING:
    from lidark.config import ModelConfig

LOGGER = get_logger("predict")

PRED_TYPES = ("all", "test")


def predict(
    layout: ProjectLayout,
    project_areas: Sequence[str],
    model_id: str,
    *,
    pred_type: str = "all",
    test_tiles: Sequence[str] = (),
    test_area: str | None = None,
    feature_list: Sequence[str] | None = None,
    write_tile_shape_files: bool = False,
    threshold: ClassSpec = 0.5,
    pred_batch_size: int | None = None,
    pred_stride: int | None = None,
    overlap: OverlapStrategy = "independent",
) -> dict[str, gpd.GeoDataFrame]:
    """Predict detections for one or more project areas with a trained model.

    Args:
        layout: Project layout.
        project_areas: Areas to predict.
        model_id: Trained model id (``<name>_<timestamp>``, the directory
            under ``DeepLearning/model/``).
        pred_type: ``"all"`` (every tile) or ``"test"`` (only the held-out
            test tiles — selected by ``test_area`` when given, otherwise by
            the ``test_tiles`` tile-ID patterns, the original substring
            semantics).
        test_tiles: Tile ID patterns for ``pred_type="test"`` (used only
            when no ``test_area`` is given).
        test_area: Test-area polygon layer (e.g. a ``.gpkg`` file under
            ``DeepLearning/train_annotations/``, the same convention as in
            training): for ``pred_type="test"``, exactly the tiles whose
            footprint intersects the polygons are predicted.
        feature_list: Override the model's recorded feature layers (must
            match the model's input count); default comes from
            ``model_config.json``.
        write_tile_shape_files: Also write per-tile GeoPackage vector files
            (the original always wrote per-tile shapefiles; off by default
            per 7, P10).
        threshold: Detection threshold (original: 0.5), or a
            ``{class_id: threshold}`` mapping for per-class operating points
            (plan 3.3/MC5); a mapping must cover every class of the model.
        pred_batch_size: Windows per forward pass (default from the model
            config, 32).
        pred_stride: Window stride (default from the model config, 128 - the
            original ``window / 4``).
        overlap: ``polygon_scores`` overlap strategy (7.1).

    Returns:
        ``{area: merged Detection IR}`` per area.

    Raises:
        ValueError: For an unknown ``pred_type``, a ``"test"`` run without a
            tile selector, or a missing feature list.
        FileNotFoundError: If the model does not exist.
    """
    if pred_type not in PRED_TYPES:
        raise ValueError(f"pred_type must be one of {PRED_TYPES}, got {pred_type!r}")
    if pred_type == "test" and test_area is None and not tuple(test_tiles):
        raise ValueError("pred_type='test' requires test_area (polygon layer) or test_tiles")
    test_mask = (
        read_polygon_union(layout.train_annotations_dir / test_area)
        if test_area is not None
        else None
    )
    model_dir = layout.model_dir(model_id)
    config_path = model_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"model {model_id!r} not found ({config_path})")
    config = load_model_config(config_path)
    features = tuple(feature_list) if feature_list is not None else config.feature_list
    if not features:
        raise ValueError("no feature_list: pass one or train with one recorded")
    # Validate per-class thresholds up front: a missing class would
    # otherwise surface as a logged per-tile failure, not a clear error.
    _thresholds_per_class(threshold, config.num_classes)

    backend = create_backend(config.backend)
    backend.load(model_dir)

    stride = pred_stride if pred_stride is not None else config.pred_stride
    batch = pred_batch_size if pred_batch_size is not None else config.pred_batch_size
    tiler = WindowTiler(config.img_size, stride=stride)

    results: dict[str, gpd.GeoDataFrame] = {}
    for area in project_areas:
        results[area] = _predict_area(
            layout,
            area,
            features,
            backend,
            config,
            model_id,
            pred_type,
            test_tiles,
            test_mask,
            tiler,
            batch,
            threshold,
            overlap,
            write_tile_shape_files,
        )
    return results


def _predict_area(
    layout: ProjectLayout,
    area: str,
    features: tuple[str, ...],
    backend: Any,
    config: ModelConfig,
    model_id: str,
    pred_type: str,
    test_tiles: Sequence[str],
    test_mask: BaseGeometry | None,
    tiler: WindowTiler,
    batch: int,
    threshold: ClassSpec,
    overlap: OverlapStrategy,
    write_tile_shapes: bool,
) -> gpd.GeoDataFrame:
    """Predict one area; returns the merged Detection IR."""
    reference_dir = layout.lidar_area(area) / features[0]
    index = ensure_tile_index(reference_dir)
    tile_ids = [str(tile_id) for tile_id in index["tile_id"]]
    if pred_type == "test":
        if test_mask is not None:
            tile_ids = [t for t in tile_ids if _footprint(index, t).intersects(test_mask)]
        else:
            tile_ids = [t for t in tile_ids if any(pattern in t for pattern in test_tiles)]
        if not tile_ids:
            LOGGER.warning("no tiles selected for pred_type='test' in area %s", area)

    raster_dir = layout.prediction_raster_dir(model_id, area)
    raster_dir.mkdir(parents=True, exist_ok=True)
    shapes_dir = layout.prediction_shapes_dir(model_id, area)
    if write_tile_shapes:
        shapes_dir.mkdir(parents=True, exist_ok=True)

    detections: list[gpd.GeoDataFrame] = []
    for tile_id in tqdm(tile_ids, desc=f"predict {area}", unit="tile", disable=None):
        raster_path = raster_dir / f"{tile_id}.tif"
        if raster_path.exists():
            LOGGER.info("skipping %s/%s: raster already exists", area, tile_id)
            continue
        try:
            footprint = _footprint(index, tile_id)
            tile_ir = _predict_tile(
                layout,
                area,
                features,
                backend,
                config,
                footprint,
                tiler,
                batch,
                threshold,
                overlap,
                raster_path,
                shapes_dir if write_tile_shapes else None,
                tile_id,
            )
        except Exception:  # noqa: BLE001 - per-tile isolation, like the
            # original's bare except, but logged with the traceback.
            LOGGER.exception("could not predict tile %s/%s", area, tile_id)
            continue
        if len(tile_ir) > 0:
            detections.append(tile_ir)

    if detections:
        merged = merge_overlapping(gpd.GeoDataFrame(pd.concat(detections, ignore_index=True)))
    else:
        merged = empty_detections(layout.epsg)
    merged_path = layout.predictions_dir(model_id, area) / f"merged_shapes_{pred_type}.gpkg"
    if len(merged) > 0:
        merged.to_file(merged_path)
    else:
        LOGGER.warning("no detections in %s; merged shapes not written: %s", area, merged_path)
    return merged


def _predict_tile(
    layout: ProjectLayout,
    area: str,
    features: tuple[str, ...],
    backend: Any,
    config: ModelConfig,
    footprint: BaseGeometry,
    tiler: WindowTiler,
    batch: int,
    threshold: ClassSpec,
    overlap: OverlapStrategy,
    raster_path: Path,
    shapes_dir: Path | None,
    tile_id: str,
) -> gpd.GeoDataFrame:
    """Inference on one tile: features -> windows -> stitched scores -> IR."""
    stack, transform, crs = _read_tile_features(layout, area, features, footprint)
    if backend.task in (TASK_INSTANCE_SEGMENTATION, TASK_INSTANCE_DETECTION):
        # Sparse path: instances/detections straight into the IR, no score
        # stitching. Detection backends deliver boxes as footprint masks,
        # which polygonize into the box rectangles.
        return _predict_tile_instances(
            layout,
            backend,
            config,
            stack,
            transform,
            crs,
            tiler,
            batch,
            threshold,
            raster_path,
            shapes_dir,
            tile_id,
        )
    height, width = stack.shape[0], stack.shape[1]
    channels = 1 if config.num_classes == 1 else config.num_classes
    prob = np.zeros((channels, height, width), dtype=np.float32)

    items = list(tiler.iter_windows(stack))
    for start in tqdm(
        range(0, len(items), batch),
        desc=f"{tile_id} windows",
        unit="batch",
        disable=None,  # bars on terminals, silent under pytest/redirects
    ):
        chunk = items[start : start + batch]
        windows = np.stack([np.asarray(window, dtype=np.float32) for _center, window in chunk])
        scores = np.asarray(backend.predict_tiles(windows))
        if scores.ndim == 3:  # binary: (B, H, W)
            scores = scores[..., None]
        for channel in range(channels):
            tiler.stitch_into(
                prob[channel],
                [(center, scores[i, :, :, channel]) for i, (center, _window) in enumerate(chunk)],
            )

    thresholds = _thresholds_per_class(threshold, config.num_classes)
    masks, raster_mask = _dense_class_masks(prob, thresholds)
    write_band(raster_path, raster_mask, transform=transform, crs=crs, nodata=None)

    crs_obj: CRS | str | None = crs if crs is not None else layout.epsg
    detections: list[gpd.GeoDataFrame] = []
    for channel in range(channels):
        if not masks[channel].any():
            continue
        polygons = polygonize_mask(
            masks[channel].astype(np.uint8), transform=transform, crs=crs_obj
        )
        if len(polygons) == 0:
            continue
        scores = polygon_scores(polygons, prob[channel], transform, overlap=overlap)
        class_id = channel + 1
        detections.append(
            make_detections(
                list(polygons.geometry),
                scores,
                class_ids=class_id,
                class_names=config.class_name(class_id),
                crs=crs_obj,
            )
        )
    if not detections:
        return empty_detections(crs_obj)
    tile_ir = gpd.GeoDataFrame(pd.concat(detections, ignore_index=True))
    validate_detections(tile_ir)
    if shapes_dir is not None:
        tile_ir.to_file(shapes_dir / f"{tile_id}.gpkg")
    return tile_ir


def _predict_tile_instances(
    layout: ProjectLayout,
    backend: Any,
    config: ModelConfig,
    stack: Array,
    transform: Affine,
    crs: CRS | None,
    tiler: WindowTiler,
    batch: int,
    threshold: ClassSpec,
    raster_path: Path,
    shapes_dir: Path | None,
    tile_id: str,
) -> gpd.GeoDataFrame:
    """Sparse path (instance segmentation): windows -> instances -> IR.

    Each window's instance masks are polygonized in window coordinates
    (window transform = tile transform shifted to the window origin), so
    the geometries land in the tile's CRS directly. The per-tile raster
    output is the class-id footprint of the predicted instances; the
    sliding-window duplicates of one object overlap almost fully and
    dissolve in the area-level IR merge.
    """
    height, width = stack.shape[0], stack.shape[1]
    crs_obj: CRS | str | None = crs if crs is not None else layout.epsg
    raster_mask = np.zeros((height, width), dtype=np.uint8)
    detections: list[gpd.GeoDataFrame] = []
    half_h, half_w = tiler.half
    items = list(tiler.iter_windows(stack))
    for start in tqdm(
        range(0, len(items), batch),
        desc=f"{tile_id} windows",
        unit="batch",
        disable=None,
    ):
        chunk = items[start : start + batch]
        windows = np.stack([np.asarray(window, dtype=np.float32) for _center, window in chunk])
        per_window = backend.predict_instances(windows)
        for (center, _window), instances in zip(chunk, per_window, strict=True):
            row0, col0 = center[0] - half_h, center[1] - half_w
            window_transform = transform * Affine.translation(col0, row0)
            for instance in instances:
                if isinstance(threshold, Mapping):
                    if int(instance["class_id"]) not in threshold:
                        raise ValueError(f"no threshold for class_id {int(instance['class_id'])}")
                    cls_threshold = float(threshold[int(instance["class_id"])])
                else:
                    cls_threshold = float(threshold)
                if float(instance["score"]) < cls_threshold:
                    continue
                mask = np.asarray(instance["mask"], dtype=np.uint8)
                if not mask.any():
                    continue
                polygons = polygonize_mask(mask, transform=window_transform, crs=crs_obj)
                if len(polygons) == 0:
                    continue
                geometry = unary_union(list(polygons.geometry))
                class_id = int(instance["class_id"])
                # Per-tile class-id raster: the instance footprints.
                region = raster_mask[row0 : row0 + mask.shape[0], col0 : col0 + mask.shape[1]]
                region[mask > 0] = class_id
                detections.append(
                    make_detections(
                        [geometry],
                        float(instance["score"]),
                        class_ids=class_id,
                        class_names=config.class_name(class_id),
                        crs=crs_obj,
                    )
                )
    write_band(raster_path, raster_mask, transform=transform, crs=crs, nodata=None)
    if not detections:
        return empty_detections(crs_obj)
    tile_ir = gpd.GeoDataFrame(pd.concat(detections, ignore_index=True))
    validate_detections(tile_ir)
    if shapes_dir is not None:
        tile_ir.to_file(shapes_dir / f"{tile_id}.gpkg")
    return tile_ir


def _thresholds_per_class(threshold: ClassSpec, num_classes: int) -> list[float]:
    """Per-class thresholds (plan 3.3/MC5): a scalar broadcasts to every
    class (original behavior); a mapping must cover class ids 1..C."""
    if isinstance(threshold, Mapping):
        missing = [c for c in range(1, num_classes + 1) if c not in threshold]
        if missing:
            raise ValueError(f"no threshold for class_id(s) {missing}")
        return [float(threshold[c]) for c in range(1, num_classes + 1)]
    return [float(threshold)] * num_classes


def _dense_class_masks(prob: Array, thresholds: Sequence[float]) -> tuple[Array, Array]:
    """Argmax-constrained class masks (plan 3.3/MC1).

    A pixel belongs to its highest-scoring class only, and only when that
    class's own probability passes the class threshold: the vector IR is
    cut from the same argmax decision as the per-tile class-id raster, so
    the two agree and no pixel reaches two classes' polygons. A single
    channel reproduces the original binary mask exactly.

    Returns:
        ``(masks, raster_mask)`` — per-channel polygonization masks (a
        pixel appears in at most one channel) and the per-pixel class-id
        map (0 = background).
    """
    prob = np.asarray(prob)
    channels = prob.shape[0]
    masks = prob > np.asarray(thresholds, dtype=np.float32)[:, None, None]
    if channels == 1:
        return masks, masks[0].astype(np.uint8)
    best = np.argmax(prob, axis=0)
    rows, cols = np.indices(prob.shape[1:])
    passes = masks[best, rows, cols]
    raster_mask = np.where(passes, best + 1, 0).astype(np.uint8)
    masks = masks & (best[None, :, :] == np.arange(channels)[:, None, None])
    return masks, raster_mask


def _read_tile_features(
    layout: ProjectLayout,
    area: str,
    features: tuple[str, ...],
    footprint: BaseGeometry,
) -> tuple[Array, Affine, CRS | None]:
    """Read the feature bands of one tile, pixel-aligned (plan B6).

    Each feature layer's tile index is ensured first (rebuilt from the
    tiles on disk when missing or stale), then its VRT mosaic — present
    layers are read through the mosaic (one windowed read per layer),
    falling back to the matching tile raster when no mosaic could be
    built. All layers must share the reference tile's transform and shape.
    """
    bands: list[Array] = []
    transform: Affine | None = None
    crs: CRS | None = None
    shape: tuple[int, ...] | None = None
    for feature in features:
        feature_dir = layout.lidar_area(area) / feature
        feature_index = ensure_tile_index(feature_dir)
        try:
            vrt = ensure_vrt(feature_dir, feature_index)
        except (FileNotFoundError, RuntimeError) as exc:
            LOGGER.warning(
                "could not build the VRT mosaic of %s (%s); reading tiles individually",
                feature_dir,
                exc,
            )
            vrt = None
        if vrt is not None:
            with open_vrt(vrt) as mosaic:
                window = _footprint_window(footprint.bounds, mosaic.transform)
                band = np.asarray(mosaic.read(1, window=window, boundless=True))
                band_transform = mosaic.transform * Affine.translation(
                    window.col_off, window.row_off
                )
                band_crs = mosaic.crs
        else:
            path = _matching_tile(feature_index, footprint)
            with rasterio.open(path) as src:
                band = np.asarray(src.read(1))
                band_transform = src.transform
                band_crs = src.crs
        if transform is None:
            transform, crs, shape = band_transform, band_crs, band.shape
        else:
            if band_transform != transform or band.shape != shape:
                raise ValueError(
                    f"feature {feature!r} is not pixel-aligned with the reference layer (B6)"
                )
        bands.append(band)
    assert transform is not None
    return np.stack(bands, axis=-1), transform, crs


def _footprint_window(bounds: tuple[float, float, float, float], transform: Affine) -> Window:
    """Integer pixel window covering the footprint, snapped outward."""
    minx, miny, maxx, maxy = bounds
    window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=transform)
    col0 = int(math.floor(window.col_off))
    row0 = int(math.floor(window.row_off))
    col1 = int(math.ceil(window.col_off + window.width))
    row1 = int(math.ceil(window.row_off + window.height))
    return Window(col0, row0, col1 - col0, row1 - row0)


def _matching_tile(feature_index: gpd.GeoDataFrame, footprint: BaseGeometry) -> Path:
    """Path of the feature tile with the largest overlap of the footprint."""
    hits = feature_index[feature_index.intersects(footprint)]
    if len(hits) == 0:
        raise ValueError("feature layer has no tile intersecting the footprint")
    if len(hits) > 1:
        areas = hits.geometry.intersection(footprint).area
        row = hits.loc[areas.idxmax()]
    else:
        row = hits.iloc[0]
    return Path(str(row["path"]))


def _footprint(index: gpd.GeoDataFrame, tile_id: str) -> BaseGeometry:
    rows = index[index["tile_id"] == tile_id]
    if len(rows) == 0:
        raise KeyError(f"tile_id {tile_id!r} missing from tile index")
    return rows.geometry.iloc[0]


__all__ = ["predict", "PRED_TYPES"]
