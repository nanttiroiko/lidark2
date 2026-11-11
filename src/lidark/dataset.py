"""Training dataset creation and storage (memmap-backed, class-aware).

Rebuilds the original ``create_samples`` with modern pieces: tiles are
discovered through the tile index (spatial label matching, never filename
prefixes), labels may carry an integer ``class_id`` column (missing column =
class 1, the original binary case; false objects contribute all-background
windows exactly like the original), observed class ids are remapped to
compact ids ``1..C`` so sparse label ids create no phantom classes
(plan 3.3/MC2), windows are placed with the original
2-per-object random sampling and extracted with the original 4-corner edge
fallback, every window records its placing object and map footprint (plus
the full bounds of every label object), and the dataset is written as a disk
memmap (``samples.dat`` + ``labels.dat`` + ``dataset_meta.json``) so training
streams per-index windows instead of copying the whole array into RAM (7, P5).
The recorded provenance lets training split object-disjointly: an object (or
a cluster of neighbouring objects) never appears on both sides of the
train/validation split.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from lidark.config import ProjectLayout
from lidark.io.products import ensure_tile_index, ensure_vrt
from lidark.io.tiles import tile_path
from lidark.io.vectors import read_polygon_union
from lidark.logging import get_logger

if TYPE_CHECKING:
    from lidark._types import Array

LOGGER = get_logger("dataset")

META_FILENAME = "dataset_meta.json"
SAMPLES_FILENAME = "samples.dat"
LABELS_FILENAME = "labels.dat"
SAMPLES_DTYPE = np.dtype("int8")
LABELS_DTYPE = np.dtype("uint8")


class Dataset:
    """A created or loaded dataset: memmap arrays plus metadata."""

    def __init__(
        self,
        dataset_id: str,
        directory: Path,
        samples: Array,
        labels: Array,
        meta: dict[str, Any],
    ) -> None:
        self.id = dataset_id
        self.dir = directory
        self.samples = samples
        self.labels = labels
        self.meta = meta

    @property
    def num_classes(self) -> int:
        return int(self.meta["num_classes"])

    @property
    def class_id_map(self) -> dict[int, int]:
        """Original label id -> compact id (identity when unmapped, MC2)."""
        pairs = self.meta.get("class_id_map") or []
        return {int(original): int(compact) for original, compact in pairs}

    @property
    def window_classes(self) -> list[int] | None:
        """Placing class per window (0 = background window), or ``None``
        for legacy datasets created before the field existed (MC7)."""
        recorded = self.meta.get("window_classes")
        return [int(c) for c in recorded] if recorded is not None else None

    @property
    def window_objects(self) -> list[str] | None:
        """Placing object id per window (``true_<row>`` / ``false_<row>``),
        or ``None`` for legacy datasets created before object provenance
        was recorded (needed by the object-disjoint validation split)."""
        recorded = self.meta.get("window_objects")
        return [str(o) for o in recorded] if recorded is not None else None

    @property
    def window_bounds(self) -> list[list[float]] | None:
        """Map-coordinate footprint (minx, miny, maxx, maxy) per window,
        or ``None`` for legacy datasets."""
        recorded = self.meta.get("window_bounds")
        return [[float(b) for b in bounds] for bounds in recorded] if recorded is not None else None

    @property
    def objects(self) -> dict[str, dict[str, Any]] | None:
        """Label objects that contributed windows: id ->
        ``{"class_id": int, "bounds": [minx, miny, maxx, maxy]}`` (the
        full, unclipped object bounds), or ``None`` for legacy datasets."""
        return self.meta.get("objects")

    @property
    def pixel_size(self) -> float | None:
        """Pixel size (m) of the reference product, or ``None`` for legacy
        datasets."""
        recorded = self.meta.get("pixel_size")
        return float(recorded) if recorded is not None else None


def dataset_dir(layout: ProjectLayout, dataset_id: str) -> Path:
    """Directory of one dataset (``DeepLearning/samples/<dataset_id>/``)."""
    return layout.samples_dir / dataset_id


def _ensure_product(feature_dir: Path) -> gpd.GeoDataFrame:
    """The product's tile index, with the VRT mosaic ensured best-effort.

    The index is required here (rebuilt from the tiles on disk when missing
    or stale); the VRT is a read optimization of predict, so a failed build
    (e.g. no ``gdalbuildvrt``) only warns — training itself reads tiles.
    """
    index = ensure_tile_index(feature_dir)
    try:
        ensure_vrt(feature_dir, index)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.warning(
            "could not build the VRT mosaic of %s (%s); continuing without it",
            feature_dir,
            exc,
        )
    return index


def create_dataset(
    layout: ProjectLayout,
    area: str | Sequence[str],
    feature_list: Sequence[str],
    true_objects: str,
    false_objects: str | None = None,
    *,
    test_area: str | None = None,
    img_size: tuple[int, int] = (512, 512),
    samples_per_object: int = 2,
    class_balanced: bool = False,
    seed: int = 0,
    dataset_id: str | None = None,
) -> Dataset:
    """Create a training dataset from labels and feature-map products.

    For every tile of the first feature layer (via its tile index), label
    geometries are matched by spatial intersection; ``samples_per_object``
    random windows are placed around each object (true labels first, then
    false labels). Feature layers are resolved per tile **spatially** and
    their transforms checked against the reference tile (plan B6).
    Missing or stale tile indexes of the feature products are rebuilt from
    the tiles on disk, and their VRT mosaics are ensured best-effort (a
    failed VRT build only warns; training reads tiles through the index).

    Args:
        layout: Project layout.
        area: Project area name, or several names to build **one** dataset
            from all of their tiles. The label shapefiles (and the test-area
            polygons) are shared and matched spatially per area — a label
            simply places windows in whichever areas its tiles it
            intersects. Every area must contain the feature layers, and
            all areas must share one pixel size (windows are placed in
            pixels; a mismatch raises ``ValueError``). Duplicate names are
            de-duplicated in order.
        feature_list: Feature layers relative to ``Lidar/<area>/`` (e.g.
            ``"02_TPI/radius_20"``); the first layer drives tile discovery.
        true_objects: Shapefile with true objects (in
            ``DeepLearning/train_annotations/``); optional ``class_id``
            column with values >= 1, default class 1.
        false_objects: Optional shapefile with false objects (all-background
            windows, the original hard-negative behaviour).
        test_area: Optional **test-area polygon layer** (e.g. a ``.gpkg``
            file, like the label shapefiles under
            ``DeepLearning/train_annotations/``). Tiles whose footprint
            intersects the polygons are excluded from sampling — the
            held-out test region replaces the original tile-ID list
            (``test_mapsheets``).
        img_size: Window size (H, W) in pixels (original: 512x512).
        samples_per_object: Windows per object (original: 2).
        class_balanced: Normalize windows per object per class so every
            class contributes approximately equally (plan 3.3/MC3); off by
            default (original 2-per-object parity). Rare classes get more
            windows per object; the largest class keeps
            ``samples_per_object``.
        seed: Seed for the window-placement RNG.
        dataset_id: Dataset name; default is the original-style timestamp
            (``%m%d%H%M%S``).

    Returns:
        The created :class:`Dataset`.

    Raises:
        ValueError: If no windows can be placed or arguments are invalid.
        FileNotFoundError: If labels or products are missing.
    """
    if not feature_list:
        raise ValueError("feature_list must not be empty")
    areas = [area] if isinstance(area, str) else list(area)
    deduped: list[str] = []
    for name in areas:
        if name not in deduped:
            deduped.append(name)
    areas = deduped
    if not areas:
        raise ValueError("at least one project area is required")
    if samples_per_object < 1:
        raise ValueError(f"samples_per_object must be >= 1, got {samples_per_object}")
    rng = random.Random(seed)

    annotations = layout.train_annotations_dir
    true = (
        _read_labels(annotations / true_objects, default_class=1)
        if true_objects is not None
        else gpd.GeoDataFrame({"class_id": np.empty(0, dtype=int), "geometry": []}, crs=layout.epsg)
    )
    false = _read_labels(annotations / false_objects, default_class=0) if false_objects else None
    # Compact class mapping (plan 3.3/MC2): the observed ids are remapped to
    # 1..C so sparse label ids (e.g. {1, 5}) do not create always-empty
    # intermediate classes. The map is recorded in the metadata and travels
    # with the model; ids are user data, so a gap is a warning, not an error.
    observed = sorted(int(c) for c in true["class_id"].unique()) if len(true) else [1]
    num_classes = len(observed)
    class_id_map = {original: compact for compact, original in enumerate(observed, start=1)}
    if len(observed) > 1 and observed != list(range(1, num_classes + 1)):
        LOGGER.warning(
            "non-contiguous class ids %s remapped to 1..%d (%s)",
            observed,
            num_classes,
            class_id_map,
        )
    if class_id_map != {c: c for c in observed}:
        true = true.assign(class_id=true["class_id"].map(class_id_map))
    # Class-balanced sampling (plan 3.3/MC3): windows per object normalized
    # per class so every class contributes approximately the window count of
    # the largest class.
    per_object: dict[int, int] = {}
    if class_balanced and num_classes > 1 and len(true):
        counts = true["class_id"].value_counts()
        largest = int(counts.max())
        per_object = {
            int(cls): max(1, int(round(largest * samples_per_object / int(n))))
            for cls, n in counts.items()
        }

    reference_indices: dict[str, gpd.GeoDataFrame] = {}
    test_mask = _read_test_area(annotations, test_area)
    excluded: list[tuple[str, str]] = []

    height, width = img_size
    # (area, tile_id, y_min, y_max, x_min, x_max) per window, original
    # sampling (the area rides along so tiles of several areas with equal
    # tile ids stay distinct);
    # window_classes records the placing class per window (0 = false-object /
    # background window) for the class-stratified validation split (MC7);
    # window_objects/window_bounds record the placing object and the window's
    # map footprint per window, and objects the full bounds of every label
    # object that placed windows, so training can split object-disjointly
    # (objects and clusters never on both sides of the split).
    windows: list[tuple[str, str, int, int, int, int]] = []
    window_classes: list[int] = []
    window_objects: list[str] = []
    window_bounds: list[list[float]] = []
    objects: dict[str, dict[str, Any]] = {}
    reference_pixel_size: float | None = None
    for area in areas:
        reference_index = _ensure_product(layout.lidar_area(area) / feature_list[0])
        reference_indices[area] = reference_index
        for _, tile in reference_index.iterrows():
            tile_id = str(tile["tile_id"])
            if test_mask is not None and tile.geometry.intersects(test_mask):
                excluded.append((area, tile_id))
                continue
            footprint = tile.geometry
            with rasterio.open(tile_path(reference_index, tile_id)) as src:
                transform = src.transform
                pixel_size = transform.a
            if reference_pixel_size is None:
                reference_pixel_size = float(pixel_size)
            elif float(pixel_size) != reference_pixel_size:
                raise ValueError(
                    f"area {area!r} pixel size {pixel_size} differs from the "
                    f"reference {reference_pixel_size}; all areas must share "
                    "one pixel size for pixel-based training windows"
                )
            local_frames = [
                (gpd.clip(true, box(*footprint.bounds)), True, "true", true),
                (
                    gpd.clip(false, box(*footprint.bounds)) if false is not None else None,
                    False,
                    "false",
                    false,
                ),
            ]
            for frame, is_true, prefix, source in local_frames:
                if frame is None or len(frame) == 0:
                    continue
                for _, row in frame.iterrows():
                    cls = int(row["class_id"]) if is_true else 0
                    # Clip preserves the source-frame index, so the row maps
                    # back to one stable object id; the object table records
                    # the full (unclipped) bounds for split-time grouping.
                    obj_id = f"{prefix}_{int(row.name)}"
                    if obj_id not in objects:
                        objects[obj_id] = {
                            "class_id": cls,
                            "bounds": list(source.loc[row.name].geometry.bounds),
                        }
                    count = (
                        per_object.get(cls, samples_per_object) if is_true else samples_per_object
                    )
                    placed = _place_windows(
                        row.geometry.bounds,
                        tile_id,
                        transform,
                        pixel_size,
                        height,
                        width,
                        count,
                        rng,
                    )
                    namespaced = [
                        (area, tile_id, ymin, ymax, xmin, xmax)
                        for _tile_id, ymin, ymax, xmin, xmax in placed
                    ]
                    windows.extend(namespaced)
                    window_classes.extend([cls] * len(namespaced))
                    window_objects.extend([obj_id] * len(namespaced))
                    for _area_id, _tile, ymin, ymax, xmin, xmax in namespaced:
                        window_bounds.append(
                            [
                                transform.c + xmin * pixel_size,
                                transform.f - ymax * pixel_size,
                                transform.c + xmax * pixel_size,
                                transform.f - ymin * pixel_size,
                            ]
                        )

    if not windows:
        raise ValueError(
            "no training windows placed: labels do not intersect any tile of "
            f"{feature_list[0]} in {', '.join(areas)}"
        )

    out_dir = dataset_dir(layout, dataset_id or datetime.now().strftime("%m%d%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        out_dir / SAMPLES_FILENAME,
        mode="w+",
        dtype=SAMPLES_DTYPE,
        shape=(len(windows), height, width, len(feature_list)),
    )
    labels = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        out_dir / LABELS_FILENAME,
        mode="w+",
        dtype=LABELS_DTYPE,
        shape=(len(windows), height, width),
    )

    by_area_tile: dict[str, dict[str, list[int]]] = {}
    for i, (window_area, tile_id, *_rest) in enumerate(windows):
        by_area_tile.setdefault(window_area, {}).setdefault(tile_id, []).append(i)

    for area, by_tile in by_area_tile.items():
        reference_index = reference_indices[area]
        for tile_id, indices in by_tile.items():
            local_true = true[true.intersects(_footprint(reference_index, tile_id))]
            label_mask = _label_mask(local_true, reference_index, tile_id)
            for i in indices:
                _area, _tile_id, ymin, ymax, xmin, xmax = windows[i]
                labels[i] = _extract_window(label_mask, ymin, ymax, xmin, xmax, height, width)

    for k, feature in enumerate(feature_list):
        for area, by_tile in by_area_tile.items():
            reference_index = reference_indices[area]
            feature_dir = layout.lidar_area(area) / feature
            feature_index = _ensure_product(feature_dir)
            for tile_id, indices in by_tile.items():
                footprint = _footprint(reference_index, tile_id)
                arr = _read_feature(feature_index, footprint, feature)
                _assert_aligned(reference_index, feature_index, tile_id, feature)
                for i in indices:
                    _area, _tile_id, ymin, ymax, xmin, xmax = windows[i]
                    samples[i, :, :, k] = _extract_window(
                        arr, ymin, ymax, xmin, xmax, height, width
                    )

    meta = {
        "dataset_id": out_dir.name,
        "area": areas[0],
        "areas": areas,
        "feature_list": list(feature_list),
        "img_size": list(img_size),
        "epsg": layout.epsg,
        "num_classes": num_classes,
        "class_names": [f"class_{c}" for c in range(1, num_classes + 1)],
        "class_id_map": [[orig, compact] for orig, compact in sorted(class_id_map.items())],
        "window_classes": window_classes,
        "window_objects": window_objects,
        "window_bounds": window_bounds,
        "objects": objects,
        "pixel_size": reference_pixel_size,
        "class_balanced": class_balanced,
        "augmented": False,
        "seed": seed,
        "samples_per_object": samples_per_object,
        # Single-area datasets record plain tile ids (the original format);
        # multi-area datasets namespace them as <area>/<tile> so equal tile
        # ids in different areas stay distinguishable.
        "excluded_tile_ids": (
            sorted(tile_id for _area, tile_id in excluded)
            if len(areas) == 1
            else sorted(f"{area}/{tile_id}" for area, tile_id in excluded)
        ),
        "test_area": test_area,
        "source_label_files": {
            "true_objects": true_objects,
            "false_objects": false_objects,
        },
        "samples_shape": list(samples.shape),
        "samples_dtype": SAMPLES_DTYPE.name,
        "labels_shape": list(labels.shape),
        "labels_dtype": LABELS_DTYPE.name,
    }
    (out_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    samples.flush()
    labels.flush()
    LOGGER.info("dataset %s: %d windows, %d classes", out_dir.name, len(windows), num_classes)
    return Dataset(out_dir.name, out_dir, samples, labels, meta)


def load_dataset(layout: ProjectLayout, dataset_id: str) -> Dataset:
    """Load a dataset by id; arrays are memory-mapped, not loaded."""
    out_dir = dataset_dir(layout, dataset_id)
    meta_path = out_dir / META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"dataset {dataset_id!r} not found ({meta_path})")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    samples = np.load(out_dir / SAMPLES_FILENAME, mmap_mode="r")
    labels = np.load(out_dir / LABELS_FILENAME, mmap_mode="r")
    return Dataset(dataset_id, out_dir, samples, labels, meta)


# --------------------------------------------------------------------- helpers


def _read_test_area(annotations: Path, test_area: str | None) -> BaseGeometry | None:
    """Union of the test-area polygons; ``None`` when no test area is given.

    The test area is an optional polygon layer (e.g. a ``.gpkg`` file) under
    the train annotations folder; tiles whose footprint intersects the union
    are excluded from sampling.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the layer contains no polygons.
    """
    if test_area is None:
        return None
    return read_polygon_union(annotations / test_area)


def _read_labels(path: Path, *, default_class: int) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"label file not found: {path}")
    frame = gpd.read_file(path)
    if "class_id" in frame.columns:
        class_ids = frame["class_id"].astype(int)
        if (class_ids < 1).any():
            raise ValueError(f"class_id values < 1 in {path}")
        frame = frame.assign(class_id=class_ids)
    else:
        frame = frame.assign(class_id=default_class)
    return frame[["class_id", "geometry"]]


def _place_windows(
    bounds: tuple[float, float, float, float],
    tile_id: str,
    transform: Any,
    pixel_size: float,
    height: int,
    width: int,
    count: int,
    rng: random.Random,
) -> list[tuple[str, int, int, int, int]]:
    """Original window placement: random (minx, maxy) window corners.

    The original drew ``minx`` from ``[maxx - W*px, minx]`` and ``maxy`` from
    ``[maxy, miny + H*px]``; both intervals are well-ordered for objects
    smaller than the window (the normal case). Objects larger than the window
    are clamped here instead of crashing.
    """
    minx, miny, maxx, maxy = bounds
    lo_x, hi_x = int(maxx - width * pixel_size), int(minx)
    lo_y, hi_y = int(maxy), int(miny + height * pixel_size)
    placed: list[tuple[str, int, int, int, int]] = []
    for _ in range(count):
        minx_pick = rng.randint(min(lo_x, hi_x), max(lo_x, hi_x))
        maxy_pick = rng.randint(min(lo_y, hi_y), max(lo_y, hi_y))
        x_arr_min = int((minx_pick - transform.c) / pixel_size)
        y_arr_min = int((transform.f - maxy_pick) / pixel_size)
        placed.append((tile_id, y_arr_min, y_arr_min + height, x_arr_min, x_arr_min + width))
    return placed


def _extract_window(
    arr: Array, ymin: int, ymax: int, xmin: int, xmax: int, height: int, width: int
) -> Array:
    """Original 4-corner fallback for windows crossing the raster edges.

    All four bounds are clipped to the raster first (the original clipped
    ``x_min``..``x_max`` to ``[0, shape]`` with ``np.clip``), then the four
    corner-anchored attempts follow in the original order.
    """
    rows, cols = arr.shape
    clipped_ymin = min(max(ymin, 0), rows)
    clipped_xmin = min(max(xmin, 0), cols)
    clipped_ymax = min(max(ymax, 0), rows)
    clipped_xmax = min(max(xmax, 0), cols)
    window = arr[clipped_ymin : clipped_ymin + height, clipped_xmin : clipped_xmin + width]
    if window.shape != (height, width):
        window = arr[
            clipped_ymax - height : clipped_ymax,
            clipped_xmax - width : clipped_xmax,
        ]
    if window.shape != (height, width):
        window = arr[clipped_ymin : clipped_ymin + height, clipped_xmax - width : clipped_xmax]
    if window.shape != (height, width):
        window = arr[
            clipped_ymax - height : clipped_ymax,
            clipped_xmin : clipped_xmin + width,
        ]
    if window.shape != (height, width):
        raise ValueError(
            f"window (y={ymin}, x={xmin}) cannot fit {height}x{width} in a "
            f"raster of shape {arr.shape}"
        )
    return window


def _footprint(index: gpd.GeoDataFrame, tile_id: str) -> BaseGeometry:
    rows = index[index["tile_id"] == tile_id]
    if len(rows) == 0:
        raise KeyError(f"tile_id {tile_id!r} missing from tile index")
    return rows.geometry.iloc[0]


def _label_mask(
    local_true: gpd.GeoDataFrame, reference_index: gpd.GeoDataFrame, tile_id: str
) -> Array:
    """Per-pixel uint8 class map of the tile's true objects.

    Rasterizes with ``all_touched=True``, matching the original
    ``geometry_mask(..., all_touched=True, invert=True)``.
    """
    path = tile_path(reference_index, tile_id)
    with rasterio.open(path) as src:
        shape = (src.height, src.width)
        transform = src.transform
    mask = np.zeros(shape, dtype=LABELS_DTYPE)
    if len(local_true) > 0:
        burned = rasterio.features.rasterize(
            (
                (geom, int(cls))
                for geom, cls in zip(local_true.geometry, local_true["class_id"], strict=True)
            ),
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=LABELS_DTYPE.name,
            all_touched=True,
        )
        mask = np.maximum(mask, burned)
    return mask


def _assert_aligned(
    reference_index: gpd.GeoDataFrame,
    feature_index: gpd.GeoDataFrame,
    tile_id: str,
    feature: str,
) -> None:
    """B6 guard: the feature tile must be pixel-aligned with the reference."""
    with (
        rasterio.open(tile_path(reference_index, tile_id)) as reference,
        rasterio.open(
            tile_path(feature_index, _matching_tile(feature_index, reference_index, tile_id))
        ) as other,
    ):
        if reference.transform != other.transform:
            raise ValueError(
                f"feature layer {feature!r} tile {tile_id!r} is not "
                "pixel-aligned with the reference layer"
            )


def _matching_tile(
    feature_index: gpd.GeoDataFrame, reference_index: gpd.GeoDataFrame, tile_id: str
) -> str:
    """Tile ID of the feature layer matching the reference tile spatially."""
    footprint = _footprint(reference_index, tile_id)
    hits = feature_index[feature_index.intersects(footprint)]
    if len(hits) == 0:
        raise ValueError(f"feature layer has no tile intersecting {tile_id!r}")
    if len(hits) > 1:
        areas = hits.geometry.intersection(footprint).area
        return str(hits.loc[areas.idxmax(), "tile_id"])
    return str(hits["tile_id"].iloc[0])


def _read_feature(feature_index: gpd.GeoDataFrame, footprint: BaseGeometry, feature: str) -> Array:
    """Read the feature band of the tile matching ``footprint`` spatially."""
    hits = feature_index[feature_index.intersects(footprint)]
    if len(hits) == 0:
        raise ValueError(f"feature {feature!r} has no tile intersecting the footprint")
    if len(hits) > 1:
        areas = hits.geometry.intersection(footprint).area
        row = hits.loc[areas.idxmax()]
    else:
        row = hits.iloc[0]
    with rasterio.open(Path(str(row["path"]))) as src:
        return np.asarray(src.read(1))


__all__ = [
    "Dataset",
    "create_dataset",
    "dataset_dir",
    "load_dataset",
    "META_FILENAME",
    "SAMPLES_FILENAME",
    "LABELS_FILENAME",
]
