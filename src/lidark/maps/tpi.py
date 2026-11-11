"""TPI (topographic position index) generation from DTM products.

Kernel semantics: ``out = z - weighted_mean(valid in-window neighbours)``
where the window is a **circular filter** of the given radius: full weight
(1.0) up to ``radius + 0.1`` px, a soft edge ring (weight 0.5) between
``radius + 0.1`` and ``radius + 0.5`` px, zero beyond, and the center pixel
excluded (weight 0). Nodata cells (``<= 0`` or non-finite) are **excluded**
from the weighted mean — they do not prevent computation. The output is
nodata only where the center itself is nodata or the window contains no
valid neighbour at all.

Border pixels use their available in-bounds neighbours (out-of-window cells
contribute nothing).

Implementation notes:

- :func:`circular_window` is the vectorized form of the original filter
  construction loop. The kernel evaluates it as the exact decomposition
  ``w = 0.5 * [d < r + 0.1] + 0.5 * [d < r + 0.5]`` — two hard-disk sums,
  each computed with per-row sliding sums over a shared prefix-sum array
  (O(H x W) per disk row, exact in float64 for float32 DTM values).
- :func:`tpi_reference` keeps the per-offset view-loop implementation with
  the same circular weights; the golden parity test proves kernel and
  reference produce identical arrays (float32 values accumulated in
  float64 are exact regardless of order).
- The driver takes the TPI radius in **metres**; ``radius_px =
  round(radius / pixel_size)``.

Output tiles cover the DTM tile footprints exactly: each tile is read from
the DTM VRT with a ``radius_px`` pixel margin — the buffer pulls the
necessary data from neighbouring tiles — and the margin is trimmed after
the kernel. Cells beyond the mosaic (missing neighbouring tiles at the area
edge) are filled with the DTM nodata value and therefore excluded by the
kernel.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import rasterio
from affine import Affine
from rasterio.windows import Window, from_bounds
from tqdm import tqdm

from lidark._types import Array
from lidark.config import ProjectLayout
from lidark.io.products import ensure_product
from lidark.io.rasters import is_valid_tile, write_band
from lidark.io.tiles import LAYER_NAME, tile_index_path
from lidark.io.vrt import build_vrt, open_vrt
from lidark.logging import get_logger

LOGGER = get_logger("maps.tpi")

TPI_NODATA = -32768  # int16 nodata constant of TPI output tiles

# Soft edge of the circular filter: full weight up to radius + EDGE_INNER,
# half weight in the ring up to radius + EDGE_OUTER, zero beyond.
EDGE_INNER = 0.1
EDGE_OUTER = 0.5

# (tile_id, vrt_path, window, radius_px, dtype, crs, mosaic_transform, out_path)
# crs/mosaic_transform are captured at task-build time so the all-nodata
# fallback tile can be written even when the VRT read itself failed.
_TpiTask = tuple[str, Path, tuple[int, int, int, int], int, Any, str, Affine, Path]


def circular_window(radius_px: int) -> Array:
    """Circular filter weights of the given pixel radius.

    Vectorized form of the original construction loop: weight 1.0 for cells
    closer than ``radius_px + 0.1``, 0.5 in the edge ring
    ``[radius_px + 0.1, radius_px + 0.5)``, 0 beyond; the center pixel is
    excluded (weight 0). The window is ``(2 r + 1) x (2 r + 1)``.

    Raises:
        ValueError: If ``radius_px`` is < 1.
    """
    if radius_px < 1:
        raise ValueError(f"radius_px must be >= 1, got {radius_px}")
    size = 2 * radius_px + 1
    y, x = np.ogrid[:size, :size]
    dist = np.hypot(x - radius_px, y - radius_px)
    win = np.ones((size, size), dtype=np.float64)
    win[dist >= radius_px + EDGE_OUTER] = 0.0
    win[(dist >= radius_px + EDGE_INNER) & (dist < radius_px + EDGE_OUTER)] = 0.5
    win[radius_px, radius_px] = 0.0
    return win


def tpi_kernel(dtm: Array, radius_px: int, dtype: npt.DTypeLike = np.float32) -> Array:
    """Vectorized TPI kernel with the circular filter: ``z - weighted mean``.

    Args:
        dtm: 2-D elevation band; values ``<= 0`` are treated as nodata.
        radius_px: Filter radius in pixels (window side ``2 * radius_px + 1``).
        dtype: Accumulation dtype (``float32`` default; ``float64`` for the
            bit-parity golden test).

    Returns:
        Float array of the kernel output; NaN where the center is nodata or
        the window contains no valid neighbour. Nodata neighbours are
        excluded from the weighted mean — they do not prevent computation.
    """
    z = np.array(dtm, dtype=dtype, copy=True)
    if z.ndim != 2:
        raise ValueError(f"expected a 2-D band, got shape {z.shape}")
    if radius_px < 1:
        raise ValueError(f"radius_px must be >= 1, got {radius_px}")
    if np.dtype(dtype).kind != "f":
        raise ValueError(f"dtype must be a float type, got {dtype!r}")
    z[z <= 0] = np.nan
    valid = np.isfinite(z)
    z0 = np.where(valid, z, 0)
    valid_f = valid.astype(z.dtype)
    # The soft-edge circular filter is the exact decomposition
    # w = 0.5 * [d < r + EDGE_INNER] + 0.5 * [d < r + EDGE_OUTER]: two
    # hard-disk sums. The center lies inside both disks (combined weight 1)
    # but must be excluded, hence the -z0 / -valid correction.
    t_in = radius_px + EDGE_INNER
    t_out = radius_px + EDGE_OUTER
    sum_nb = 0.5 * _disk_sum(z0, t_in) + 0.5 * _disk_sum(z0, t_out) - z0
    cnt_nb = 0.5 * _disk_sum(valid_f, t_in) + 0.5 * _disk_sum(valid_f, t_out) - valid_f
    with np.errstate(invalid="ignore", divide="ignore"):
        out = z - sum_nb / cnt_nb
    out = np.where(valid & (cnt_nb > 0), out, np.nan)
    return out


def _disk_half_width(t: float, dy: int) -> int:
    """Max ``|dx|`` with ``dy^2 + dx^2 < t^2`` (-1: no cells in the row).

    ``t`` is never an integer here (``radius + 0.1`` / ``radius + 0.5``), so
    the strict comparison is guarded with a tiny epsilon only.
    """
    v = t * t - dy * dy
    if v <= 0.0:
        return -1
    return int(math.floor(math.sqrt(v - 1e-9)))


def _disk_sum(a: Array, t: float) -> Array:
    """Sliding sum over the disk ``{sqrt(dy^2 + dx^2) < t}``.

    Per-row sliding sums over a shared prefix-sum array (computed once),
    one vectorized pass per disk row: O(H x W) per row, exact for
    integer-grid inputs (e.g. float32 DTM values accumulated in float64).
    Out-of-bounds cells contribute zero (the border behaviour).
    """
    height, width = a.shape
    pad = int(math.floor(t))
    padded = np.zeros((height + 2 * pad, width + 2 * pad), dtype=a.dtype)
    padded[pad : pad + height, pad : pad + width] = a
    csum = np.zeros((height + 2 * pad, width + 2 * pad + 1), dtype=a.dtype)
    np.cumsum(padded, axis=1, out=csum[:, 1:])
    out = np.zeros((height, width), dtype=a.dtype)
    for dy in range(-pad, pad + 1):
        h = _disk_half_width(t, dy)
        if h < 0:
            continue
        rows = csum[pad + dy : pad + dy + height]
        # Sum over dx in [-h, h] at each output column x:
        # csum[x + pad + h + 1] - csum[x + pad - h].
        out += rows[:, pad + h + 1 : pad + h + 1 + width]
        out -= rows[:, pad - h : pad - h + width]
    return out


def tpi_reference(dtm: Array, radius_px: int, dtype: npt.DTypeLike = np.float64) -> Array:
    """Per-offset view-loop reference implementation.

    Same loop structure as the original ``CreateMap.tpi_index``, with the
    circular filter weights (:func:`circular_window`) and the revised nodata
    semantics: each window offset contributes only its valid cells to the
    weighted sum and count, so nodata neighbours are excluded from the mean
    instead of propagating NaN (the original's behaviour).
    """
    win = circular_window(radius_px).astype(dtype)
    range_list = [[(int(y), int(x)), weight] for (y, x), weight in np.ndenumerate(win)]
    mx_z: Array = np.array(dtm, dtype=dtype, copy=True)
    mx_z[mx_z <= 0] = np.nan
    mx_temp = np.zeros(mx_z.shape, dtype=dtype)
    mx_count = np.zeros(mx_z.shape, dtype=dtype)
    size_y, size_x = mx_z.shape

    def view(offset_y: int, offset_x: int) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
        """Two matching views for a moving-window offset (original code)."""
        x, y = abs(offset_x), abs(offset_y)
        x_in = slice(x, size_x)
        x_out = slice(0, size_x - x)
        y_in = slice(y, size_y)
        y_out = slice(0, size_y - y)
        # the swapping trick
        if offset_x < 0:
            x_in, x_out = x_out, x_in
        if offset_y < 0:
            y_in, y_out = y_out, y_in
        return (y_in, x_in), (y_out, x_out)

    for (y, x), weight in range_list:
        if weight == 0:
            continue  # skip zero values!
        view_in, view_out = view(y - radius_px, x - radius_px)
        z_in = mx_z[view_in]
        ok = np.isfinite(z_in)
        mx_temp[view_out] += np.where(ok, z_in, 0.0) * weight
        mx_count[view_out] += ok.astype(dtype) * weight

    with np.errstate(invalid="ignore", divide="ignore"):
        out: Array = mx_z - mx_temp / mx_count
    return out


def tpi_postprocess(
    out: Array, scale: float = 100.0, clip: float = 100.0, nodata: int = TPI_NODATA
) -> Array:
    """Post-processing: ``*scale``, clip, int16.

    Kernel nodata (NaN) maps to the int16 ``nodata`` constant instead of the
    original's 0, so nodata cells stay distinguishable from a flat surface.
    """
    arr = np.asarray(out)
    scaled = np.asarray(arr * scale)
    scaled = np.clip(scaled, -clip, clip)
    return np.where(np.isfinite(scaled), scaled, nodata).astype("int16")


def create_tpi(
    layout: ProjectLayout,
    area: str,
    radius: float,
    *,
    gpu: bool = False,
    dtype: npt.DTypeLike = np.float32,
    cores: int = 1,
) -> list[Path]:
    """Create TPI tiles (``02_TPI/radius_<radius>/``) from the area's DTM.

    For each DTM tile in the tile index, one windowed read covering the tile
    footprint plus ``radius_px`` pixels comes from ``01_DTM/dtm.vrt`` — the
    buffer pulls the necessary data from neighbouring tiles. The circular
    filter of :func:`circular_window` (soft edge at the radius) computes the
    TPI. Cells beyond the mosaic (no neighbouring tile at the area edge) are
    filled with the DTM nodata value; the kernel excludes them (nodata does
    not prevent computation). The margin is trimmed afterwards so output
    tiles cover the DTM tile footprints exactly. Nodata cells in the DTM
    propagate as ``TPI_NODATA`` in the output. Afterwards the TPI VRT mosaic
    and tile index are written. A missing or stale DTM mosaic or tile index
    is rebuilt from the DTM tiles on disk first.

    Per-tile failures never abort the run: an index row with a bad footprint
    is skipped with a warning, a failing tile (window read, kernel, write —
    on the CPU or the batched GPU path) is replaced by an all-nodata tile,
    and a tile that could not be written at all is excluded from the mosaic
    and tile index (also warned about).

    Re-running resumes instead of restarting: a tile whose output already
    exists (and is a readable raster at the DTM's pixel size) is skipped,
    with one log line per skipped tile. Delete a tile's output (or the whole
    ``02_TPI/radius_<r>/`` folder) to force reprocessing.

    Args:
        layout: Project layout.
        area: Project area name.
        radius: TPI radius in metres (also the output folder name).
        gpu: If True, batch tiles through a single torch ``conv2d`` per chunk
            (7, P3); falls back to CPU when CUDA is unavailable.
        dtype: Kernel accumulation dtype (``float64`` for bit parity).
        cores: Worker processes for the CPU path.

    Returns:
        Paths of the TPI tiles in the product (written and skipped alike).

    Raises:
        FileNotFoundError: If the DTM directory does not exist or holds no
            tiles; generate the DTM first (or ``gdalbuildvrt`` is missing
            for the mosaic rebuild).
        RuntimeError: If no TPI tile could be written at all.
        ValueError: For invalid radius or pixel alignment.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    dtm_dir = layout.dtm_dir(area)
    vrt_path, index = ensure_product(dtm_dir)
    out_dir = layout.tpi_dir(area, radius)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[_TpiTask] = []
    skipped: dict[str, Path] = {}
    with open_vrt(vrt_path) as mosaic:
        transform = mosaic.transform
        crs = mosaic.crs.to_string() if mosaic.crs is not None else layout.epsg
        pixel_size = transform.a
        radius_px = int(round(radius / pixel_size))
        if radius_px < 1:
            raise ValueError(f"radius {radius} is smaller than one pixel ({pixel_size})")
        for _, row in index.iterrows():
            tile_id = str(row["tile_id"])
            try:
                window = _tile_window(row.geometry.bounds, radius_px, transform)
            except Exception as exc:  # noqa: BLE001 - per-tile isolation, logged
                LOGGER.warning("skipping TPI tile %s: bad tile footprint (%s)", tile_id, exc)
                continue
            out_path = out_dir / f"{tile_id}_TPI.tif"
            # Resume on re-run: an existing, valid output tile is not recomputed.
            if is_valid_tile(out_path, pixel_size):
                skipped[tile_id] = out_path
                LOGGER.info("skipping %s: TPI tile already processed", out_path.name)
                continue
            tasks.append((tile_id, vrt_path, window, radius_px, dtype, crs, transform, out_path))
    if skipped:
        LOGGER.info("%d of %d TPI tiles already processed; skipped", len(skipped), len(index))

    if gpu:
        _create_tpi_gpu(tasks)
    elif cores > 1:
        _run_tpi_jobs_pool(tasks, cores)
    else:
        for task in tqdm(tasks, desc="TPI tiles", disable=len(tasks) < 4):
            _tpi_tile_worker(task)

    # Keep only tiles actually written (plus the skipped, already existing
    # ones): a tile whose fallback write also failed is excluded instead of
    # breaking the mosaic or the index.
    written = {**skipped, **{task[0]: task[-1] for task in tasks if task[-1].exists()}}
    missing = [task[-1].name for task in tasks if not task[-1].exists()]
    if missing:
        LOGGER.warning(
            "%d of %d TPI tiles were not written: %s", len(missing), len(tasks), ", ".join(missing)
        )
    if not written:
        raise RuntimeError(f"no TPI tiles could be written for area {area!r} (radius {radius})")
    tile_paths = [written[str(tile_id)] for tile_id in index["tile_id"] if str(tile_id) in written]
    build_vrt(tile_paths, out_dir / "tpi.vrt")
    # The TPI tile index reuses the DTM tile IDs and footprints (plan:
    # spatial identity, never filename parsing) with the TPI paths.
    keep = [str(tile_id) in written for tile_id in index["tile_id"]]
    tpi_index = index[keep].copy()
    tpi_index["path"] = [str(written[str(tile_id)]) for tile_id in tpi_index["tile_id"]]
    tpi_index.to_file(tile_index_path(out_dir), layer=LAYER_NAME)
    LOGGER.info(
        "TPI product for %s (radius %s): %d tiles in product (%d skipped as already processed)",
        area,
        radius,
        len(tile_paths),
        len(skipped),
    )
    return tile_paths


def _run_tpi_jobs_pool(tasks: list[_TpiTask], cores: int) -> None:
    """TPI tiles across a worker pool.

    A broken pool (e.g. a native crash in one tile's read/kernel) does not
    end the run: the unfinished tiles are re-run serially. Per-tile Python
    exceptions never reach this level — the worker contains them.
    """
    if not tasks:
        return
    done: set[Path] = set()
    try:
        with ProcessPoolExecutor(max_workers=cores) as pool:
            for path in tqdm(pool.map(_tpi_tile_worker, tasks), total=len(tasks), desc="TPI tiles"):
                done.add(path)
        return
    except BrokenProcessPool as exc:
        LOGGER.warning("TPI worker pool broke (%s); re-running unfinished tiles serially", exc)
    for task in tqdm(tasks, desc="TPI tiles (serial fallback)"):
        if task[-1] not in done:
            _tpi_tile_worker(task)


def _tile_window(
    bounds: tuple[float, float, float, float], radius_px: int, transform: Affine
) -> tuple[int, int, int, int]:
    """Tile-footprint window expanded by exactly ``radius_px`` pixels a side."""
    win = from_bounds(*bounds, transform=transform)
    col0 = int(round(win.col_off)) - radius_px
    row0 = int(round(win.row_off)) - radius_px
    width = int(round(win.width)) + 2 * radius_px
    height = int(round(win.height)) + 2 * radius_px
    return col0, row0, width, height


def _read_window(vrt_path: Path, window: tuple[int, int, int, int]) -> tuple[Array, Any]:
    """Read one window boundless; outside the mosaic is the DTM nodata fill.

    Cells with no neighbouring tile at the area edge become nodata and are
    excluded by the kernel (they do not prevent TPI computation).
    """
    col0, row0, width, height = window
    with rasterio.open(vrt_path) as mosaic:
        fill = mosaic.nodata if mosaic.nodata is not None else 0.0
        arr = np.asarray(
            mosaic.read(
                1, window=Window(col0, row0, width, height), boundless=True, fill_value=fill
            )
        )
        crs = mosaic.crs
        sub_transform = mosaic.transform * Affine.translation(col0, row0)
    return arr, (crs, sub_transform)


def _tpi_tile_worker(task: _TpiTask) -> Path:
    """Compute and write one TPI tile (runs in worker processes too).

    A failure of any stage (window read, kernel, write) is logged and
    replaced by an all-nodata tile; it never propagates — one bad tile must
    not end the run."""
    _tile_id, vrt_path, window, radius_px, dtype, _crs, _transform, out_path = task
    try:
        arr, (crs, sub_transform) = _read_window(vrt_path, window)
        out = tpi_kernel(arr, radius_px, dtype=dtype)
        post = tpi_postprocess(out)
        trimmed, shifted = _trim_pixels(post, sub_transform, radius_px)
        write_band(out_path, trimmed, transform=shifted, crs=crs, nodata=TPI_NODATA)
    except Exception as exc:  # noqa: BLE001 - per-tile isolation, logged
        LOGGER.warning(
            "TPI tile failed for %s (%s); writing an all-nodata tile", out_path.name, exc
        )
        _write_failed_tpi_tile(task)
    return out_path


def _write_failed_tpi_tile(task: _TpiTask) -> None:
    """Best-effort all-nodata fallback tile; a failure here is logged, not raised."""
    try:
        _write_nodata_tpi_tile(task)
    except Exception:  # noqa: BLE001 - last-resort fallback, logged
        LOGGER.exception("could not write the all-nodata fallback tile %s", task[-1])


def _write_nodata_tpi_tile(task: _TpiTask) -> None:
    """A tile-footprint-shaped, all-nodata int16 GeoTIFF (failure fallback).

    Shape and transform mirror the normal output (read window minus the
    ``radius_px`` margin, shifted accordingly), so the mosaic and tile index
    stay consistent; the CRS and mosaic transform come from the task (captured
    when it was built), not from a fresh VRT read that may be what failed.
    """
    _tile_id, _vrt_path, window, radius_px, _dtype, crs, mosaic_transform, out_path = task
    col0, row0, width, height = window
    sub_transform = mosaic_transform * Affine.translation(col0, row0)
    h, w = height - 2 * radius_px, width - 2 * radius_px
    if h > 0 and w > 0:
        arr = np.full((h, w), TPI_NODATA, dtype=np.int16)
        out_transform = sub_transform * Affine.translation(radius_px, radius_px)
    else:
        # Degenerate window (the radius covers the whole tile): nodata over
        # the full window, so the tile still exists.
        arr = np.full((height, width), TPI_NODATA, dtype=np.int16)
        out_transform = sub_transform
    write_band(out_path, arr, transform=out_transform, crs=crs, nodata=TPI_NODATA)


def _trim_pixels(arr: Array, transform: Affine, px: int) -> tuple[Array, Affine]:
    """Slice ``px`` pixels from each edge and shift the transform to match."""
    if px <= 0:
        return arr, transform
    if 2 * px >= min(arr.shape[0], arr.shape[1]):
        raise ValueError(f"margin of {px} px covers the array of shape {arr.shape}")
    trimmed = arr[px:-px, px:-px]
    shifted = transform * Affine.translation(px, px)
    return trimmed, shifted


def _create_tpi_gpu(tasks: list[_TpiTask], batch_size: int = 8) -> None:
    """Batched torch path: one ``conv2d`` per chunk of same-shaped tiles (7, P3).

    Per-tile isolation: a failed window read or tile write becomes an
    all-nodata tile; windows of different tile shapes are grouped (they
    cannot be stacked); a failed batch (e.g. OOM) degrades to the per-tile
    CPU worker for that chunk. No single tile failure ends the run.
    """
    import torch

    if not tasks:
        return
    if not torch.cuda.is_available():
        LOGGER.warning("CUDA unavailable; running the batched path on CPU")
    radius_px = tasks[0][3]
    for start in tqdm(range(0, len(tasks), batch_size), desc="TPI batches"):
        chunk = tasks[start : start + batch_size]
        readable: list[tuple[_TpiTask, Array, Any]] = []
        for task in chunk:
            try:
                arr, meta = _read_window(task[1], task[2])
            except Exception as exc:  # noqa: BLE001 - per-tile isolation, logged
                LOGGER.warning(
                    "TPI window read failed for %s (%s); writing an all-nodata tile",
                    task[-1].name,
                    exc,
                )
                _write_failed_tpi_tile(task)
                continue
            readable.append((task, arr, meta))
        # Windows of different tile shapes cannot be stacked: group by shape.
        groups: dict[tuple[int, ...], list[tuple[_TpiTask, Array, Any]]] = {}
        for item in readable:
            groups.setdefault(item[1].shape, []).append(item)
        for items in groups.values():
            try:
                out = _tpi_kernel_torch(np.stack([arr for _, arr, _ in items]), radius_px)
            except Exception as exc:  # noqa: BLE001 - batch isolation, logged
                LOGGER.warning("batched TPI kernel failed (%s); per-tile CPU fallback", exc)
                for task, _arr, _meta in items:
                    _tpi_tile_worker(task)
                continue
            for (task, _arr, (crs, sub_transform)), tile in zip(items, out, strict=True):
                try:
                    post = tpi_postprocess(tile)
                    trimmed, shifted = _trim_pixels(post, sub_transform, radius_px)
                    write_band(task[-1], trimmed, transform=shifted, crs=crs, nodata=TPI_NODATA)
                except Exception as exc:  # noqa: BLE001 - per-tile isolation, logged
                    LOGGER.warning(
                        "TPI tile write failed for %s (%s); writing an all-nodata tile",
                        task[-1].name,
                        exc,
                    )
                    _write_failed_tpi_tile(task)


def _tpi_kernel_torch(batch: Array, radius_px: int) -> Array:
    """TPI kernel over a ``(B, H, W)`` batch via a single ``conv2d`` per image.

    Matches :func:`tpi_kernel` semantics (circular filter, revised nodata
    handling: nodata neighbours are excluded, constant zero padding for
    borders). Accumulates in float32.
    """
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = torch.as_tensor(np.asarray(batch), dtype=torch.float32, device=device)
    z = torch.where(z <= 0, torch.full_like(z, float("nan")), z)
    valid = torch.isfinite(z)
    z0 = torch.nan_to_num(z, nan=0.0)
    size = 2 * radius_px + 1
    # Circular filter weights; the center weight is 0, so the convolutions
    # already exclude the center pixel.
    kernel = torch.as_tensor(
        circular_window(radius_px), dtype=torch.float32, device=device
    ).reshape(1, 1, size, size)
    images = z0.unsqueeze(1)
    # Zero padding means out-of-bounds cells contribute 0 to sums and counts,
    # reproducing the original border behaviour. The count convolves the
    # validity mask, so nodata neighbours are excluded from the mean.
    sum_nb = F.conv2d(images, kernel, padding=radius_px)
    cnt_nb = F.conv2d(valid.float().unsqueeze(1), kernel, padding=radius_px)
    # Weighted counts can be fractional (the edge ring carries weight 0.5),
    # so guard only the zero: cells with no valid neighbour are masked to
    # NaN below, the dummy 1.0 just keeps the division finite.
    cnt_safe = torch.where(cnt_nb > 0, cnt_nb, torch.ones_like(cnt_nb))
    # z is (B, H, W); the conv results are (B, 1, H, W) — align the batch
    # axis before the subtraction (a bare `z - ...` would broadcast the
    # batch against the channel axis, silently producing (B, B, H, W)).
    out = z.unsqueeze(1) - sum_nb / cnt_safe
    out = torch.where(valid.unsqueeze(1) & (cnt_nb > 0), out, torch.full_like(out, float("nan")))
    return cast(Array, out.squeeze(1).cpu().numpy())
