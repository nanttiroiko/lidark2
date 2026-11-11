"""DTM generation from LiDAR point clouds with PDAL (replaces LAStools).

Tile-index neighbour reads (plan Phase 2 remark): **no ground-point
intermediate is written to disk**. For each output tile, the input LAZ tiles
whose header bounds intersect the tile extent + ``buffer`` are read directly
(one pipeline per output tile): the intersecting files are merged, filtered
to ground points (``Classification == 2``, LAStools ``-keep_class 2``), and
cropped to the tile extent + ``buffer`` — only the few tiles actually
bordering an output tile are opened, and nothing is persisted in between.

The DTM itself is a TIN rasterization like the original ``las2dem``:
``filters.delaunay`` triangulates the ground points and
``filters.faceraster`` rasterizes the triangle faces. The raster grid is
pinned to the **original tile bounds** (``origin_x``/``origin_y``/
``width``/``height``; ``origin_y`` is the bottom edge), so the buffer points
only improve triangulation at the seams — the output tile is clipped to the
tile extent and adjacent tiles abut exactly, without overlap.

``filters.faceraster`` also accepts ``max_triangle_edge_length`` (a direct
equivalent of the original ``-kill 50`` TIN edge limit); it is left at PDAL's
default (no limit) here.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box
from tqdm import tqdm

from lidark.config import ProjectLayout
from lidark.io.rasters import is_valid_tile, write_band
from lidark.io.tiles import LAYER_NAME, tile_index_path
from lidark.io.vrt import build_vrt
from lidark.logging import get_logger

LOGGER = get_logger("maps.dtm")

GROUND_CLASS = 2
DTM_NODATA = -9999.0
DEFAULT_BUFFER = 2.0
Bounds = tuple[float, float, float, float]

# (pipeline dict, out_path, tile bounds, pixel size, epsg fallback)
_DtmJob = tuple[dict[str, Any], Path, Bounds, float, str]


def snap_bounds(bounds: Bounds, pixel_size: float) -> Bounds:
    """Snap bounds outward onto the pixel grid (shared by all tiles)."""
    minx, miny, maxx, maxy = bounds
    return (
        math.floor(minx / pixel_size) * pixel_size,
        math.floor(miny / pixel_size) * pixel_size,
        math.ceil(maxx / pixel_size) * pixel_size,
        math.ceil(maxy / pixel_size) * pixel_size,
    )


def expand_bounds(bounds: Bounds, buffer: float) -> Bounds:
    """Expand bounds by ``buffer`` on each side."""
    minx, miny, maxx, maxy = bounds
    return (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)


def _bounds_str(bounds: Bounds) -> str:
    """PDAL bounds string ``([minx,maxx],[miny,maxy])``."""
    minx, miny, maxx, maxy = bounds
    return f"([{minx!r},{maxx!r}],[{miny!r},{maxy!r}])"


def tile_pipeline(
    laz_paths: list[Path],
    tile_bounds: Bounds,
    crop_bounds: Bounds,
    out_path: Path,
    pixel_size: float,
) -> dict[str, Any]:
    """One output tile: neighbour reads -> merge -> ground -> crop -> TIN.

    Points are cropped to ``crop_bounds`` (tile extent + buffer) so the
    Delaunay triangulation at tile edges uses neighbouring points (clean
    seams); the raster grid is pinned to ``tile_bounds`` — the original tile
    extent — so the written tile is clipped to the tile itself.

    Args:
        laz_paths: Input LAZ tiles intersecting ``crop_bounds``.
        tile_bounds: Output tile extent (the raster grid).
        crop_bounds: Point-crop extent (tile + buffer).
        out_path: Output GeoTIFF.
        pixel_size: Raster resolution (metres per pixel).

    Returns:
        The PDAL pipeline dict.
    """
    minx, miny, maxx, maxy = tile_bounds
    width = int(round((maxx - minx) / pixel_size))
    height = int(round((maxy - miny) / pixel_size))
    return {
        "pipeline": [
            # Multiple readers require an explicit filters.merge (modern PDAL
            # streams do not concatenate reader stages).
            *[{"type": "readers.las", "filename": str(p)} for p in laz_paths],
            {"type": "filters.merge"},
            {
                "type": "filters.range",
                "limits": f"Classification[{GROUND_CLASS}:{GROUND_CLASS}]",
            },
            {"type": "filters.crop", "bounds": _bounds_str(crop_bounds)},
            {"type": "filters.delaunay"},
            {
                "type": "filters.faceraster",
                "resolution": pixel_size,
                "origin_x": minx,
                "origin_y": miny,
                "width": width,
                "height": height,
                "nodata": DTM_NODATA,
            },
            {
                "type": "writers.raster",
                "filename": str(out_path),
                "gdaldriver": "GTiff",
                "data_type": "float",
                "nodata": DTM_NODATA,
            },
        ]
    }


def las_bounds(path: Path) -> Bounds:
    """Point-cloud bounds via a PDAL header read (``quickinfo``).

    Searches the quickinfo payload defensively for a bbox with
    ``minx/maxx/miny/maxy`` so minor PDAL metadata format changes do not break
    tile-grid construction.
    """
    import pdal

    info = pdal.Reader(str(path)).pipeline().quickinfo
    bbox = _find_bbox(info)
    if bbox is None:
        raise ValueError(f"could not read bounds of {path}")
    return bbox


def _find_bbox(obj: Any) -> Bounds | None:
    if isinstance(obj, dict):
        if {"minx", "maxx", "miny", "maxy"} <= obj.keys():
            return (
                float(obj["minx"]),
                float(obj["miny"]),
                float(obj["maxx"]),
                float(obj["maxy"]),
            )
        for value in obj.values():
            found = _find_bbox(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_bbox(value)
            if found is not None:
                return found
    return None


def _run_pipeline(pipe: dict[str, Any]) -> int:
    """Execute one PDAL pipeline (module-level: runs in worker processes)."""
    import pdal

    pipeline = pdal.Pipeline(json.dumps(pipe))
    return int(pipeline.execute())


def _dtm_tile_worker(job: _DtmJob) -> Path:
    """Run one tile pipeline; an all-nodata tile on failure keeps the product
    consistent (e.g. tiles with no ground points at all). A failure of any
    stage — including the fallback write — is logged, never raised: one bad
    tile must not end the run."""
    pipe, out_path, bounds, pixel_size, epsg = job
    try:
        _run_pipeline(pipe)
    except Exception as exc:  # noqa: BLE001 - per-tile isolation, logged
        LOGGER.warning(
            "DTM pipeline failed for %s (%s); writing an all-nodata tile",
            out_path.name,
            exc,
        )
        try:
            _write_nodata_tile(out_path, bounds, pixel_size, epsg)
        except Exception:  # noqa: BLE001 - last-resort fallback, logged
            LOGGER.exception("could not write the all-nodata fallback tile %s", out_path)
    return out_path


def _write_nodata_tile(out_path: Path, bounds: Bounds, pixel_size: float, epsg: str) -> None:
    """A tile-shaped, all-nodata GeoTIFF (fallback for ground-free tiles)."""
    minx, miny, maxx, maxy = bounds
    width = int(round((maxx - minx) / pixel_size))
    height = int(round((maxy - miny) / pixel_size))
    arr = np.full((height, width), DTM_NODATA, dtype=np.float32)
    write_band(
        out_path,
        arr,
        transform=from_origin(minx, maxy, pixel_size, pixel_size),
        crs=epsg,
        nodata=DTM_NODATA,
    )


def _run_jobs(jobs: list[_DtmJob], cores: int, desc: str) -> None:
    """Execute tile jobs serially or across a worker pool.

    A broken pool (e.g. a native crash inside one tile's PDAL run) does not
    end the run: the unfinished jobs are re-run serially. Per-tile Python
    exceptions never reach this level — the worker contains them.
    """
    if not jobs:
        return
    if cores > 1:
        done: set[Path] = set()
        try:
            with ProcessPoolExecutor(max_workers=cores) as pool:
                for path in tqdm(pool.map(_dtm_tile_worker, jobs), total=len(jobs), desc=desc):
                    done.add(path)
            return
        except BrokenProcessPool as exc:
            LOGGER.warning(
                "%s: worker pool broke (%s); re-running unfinished tiles serially", desc, exc
            )
        jobs = [job for job in jobs if job[1] not in done]
    for job in tqdm(jobs, desc=desc):
        _dtm_tile_worker(job)


def generate_dtm(
    layout: ProjectLayout,
    area: str,
    *,
    pixel_size: float = 0.25,
    buffer: float = DEFAULT_BUFFER,
    cores: int = 8,
) -> list[Path]:
    """Generate DTM tiles (``01_DTM/``) from the area's original LAZ files.

    One output tile per input LAZ file, bounds from the file header snapped
    to the pixel grid. Each tile pipeline reads only the intersecting
    neighbouring LAZ tiles (tile extent + ``buffer``) — no intermediate
    point cloud is written to disk — triangulates the ground points
    (``filters.delaunay``) and rasterizes the TIN clipped to the original
    tile bounds (``filters.faceraster``).

    Args:
        layout: Project layout.
        area: Project area name.
        pixel_size: Output raster resolution (metres per pixel).
        buffer: Point-read margin in metres on each side of the tile
            (triangulation overlap at the seams; default 2.0).
        cores: Worker processes for the per-tile pipelines.

    Per-tile failures never abort the run: an input file whose bounds cannot
    be read is skipped with a warning, a failing tile pipeline is replaced by
    an all-nodata tile, and a tile that could not be written at all is
    excluded from the mosaic and tile index (also warned about).

    Re-running resumes instead of restarting: a tile whose output already
    exists (and is a readable raster at the current ``pixel_size``) is
    skipped, with one log line per skipped tile. Delete a tile's output (or
    the whole product directory) to force reprocessing.

    Returns:
        Paths of the DTM tiles in the product (written and skipped alike).

    Raises:
        FileNotFoundError: If the area has no ``.las``/``.laz`` files, or none
            of them is readable.
        RuntimeError: If no DTM tile could be written at all.
        ValueError: For invalid arguments.
    """
    if pixel_size <= 0:
        raise ValueError(f"pixel_size must be > 0, got {pixel_size}")
    if buffer < 0:
        raise ValueError(f"buffer must be >= 0, got {buffer}")
    if cores < 1:
        raise ValueError(f"cores must be >= 1, got {cores}")
    originals = layout.originals_dir(area)
    laz_paths = sorted(p for p in originals.glob("*.la[sz]") if p.is_file())
    if not laz_paths:
        raise FileNotFoundError(f"no .las/.laz files found in {originals}")

    dtm_dir = layout.dtm_dir(area)
    dtm_dir.mkdir(parents=True, exist_ok=True)

    # Output tile grid: one tile per input file, snapped to the pixel grid.
    # An unreadable LAZ header (corrupt file) skips only that tile.
    tiles: list[tuple[str, Bounds, Path, Path]] = []
    for path in laz_paths:
        tile_id = path.stem
        try:
            bounds = snap_bounds(las_bounds(path), pixel_size)
        except Exception as exc:  # noqa: BLE001 - per-tile isolation, logged
            LOGGER.warning("skipping %s: cannot read its bounds (%s)", path.name, exc)
            continue
        tiles.append((tile_id, bounds, path, dtm_dir / f"{tile_id}_DTM.tif"))
    if not tiles:
        raise FileNotFoundError(f"no readable .las/.laz files in {originals}")

    # Tile-index neighbour reads: each pipeline opens only the LAZ tiles
    # intersecting its extent + buffer (plan Phase 2 remark). Tiles whose
    # output already exists and is valid are skipped (resume on re-run).
    jobs: list[_DtmJob] = []
    skipped = 0
    for _tile_id, bounds, _source, out_path in tiles:
        if is_valid_tile(out_path, pixel_size):
            skipped += 1
            LOGGER.info("skipping %s: DTM tile already processed", out_path.name)
            continue
        try:
            pipe = tile_pipeline(
                _intersecting(tiles, bounds, buffer),
                bounds,
                expand_bounds(bounds, buffer),
                out_path,
                pixel_size,
            )
        except Exception as exc:  # noqa: BLE001 - per-tile isolation, logged
            LOGGER.warning("skipping DTM tile %s: pipeline setup failed (%s)", out_path.name, exc)
            continue
        jobs.append((pipe, out_path, bounds, pixel_size, layout.epsg))
    if skipped:
        LOGGER.info("%d of %d DTM tiles already processed; skipped", skipped, len(tiles))
    _run_jobs(jobs, cores, desc="DTM tiles")

    # Keep only tiles actually written: a tile whose fallback write also
    # failed is excluded instead of breaking the mosaic or the index.
    written = [tile for tile in tiles if tile[3].exists()]
    missing = [tile[3].name for tile in tiles if not tile[3].exists()]
    if missing:
        LOGGER.warning(
            "%d of %d DTM tiles were not written: %s", len(missing), len(tiles), ", ".join(missing)
        )
    if not written:
        raise RuntimeError(f"no DTM tiles could be written for area {area!r}")
    tile_paths = [out_path for _, _, _, out_path in written]
    build_vrt(tile_paths, dtm_dir / "dtm.vrt")
    # Tile index with the LAZ-stem tile IDs and the (unbuffered) tile
    # footprints — the output tiles are clipped to the original tile bounds.
    index = gpd.GeoDataFrame(
        {
            "tile_id": [tile_id for tile_id, _, _, _ in written],
            "path": [str(path) for path in tile_paths],
            "geometry": [box(*bounds) for _, bounds, _, _ in written],
        },
        crs=layout.epsg,
    )
    index.to_file(tile_index_path(dtm_dir), layer=LAYER_NAME)
    LOGGER.info(
        "DTM product for %s: %d tiles in product (%d skipped as already processed)",
        area,
        len(tile_paths),
        skipped,
    )
    return tile_paths


def _intersecting(
    tiles: list[tuple[str, Bounds, Path, Path]], bounds: Bounds, buffer: float
) -> list[Path]:
    """Original LAZ paths whose footprint intersects ``bounds`` (+ buffer)."""
    minx, miny, maxx, maxy = expand_bounds(bounds, buffer)
    return [
        source
        for _tile_id, (ominx, ominy, omaxx, omaxy), source, _out in tiles
        if ominx <= maxx and omaxx >= minx and ominy <= maxy and omaxy >= miny
    ]
