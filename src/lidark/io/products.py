"""Ensure tiled products carry their tile index and VRT mosaic on load.

The map generators write ``tiles.gpkg`` and the product VRT mosaic
alongside the tiles; the loaders (TPI driver, dataset creation, predict)
previously failed outright when either artifact was missing — a copied
project, a partially synced drive, a hand-built product. These helpers
rebuild the missing piece from the GeoTIFF tiles on disk before loading:

- The **tile index** is required everywhere: it is loaded when present
  and rebuilt when missing or stale (its indexed path set no longer
  matches the tiles on disk).
- The **VRT mosaic** is rebuilt when missing; :func:`ensure_product`
  also rebuilds it after a stale-index rebuild (its tile set may have
  changed). Loaders that use the VRT only as a read optimization (train,
  predict) may treat a failed build as a warning and fall back to
  per-tile reads through the index.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from lidark.config import DEFAULT_EPSG
from lidark.io.tiles import build_tile_index, load_tile_index, tile_index_path
from lidark.io.vrt import build_vrt, find_vrt
from lidark.logging import get_logger

LOGGER = get_logger("io.products")

_TILE_SUFFIXES = {".tif", ".tiff"}


def _tiles_on_disk(product_dir: Path) -> list[Path]:
    """The product's GeoTIFF tiles in deterministic (case-insensitive) order."""
    return sorted(
        (p for p in product_dir.iterdir() if p.suffix.lower() in _TILE_SUFFIXES),
        key=lambda p: p.name.lower(),
    )


def _generate_first_error(product_dir: Path, reason: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"{reason}: {product_dir}; generate the product first (lidark create-map)"
    )


def _ensure_tile_index(product_dir: Path, default_epsg: str) -> tuple[gpd.GeoDataFrame, bool]:
    """Load the tile index; rebuild it when missing or stale.

    Returns ``(index, rebuilt)`` — ``rebuilt`` is True when the index was
    (re)written to disk, i.e. a stale VRT should be rebuilt with it.

    Raises:
        FileNotFoundError: If the directory does not exist or holds no
            tiles.
    """
    if not product_dir.exists():
        raise _generate_first_error(product_dir, "product directory not found")
    index_file = tile_index_path(product_dir)
    tiles = _tiles_on_disk(product_dir)
    if index_file.exists():
        index = load_tile_index(product_dir)
        indexed = {Path(str(p)).resolve() for p in index["path"]}
        on_disk = {p.resolve() for p in tiles}
        if indexed == on_disk:
            return index, False
        LOGGER.info(
            "tile index of %s is out of date (%d indexed vs %d tiles on disk); rebuilding",
            product_dir,
            len(indexed),
            len(on_disk),
        )
    if not tiles:
        raise _generate_first_error(product_dir, "no GeoTIFF tiles found")
    LOGGER.info("tile index missing in %s; building from %d tiles", product_dir, len(tiles))
    index = build_tile_index(product_dir, default_epsg=default_epsg)
    return index, True


def _vrt_filename(product_dir: Path) -> str:
    """The product-specific VRT name (:func:`find_vrt` prefers these)."""
    name = product_dir.name
    if name.startswith("radius_"):
        return "tpi.vrt"
    if "dtm" in name.lower():
        return "dtm.vrt"
    return "mosaic.vrt"


def _build_vrt_for(index: gpd.GeoDataFrame, product_dir: Path) -> Path:
    LOGGER.info("VRT mosaic missing in %s; building from %d tiles", product_dir, len(index))
    return build_vrt(
        [Path(str(p)) for p in index["path"]],
        product_dir / _vrt_filename(product_dir),
    )


def ensure_tile_index(
    product_dir: Path | str, *, default_epsg: str = DEFAULT_EPSG
) -> gpd.GeoDataFrame:
    """The product's tile index, loaded or rebuilt from the tiles on disk.

    A stale index (indexed paths no longer matching the tiles on disk) is
    rebuilt like a missing one.

    Raises:
        FileNotFoundError: If the directory does not exist or holds no
            tiles — the product must be generated first.
    """
    return _ensure_tile_index(Path(product_dir), default_epsg)[0]


def ensure_vrt(
    product_dir: Path | str,
    index: gpd.GeoDataFrame | None = None,
    *,
    default_epsg: str = DEFAULT_EPSG,
) -> Path:
    """The product's VRT mosaic path, building the mosaic when missing.

    ``index`` may be an already-ensured tile index (skips a redundant
    load). A present VRT is trusted as-is; :func:`ensure_product` is the
    variant that rebuilds it after a stale-index rebuild.

    Raises:
        FileNotFoundError: If the tiles or the ``gdalbuildvrt`` binary
            are missing.
        RuntimeError: If gdalbuildvrt fails.
    """
    product_dir = Path(product_dir)
    existing = find_vrt(product_dir)
    if existing is not None:
        return existing
    if index is None:
        index = ensure_tile_index(product_dir, default_epsg=default_epsg)
    return _build_vrt_for(index, product_dir)


def ensure_product(
    product_dir: Path | str, *, default_epsg: str = DEFAULT_EPSG
) -> tuple[Path, gpd.GeoDataFrame]:
    """The product's ``(vrt_path, tile_index)``, building missing pieces.

    Both artifacts are checked against the tiles on disk: a missing or
    stale tile index is rebuilt, and the VRT mosaic is (re)built when
    missing or when the index was just rebuilt (its tile set may have
    changed). Use :func:`ensure_tile_index` and :func:`ensure_vrt`
    separately when only one artifact is needed or a VRT build failure
    should degrade to per-tile reads.

    Raises:
        FileNotFoundError: If the directory holds no tiles or the
            ``gdalbuildvrt`` binary is missing.
        RuntimeError: If gdalbuildvrt fails.
    """
    product_dir = Path(product_dir)
    index, index_rebuilt = _ensure_tile_index(product_dir, default_epsg)
    existing = find_vrt(product_dir)
    if existing is not None and not index_rebuilt:
        return existing, index
    return _build_vrt_for(index, product_dir), index


__all__ = ["ensure_product", "ensure_tile_index", "ensure_vrt"]
