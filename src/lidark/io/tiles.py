"""Tile index: footprints plus opaque tile IDs.

Replaces the original fixed 9-character mapsheet-name assumption. Tile
identity is the filename stem — an opaque label of arbitrary length carrying
no mapsheet information. Tiles from different layers are matched spatially
(footprint intersection), never by name prefix.

The index is a GeoPackage (``<product_dir>/tiles.gpkg``) with columns
``tile_id``, ``path``, and a footprint geometry per tile.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

INDEX_FILENAME = "tiles.gpkg"
LAYER_NAME = "tiles"


def tile_index_path(product_dir: Path | str) -> Path:
    """Path of the tile index of a product directory."""
    return Path(product_dir) / INDEX_FILENAME


def build_tile_index(
    product_dir: Path | str,
    *,
    default_epsg: str = "EPSG:3067",
) -> gpd.GeoDataFrame:
    """Scan the GeoTIFF tiles of a product directory and write the index.

    Args:
        product_dir: Directory holding the product's ``.tif`` tiles.
        default_epsg: CRS assigned to tiles that carry no CRS metadata.

    Returns:
        GeoDataFrame with ``tile_id`` (filename stem, opaque), ``path``
        (absolute), and the tile footprint geometry; also written to
        ``<product_dir>/tiles.gpkg``.

    Raises:
        FileNotFoundError: If the directory holds no tiles.
        ValueError: If the tiles disagree on CRS.
    """
    product_dir = Path(product_dir)
    paths = sorted(
        (p for p in product_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"}),
        key=lambda p: p.name.lower(),
    )
    if not paths:
        raise FileNotFoundError(f"no GeoTIFF tiles found in {product_dir}")

    reference_crs: CRS | None = None
    records: list[dict[str, object]] = []
    for path in paths:
        with rasterio.open(path) as src:
            crs = src.crs
            if crs is None:
                crs = CRS.from_string(default_epsg)
            if reference_crs is None:
                reference_crs = crs
            elif crs != reference_crs:
                raise ValueError(
                    f"CRS mismatch in {product_dir}: {path.name} has {crs}, "
                    f"expected {reference_crs}"
                )
            footprint = box(*cast("tuple[float, float, float, float]", src.bounds))
        records.append(
            {
                "tile_id": path.stem,
                "path": str(path.resolve()),
                "geometry": footprint,
            }
        )
    if reference_crs is None:  # unreachable: at least one tile exists
        raise ValueError("no CRS could be determined")

    index = gpd.GeoDataFrame(records, crs=reference_crs)
    index.to_file(tile_index_path(product_dir), layer=LAYER_NAME)
    return index


def load_tile_index(product_dir: Path | str) -> gpd.GeoDataFrame:
    """Load the tile index of a product directory.

    Raises:
        FileNotFoundError: If the index does not exist (run
            :func:`build_tile_index` first).
    """
    path = tile_index_path(product_dir)
    if not path.exists():
        raise FileNotFoundError(f"tile index not found: {path}")
    return gpd.read_file(path, layer=LAYER_NAME)


def tile_path(index: gpd.GeoDataFrame, tile_id: str) -> Path:
    """Resolve the raster path of a tile ID.

    Raises:
        KeyError: If the tile ID is not in the index.
    """
    rows = index[index["tile_id"] == tile_id]
    if len(rows) == 0:
        raise KeyError(f"tile_id {tile_id!r} not in tile index")
    return Path(str(rows["path"].iloc[0]))


def intersecting_tiles(index: gpd.GeoDataFrame, geometry: BaseGeometry) -> gpd.GeoDataFrame:
    """Tiles whose footprint intersects ``geometry`` (STRtree-backed)."""
    positions = np.asarray(index.sindex.query(geometry, predicate="intersects"))
    return index.iloc[positions]
