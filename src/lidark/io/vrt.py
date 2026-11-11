"""GDAL VRT mosaics over tiled products.

VRTs are created with the standalone ``gdalbuildvrt`` binary (``gdal-utils``)
and only *read* through rasterio. The plan deliberately avoids importing
``osgeo.gdal`` alongside rasterio: rasterio wheels bundle their own GDAL and
mixing the two can crash the interpreter.

The tile list is passed to ``gdalbuildvrt`` via ``-input_file_list`` (a
temporary text file, one absolute path per line), never as command-line
arguments: Windows caps the CreateProcess command line at 32767 characters,
which a few thousand tile paths exceed (WinError 206).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import rasterio

if TYPE_CHECKING:
    import rasterio.io


def _gdalbuildvrt_path() -> str:
    """Locate the gdalbuildvrt executable.

    Tries ``PATH`` first, then the usual conda environment layouts relative to
    the running interpreter.
    """
    exe = shutil.which("gdalbuildvrt")
    if exe:
        return exe
    env_root = Path(sys.executable).parent
    candidates = [
        env_root / "Library" / "bin" / "gdalbuildvrt.exe",  # Windows conda
        env_root / "bin" / "gdalbuildvrt",  # Linux conda / venv
        env_root / "Scripts" / "gdalbuildvrt.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "gdalbuildvrt not found. Install gdal-utils (e.g. conda-forge) or ensure it is on PATH."
    )


def build_vrt(tile_paths: Sequence[Path | str], out_path: Path | str) -> Path:
    """Build a VRT mosaic of ``tile_paths`` with gdalbuildvrt.

    The tile list is passed via ``-input_file_list`` (one absolute path per
    line in a temporary UTF-8 file) instead of as command-line arguments:
    Windows caps the CreateProcess command line at 32767 characters, so with
    thousands of tiles the path list no longer fits and the spawn fails with
    WinError 206 ("filename or extension too long") before gdalbuildvrt
    starts.

    Args:
        tile_paths: Raster tiles to mosaic.
        out_path: Output ``.vrt`` path; parent directories are created.

    Returns:
        The output path.

    Raises:
        ValueError: If ``tile_paths`` is empty.
        RuntimeError: If gdalbuildvrt fails; its stderr is included.
    """
    if not tile_paths:
        raise ValueError("tile_paths is empty")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exe = _gdalbuildvrt_path()
    list_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            prefix="lidark_vrt_inputs_",
            delete=False,
        ) as list_file:
            list_path = Path(list_file.name)
            for p in tile_paths:
                list_file.write(f"{Path(p).absolute()}\n")
        cmd = [exe, "-overwrite", "-input_file_list", str(list_path), str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    finally:
        if list_path is not None:
            list_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"gdalbuildvrt failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return out


def find_vrt(product_dir: Path | str) -> Path | None:
    """The product's VRT mosaic path, if one exists.

    Prefers the product-specific names (``tpi.vrt``, ``dtm.vrt``) written by
    the map generators, then any other ``.vrt`` in the directory.
    """
    product_dir = Path(product_dir)
    for name in ("tpi.vrt", "dtm.vrt"):
        if (product_dir / name).exists():
            return product_dir / name
    vrts = sorted(product_dir.glob("*.vrt"))
    return vrts[0] if vrts else None


def open_vrt(vrt_path: Path | str) -> rasterio.io.DatasetReader:
    """Open a VRT mosaic with rasterio for windowed reads.

    The returned dataset is a context manager; use it for single windowed
    reads across the whole mosaic instead of iterating tiles.
    """
    return cast("rasterio.io.DatasetReader", rasterio.open(vrt_path))
