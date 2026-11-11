"""Minimal LAS 1.2 writer for test fixtures (no laspy dependency).

Writes point format 0 records (20 bytes) with an uncompressed LAS 1.2 header
(227 bytes). Just enough for PDAL to read the file and rasterize the points.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_HEADER_SIZE = 227
_POINT_FORMAT = 0
_POINT_SIZE = 20


def write_las(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
    *,
    scale: float = 0.01,
) -> None:
    """Write points to an uncompressed LAS 1.2 file.

    Args:
        path: Output file path (``.las``).
        x, y, z: Point coordinates (world units).
        classification: Per-point ASPRS classification code (2 = ground).
        scale: Integer quantization scale of the header.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    classification = np.asarray(classification, dtype=np.uint8)
    n = len(x)
    if not (len(y) == len(z) == len(classification) == n):
        raise ValueError("x, y, z, classification must have equal length")

    ix = np.rint(x / scale).astype("<i4")
    iy = np.rint(y / scale).astype("<i4")
    iz = np.rint(z / scale).astype("<i4")

    header = bytearray(_HEADER_SIZE)
    struct.pack_into("<4s", header, 0, b"LASF")
    struct.pack_into("<BB", header, 24, 1, 2)  # version 1.2
    struct.pack_into("<32s", header, 26, b"lidark-tests".ljust(32, b"\0"))
    struct.pack_into("<32s", header, 58, b"lidark-tests".ljust(32, b"\0"))
    struct.pack_into("<H", header, 90, 1)  # creation day
    struct.pack_into("<H", header, 92, 2026)  # creation year
    struct.pack_into("<H", header, 94, _HEADER_SIZE)
    struct.pack_into("<I", header, 96, _HEADER_SIZE)  # offset to point data
    struct.pack_into("<I", header, 100, 0)  # number of VLRs
    struct.pack_into("<B", header, 104, _POINT_FORMAT)
    struct.pack_into("<H", header, 105, _POINT_SIZE)
    struct.pack_into("<I", header, 107, n)
    for i in range(5):
        struct.pack_into("<I", header, 111 + 4 * i, n if i == 0 else 0)
    struct.pack_into("<ddd", header, 131, scale, scale, scale)
    struct.pack_into("<ddd", header, 155, 0.0, 0.0, 0.0)
    struct.pack_into(
        "<dddddd",
        header,
        179,
        float(ix.max() * scale) if n else 0.0,
        float(ix.min() * scale) if n else 0.0,
        float(iy.max() * scale) if n else 0.0,
        float(iy.min() * scale) if n else 0.0,
        float(iz.max() * scale) if n else 0.0,
        float(iz.min() * scale) if n else 0.0,
    )

    records = np.zeros(
        n,
        dtype=np.dtype(
            [
                ("x", "<i4"),
                ("y", "<i4"),
                ("z", "<i4"),
                ("intensity", "<u2"),
                ("flags", "u1"),
                ("classification", "u1"),
                ("scan_angle", "i1"),
                ("user_data", "u1"),
                ("source_id", "<u2"),
            ]
        ),
    )
    records["x"] = ix
    records["y"] = iy
    records["z"] = iz
    records["classification"] = classification

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(bytes(header))
        fh.write(records.tobytes())
