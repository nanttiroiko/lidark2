"""Shared type aliases."""

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy.typing as npt

# A raster array of any shape and dtype.
Array: TypeAlias = npt.NDArray[Any]

# A per-class operating point: one value for all classes, or a
# {class_id: value} mapping (plan 3.3/MC5).
ClassSpec: TypeAlias = float | Mapping[int, float]
