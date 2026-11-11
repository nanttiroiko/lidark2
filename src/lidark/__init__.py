"""lidark: LiDAR-based detection of archaeological features with deep learning.

Modernized implementation of the LIDARK project script. See
MODERNIZATION_PLAN.md for the design and workstream breakdown.
"""

import os

# Windows MKL workaround: conda-forge numpy/scipy link libiomp (OpenMP) and
# the pip-installed torch wheels bundle their own copy — with both loaded in
# one process (GPU TPI or the maskrcnn backend next to numpy), BLAS calls
# can hard-abort with "Fatal Python error: Aborted". This documented Intel
# switch makes the duplicate runtime tolerable; it must be set before torch
# is first imported. ``setdefault`` respects an explicit user override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"

from lidark.evaluate import evaluate
from lidark.logging import configure_logging, get_logger
from lidark.maps import create_map
from lidark.predict import predict
from lidark.train import train

__all__ = [
    "__version__",
    "configure_logging",
    "get_logger",
    "create_map",
    "train",
    "predict",
    "evaluate",
]
