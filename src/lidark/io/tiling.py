"""Window tiling for prediction: original 512/128 semantics plus stitching.

Replaces the inline double loop of ``Predict.pred_mapsheet``. Window centers
reproduce the original ``range(window // 2, dim - window // 2, stride)`` with
``stride = window // 4`` by default, and stitching blends overlapping window
results by per-pixel maximum (the original max-probability semantics) using
``np.maximum`` on preallocated window views — no per-window Python arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeAlias

import numpy as np

from lidark._types import Array

Center: TypeAlias = tuple[int, int]


class WindowTiler:
    """Generate sliding windows over a raster and stitch results back.

    Attributes:
        window_size: Window (height, width) in pixels (default 512x512, the
            original training/prediction size).
        stride: Step between window centers in pixels. The original used
            ``window // 4`` (= 128 for a 512 window), which overlaps adjacent
            windows by 75%.
    """

    def __init__(self, window_size: int | tuple[int, int] = 512, stride: int | None = None) -> None:
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        height, width = window_size
        if height <= 0 or width <= 0:
            raise ValueError(f"window size must be positive, got {window_size}")
        self.window_size: tuple[int, int] = (height, width)
        self.stride = stride if stride is not None else max(height, width) // 4
        if self.stride <= 0 or self.stride > max(height, width):
            raise ValueError(f"stride must be in [1, {max(height, width)}], got {self.stride}")

    @property
    def half(self) -> tuple[int, int]:
        """Half window sizes (height, width)."""
        return self.window_size[0] // 2, self.window_size[1] // 2

    def centers(self, height: int, width: int) -> list[Center]:
        """Window centers (row, col) over a raster of the given shape.

        Replicates the original iteration
        ``range(window // 2, dim - window // 2, stride)``; the top/left and
        bottom/right edge strips of width ``stride``-worth of pixels beyond
        the last center remain uncovered, matching the original behavior.

        Raises:
            ValueError: If the raster is smaller than the window.
        """
        half_h, half_w = self.half
        if height < self.window_size[0] or width < self.window_size[1]:
            raise ValueError(f"raster {height}x{width} smaller than window {self.window_size}")
        rows = range(half_h, height - half_h, self.stride)
        cols = range(half_w, width - half_w, self.stride)
        return [(row, col) for row in rows for col in cols]

    def iter_windows(self, arr: Array) -> Iterator[tuple[Center, Array]]:
        """Yield ``(center, window_view)`` pairs over a 2-D array."""
        half_h, half_w = self.half
        for row, col in self.centers(arr.shape[0], arr.shape[1]):
            yield (row, col), arr[row - half_h : row + half_h, col - half_w : col + half_w]

    def stitch(
        self,
        shape: tuple[int, int],
        tiles: Sequence[tuple[Center, Array]],
    ) -> Array:
        """Blend window results back by per-pixel maximum.

        Args:
            shape: Full raster shape (height, width).
            tiles: ``(center, result)`` pairs, e.g. from model predictions on
                the windows of :meth:`iter_windows`.

        Returns:
            Float32 array of ``shape`` holding the maximum overlapping window
            value per pixel.
        """
        prob = np.zeros(shape, dtype=np.float32)
        self.stitch_into(prob, tiles)
        return prob

    def stitch_into(self, prob: Array, tiles: Sequence[tuple[Center, Array]]) -> None:
        """Blend window results into a preallocated array, in place.

        Batch-friendly variant of :meth:`stitch`: call repeatedly per
        inference batch — the per-pixel maximum is order-independent.
        """
        half_h, half_w = self.half
        for (row, col), tile in tiles:
            r0, r1 = row - half_h, row + half_h
            c0, c1 = col - half_w, col + half_w
            view = prob[r0:r1, c0:c1]
            np.maximum(view, tile, out=view)
