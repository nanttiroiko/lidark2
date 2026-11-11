"""Tests for lidark.io.tiling: window generation and max-prob stitching."""

import numpy as np
import pytest

from lidark.io.tiling import WindowTiler


def test_centers_match_original_range_semantics() -> None:
    # Original: range(window // 2, dim - window // 2, stride) per axis.
    tiler = WindowTiler(window_size=512, stride=128)
    centers = tiler.centers(1024, 1024)
    expected_rows = list(range(256, 1024 - 256, 128))
    expected_cols = list(range(256, 1024 - 256, 128))
    assert [r for r, _ in centers] == [r for r in expected_rows for _ in expected_cols]
    assert [c for _, c in centers] == [c for _ in expected_rows for c in expected_cols]
    assert len(centers) == len(expected_rows) * len(expected_cols)


def test_default_stride_is_window_div4() -> None:
    assert WindowTiler(512).stride == 128
    assert WindowTiler((256, 256)).stride == 64


def test_iter_windows_views() -> None:
    tiler = WindowTiler(window_size=8, stride=4)
    arr = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    windows = list(tiler.iter_windows(arr))
    assert len(windows) == len(tiler.centers(16, 16))
    for (row, col), view in windows:
        assert view.shape == (8, 8)
        assert np.array_equal(view, arr[row - 4 : row + 4, col - 4 : col + 4])


def test_stitch_equals_reference_max_blend() -> None:
    rng = np.random.default_rng(42)
    tiler = WindowTiler(window_size=16, stride=8)
    shape = (48, 48)
    tiles = [(center, rng.random((16, 16)).astype(np.float32)) for center in tiler.centers(*shape)]

    stitched = tiler.stitch(shape, tiles)

    # Reference: the original per-window mask arithmetic.
    ref = np.zeros(shape)
    half = 8
    for (row, col), tile in tiles:
        r0, r1 = row - half, row + half
        c0, c1 = col - half, col + half
        init = ref[r0:r1, c0:c1].copy()
        mask = (tile > init).astype(int)
        ref[r0:r1, c0:c1] = tile * mask + init * (mask == 0).astype(int)

    assert np.allclose(stitched, ref, atol=1e-6)


def test_stitch_uncovered_edges_are_zero() -> None:
    tiler = WindowTiler(window_size=8, stride=4)
    shape = (16, 16)
    tiles = [(c, np.ones((8, 8), dtype=np.float32)) for c in tiler.centers(*shape)]
    stitched = tiler.stitch(shape, tiles)
    # Centers are at 4 and 8: rows/cols 0..11 are covered, 12..15 never written.
    assert np.all(stitched[:12, :12] == 1.0)
    assert np.all(stitched[12:, :] == 0)
    assert np.all(stitched[:, 12:] == 0)


def test_raster_smaller_than_window_raises() -> None:
    tiler = WindowTiler(window_size=32)
    with pytest.raises(ValueError, match="smaller than window"):
        tiler.centers(16, 16)


def test_stride_validation() -> None:
    with pytest.raises(ValueError, match="stride"):
        WindowTiler(window_size=16, stride=0)
    with pytest.raises(ValueError, match="stride"):
        WindowTiler(window_size=16, stride=32)
    with pytest.raises(ValueError, match="positive"):
        WindowTiler(window_size=0)
