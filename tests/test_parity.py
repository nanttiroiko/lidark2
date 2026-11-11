"""Golden parity tests: the vectorized kernel vs the original implementation.

The plan (section 6) requires proof that the vectorized TPI is numerically
identical to the original per-offset accumulation. Realistic DTM values are
float32 (from las2dem / writers.gdal); accumulated in float64, such sums are
exact regardless of addition order, so equality must be bit-exact here.
"""

import numpy as np
import pytest

from lidark.maps.tpi import tpi_kernel, tpi_postprocess, tpi_reference


def _synthetic_dtm(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Realistic DTM band: float32, positive elevations, some nodata (<= 0)."""
    rng = np.random.default_rng(seed)
    arr = rng.uniform(1.0, 500.0, size=shape).astype(np.float32)
    arr[rng.random(shape) < 0.1] = 0.0  # nodata
    arr[rng.random(shape) < 0.05] = -1.0  # negative values are also nodata
    return arr


@pytest.mark.parametrize("radius_px", [1, 2, 3])
@pytest.mark.parametrize("shape", [(24, 24), (32, 17)])
def test_kernel_bit_identical_to_original_float64(radius_px: int, shape: tuple) -> None:
    dtm = _synthetic_dtm(shape, seed=radius_px * 10 + shape[0])
    out = tpi_kernel(dtm, radius_px, dtype=np.float64)
    ref = tpi_reference(dtm, radius_px, dtype=np.float64)
    np.testing.assert_array_equal(out, ref)


def test_kernel_float32_close_to_reference() -> None:
    # float32 accumulation trades exactness for bandwidth; values stay close.
    dtm = _synthetic_dtm((32, 32), seed=5)
    out = tpi_kernel(dtm, 3, dtype=np.float32)
    ref = tpi_reference(dtm, 3, dtype=np.float64)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-3)


def test_full_pipeline_parity() -> None:
    # Post-processed int16 output must be identical too.
    dtm = _synthetic_dtm((40, 40), seed=9)
    out = tpi_postprocess(tpi_kernel(dtm, 2, dtype=np.float64))
    ref = tpi_postprocess(tpi_reference(dtm, 2, dtype=np.float64))
    np.testing.assert_array_equal(out, ref)


def test_window_larger_than_raster_is_well_defined() -> None:
    # A window larger than the raster clips to the available cells: a flat
    # plateau has TPI 0 everywhere. The original per-offset loop crashes on
    # this degenerate case (empty/negative view shapes); the vectorized
    # kernel is well-defined.
    dtm = np.full((3, 3), 10.0, dtype=np.float32)
    out = tpi_kernel(dtm, 5, dtype=np.float64)
    assert np.allclose(out, 0.0)


def test_torch_kernel_matches_cpu() -> None:
    """The batched torch path (GPU TPI, 7, P3) equals the CPU kernel:
    batch > 1, nodata cells, and area borders included.

    Regression tests for two bugs the first torch run exposed: broadcasting
    the (B, H, W) input against (B, 1, H, W) conv results (silently
    producing (B, B, H, W)) and clamping fractional weighted counts
    (the circular filter's 0.5 edge ring) to 1.
    """
    pytest.importorskip("torch", reason="torch not installed (lidark[torch])")
    from lidark.maps.tpi import _tpi_kernel_torch

    rng = np.random.default_rng(11)
    batch = rng.uniform(1.0, 500.0, size=(3, 24, 24)).astype(np.float32)
    batch[rng.random((3, 24, 24)) < 0.15] = 0.0  # nodata cells
    for radius_px in (1, 2, 3):
        cpu = np.stack([tpi_kernel(img, radius_px, dtype=np.float32) for img in batch])
        gpu = _tpi_kernel_torch(batch, radius_px)
        np.testing.assert_array_equal(np.isnan(cpu), np.isnan(gpu))
        np.testing.assert_allclose(cpu, gpu, rtol=1e-4, atol=1e-2)
