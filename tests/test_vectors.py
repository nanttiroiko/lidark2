"""Tests for lidark.io.vectors."""

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from lidark.io import vectors
from lidark.io.detections import make_detections
from lidark.io.vectors import merge_overlapping, polygon_scores, polygonize_mask

TRANSFORM = from_origin(0.0, 100.0, 1.0, 1.0)


def test_polygonize_mask_single_region() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 3:6] = 1
    polygons = polygonize_mask(mask, transform=TRANSFORM, crs="EPSG:3067")
    assert len(polygons) == 1
    # World bounds of the 3x3 region at 1 m pixels, origin (0, 100).
    bounds = polygons.geometry.iloc[0].bounds
    assert bounds == (3.0, 95.0, 6.0, 98.0)


def test_polygonize_mask_empty() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    polygons = polygonize_mask(mask, transform=TRANSFORM, crs="EPSG:3067")
    assert len(polygons) == 0


def test_polygonize_mask_two_regions() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    mask[5:7, 5:7] = 1
    polygons = polygonize_mask(mask, transform=TRANSFORM, crs="EPSG:3067")
    assert len(polygons) == 2


def test_polygon_scores_matches_manual_means() -> None:
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    # 1 m pixels, origin (0, 100): pixel centers at x+0.5, y-0.5.
    # a covers pixel centers (rows 0-1, cols 0-1); b covers rows 2-3, cols 2-3.
    a = box(0, 98, 2, 100)
    b = box(2, 96, 4, 98)
    scores = polygon_scores(_frame([a, b]), values, TRANSFORM)
    assert scores[0] == pytest.approx(values[0:2, 0:2].mean())
    assert scores[1] == pytest.approx(values[2:4, 2:4].mean())


def test_polygon_scores_empty_frame() -> None:
    values = np.zeros((4, 4))
    scores = polygon_scores(_frame([]), values, TRANSFORM)
    assert scores.shape == (0,)


def test_polygon_scores_disjoint_polygon_is_nan() -> None:
    values = np.ones((4, 4))
    far = box(1000, 1000, 1001, 1001)
    scores = polygon_scores(_frame([far]), values, TRANSFORM)
    assert np.isnan(scores[0])


def test_polygon_scores_strategies_agree_when_disjoint() -> None:
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    a = box(0, 98, 2, 100)
    b = box(2, 96, 4, 98)
    frame = _frame([a, b])
    by_strategy = {
        strategy: polygon_scores(frame, values, TRANSFORM, overlap=strategy)
        for strategy in ("independent", "exclusive", "priority")
    }
    for scores in by_strategy.values():
        np.testing.assert_allclose(scores, by_strategy["independent"])


def test_polygon_scores_touching_is_not_overlap() -> None:
    # Polygons sharing only a boundary: no contested pixel centers, so every
    # strategy takes the fast one-pass path and gives full-footprint means.
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    a = box(0, 98, 2, 100)
    b = box(2, 98, 4, 100)
    frame = _frame([a, b])
    assert not vectors._has_area_overlap(frame)
    for strategy in ("independent", "exclusive", "priority"):
        scores = polygon_scores(frame, values, TRANSFORM, overlap=strategy)
        assert scores[0] == pytest.approx(values[0:2, 0:2].mean())
        assert scores[1] == pytest.approx(values[0:2, 2:4].mean())


def test_polygon_scores_independent_overlapping() -> None:
    # 8x8 values, 1 m pixels, origin (0, 8).
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    transform = from_origin(0, 8, 1, 1)
    # a covers rows 0-3, cols 0-3; b covers rows 0-3, cols 2-5 (overlap cols 2-3).
    a = box(0, 4, 4, 8)
    b = box(2, 4, 6, 8)
    scores = polygon_scores(_frame([a, b]), values, transform, overlap="independent")
    # Full-footprint means: contested pixels count for both.
    assert scores[0] == pytest.approx(values[0:4, 0:4].mean())
    assert scores[1] == pytest.approx(values[0:4, 2:6].mean())


def test_polygon_scores_priority_overlapping() -> None:
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    transform = from_origin(0, 8, 1, 1)
    a = box(0, 4, 4, 8)
    b = box(2, 4, 6, 8)
    scores = polygon_scores(_frame([a, b]), values, transform, overlap="priority")
    # b (rasterized last) claims the contested cols 2-3; a loses them.
    assert scores[0] == pytest.approx(values[0:4, 0:2].mean())
    assert scores[1] == pytest.approx(values[0:4, 2:6].mean())


def test_polygon_scores_exclusive_overlapping() -> None:
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    transform = from_origin(0, 8, 1, 1)
    a = box(0, 4, 4, 8)
    b = box(2, 4, 6, 8)
    scores = polygon_scores(_frame([a, b]), values, transform, overlap="exclusive")
    # Contested cols 2-3 split at the midpoint: col 2 -> a, col 3 -> b.
    assert scores[0] == pytest.approx(values[0:4, 0:3].mean())
    assert scores[1] == pytest.approx(values[0:4, 3:6].mean())


def test_polygon_scores_exclusive_fully_covered_is_nan() -> None:
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    transform = from_origin(0, 8, 1, 1)
    outer = box(0, 4, 6, 8)
    inner = box(1, 5, 3, 7)
    scores = polygon_scores(_frame([outer, inner]), values, transform, overlap="exclusive")
    assert scores[0] == pytest.approx(values[0:4, 0:6].mean())
    assert np.isnan(scores[1])


def test_polygon_scores_independent_off_raster_is_nan() -> None:
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    transform = from_origin(0, 8, 1, 1)
    far = box(100, 100, 102, 102)
    scores = polygon_scores(_frame([far]), values, transform, overlap="independent")
    assert np.isnan(scores[0])


def test_polygon_scores_rejects_unknown_strategy() -> None:
    values = np.zeros((4, 4))
    with pytest.raises(ValueError, match="overlap"):
        polygon_scores(_frame([box(0, 0, 1, 1)]), values, TRANSFORM, overlap="bogus")


def _frame(geoms):
    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:3067")


def test_merge_overlapping_same_class() -> None:
    detections = make_detections(
        [box(0, 0, 10, 10), box(5, 0, 15, 10)],
        scores=[0.9, 0.6],
        crs="EPSG:3067",
    )
    merged = merge_overlapping(detections)
    assert len(merged) == 1
    assert merged.geometry.iloc[0].area == pytest.approx(150.0)
    assert merged["area"].iloc[0] == pytest.approx(150.0)
    # Attributes come from the first overlapping detection (dissolve default).
    assert merged["score"].iloc[0] == pytest.approx(0.9)


def test_merge_overlapping_keeps_classes_apart() -> None:
    detections = make_detections(
        [box(0, 0, 10, 10), box(5, 0, 15, 10)],
        scores=[0.9, 0.6],
        class_ids=[1, 2],
        crs="EPSG:3067",
    )
    merged = merge_overlapping(detections)
    assert len(merged) == 2
    assert set(merged["class_id"]) == {1, 2}


def test_merge_overlapping_disjoint_stay_separate() -> None:
    detections = make_detections(
        [box(0, 0, 10, 10), box(20, 0, 30, 10)],
        scores=[0.9, 0.6],
        crs="EPSG:3067",
    )
    merged = merge_overlapping(detections)
    assert len(merged) == 2
    assert merged["area"].tolist() == [100.0, 100.0]


def test_merge_overlapping_empty() -> None:
    from lidark.io.detections import empty_detections

    merged = merge_overlapping(empty_detections(crs="EPSG:3067"))
    assert len(merged) == 0
