"""Tests for lidark.io.detections."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from lidark.io.detections import (
    COLUMNS,
    empty_detections,
    is_empty,
    make_detections,
    validate_detections,
)


def test_make_detections_schema_and_area() -> None:
    detections = make_detections(
        [box(0, 0, 10, 5), box(20, 0, 30, 2)],
        scores=[0.9, 0.4],
        class_ids=[1, 2],
        crs="EPSG:3067",
    )
    assert list(detections.columns) == [*COLUMNS, "geometry"]
    assert detections["score"].tolist() == [0.9, 0.4]
    assert detections["class_id"].tolist() == [1, 2]
    # Default class names derive from the ids.
    assert detections["class_name"].tolist() == ["class_1", "class_2"]
    assert detections["area"].tolist() == [50.0, 20.0]
    assert detections.crs is not None
    validate_detections(detections)


def test_make_detections_broadcasts_scalars() -> None:
    detections = make_detections([box(0, 0, 4, 4), box(8, 0, 12, 4)], scores=0.7, crs="EPSG:3067")
    assert detections["score"].tolist() == [0.7, 0.7]
    assert detections["class_id"].tolist() == [1, 1]
    assert detections["class_name"].tolist() == ["class_1", "class_1"]
    assert detections["area"].tolist() == [16.0, 16.0]


def test_make_detections_named_classes() -> None:
    detections = make_detections([box(0, 0, 4, 4)], scores=0.5, class_ids=2, class_names="tar_kiln")
    assert detections["class_name"].tolist() == ["tar_kiln"]


def test_make_detections_rejects_class_zero() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        make_detections([box(0, 0, 4, 4)], scores=0.5, class_ids=0)


def test_validate_rejects_missing_columns() -> None:
    broken = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:3067")
    with pytest.raises(ValueError, match="missing column"):
        validate_detections(broken)


def test_validate_rejects_out_of_range_score() -> None:
    detections = make_detections([box(0, 0, 1, 1)], scores=1.5)
    with pytest.raises(ValueError, match="score"):
        validate_detections(detections)


def test_empty_detections_has_schema() -> None:
    detections = empty_detections(crs="EPSG:3067")
    assert list(detections.columns) == [*COLUMNS, "geometry"]
    assert is_empty(detections)
    validate_detections(detections)
