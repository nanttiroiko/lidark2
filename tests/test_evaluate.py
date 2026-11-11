"""Tests for lidark.evaluate: constructed label/prediction pairs."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from lidark.config import DEFAULT_EPSG, ProjectLayout
from lidark.evaluate import COUNTERS, evaluate


@pytest.fixture()
def layout(tmp_path: Path) -> ProjectLayout:
    return ProjectLayout(project_folder=tmp_path, epsg=DEFAULT_EPSG)


TIMESTAMP = "20260101-120000"
AREA = "Naljanka"


def _write_shapes(path: Path, geometries: list, class_ids: list[int] | None = None) -> None:
    frame = gpd.GeoDataFrame(geometry=geometries, crs=DEFAULT_EPSG)
    if class_ids is not None:
        frame["class_id"] = class_ids
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path)


def _write_predictions(
    layout: ProjectLayout,
    geometries: list,
    scores: list[float],
    class_ids: list[int],
    pred_type: str = "all",
) -> None:
    frame = gpd.GeoDataFrame(
        {
            "score": scores,
            "class_id": class_ids,
            "class_name": [f"class_{c}" for c in class_ids],
            "geometry": geometries,
        },
        crs=DEFAULT_EPSG,
    )
    frame["area"] = frame.geometry.area
    out = layout.predictions_dir(TIMESTAMP, AREA) / f"merged_shapes_{pred_type}.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(out)


def _write_annotations(
    layout: ProjectLayout,
    eval_name: str,
    true_geometry: list,
    true_classes: list[int],
    false_geometry: list | None = None,
) -> None:
    annotations = layout.evaluate_annotations_dir(eval_name)
    _write_shapes(annotations / "true.shp", true_geometry, true_classes)
    if false_geometry is not None:
        _write_shapes(annotations / "false.shp", false_geometry)


def test_evaluate_six_counts_single_class(layout: ProjectLayout) -> None:
    # Labels: A (fully covered), B (partly covered), C (not covered);
    # F: a false object covered by P3; P4 touches no label at all.
    _write_annotations(
        layout,
        "test1",
        [box(0, 0, 10, 10), box(20, 0, 30, 10), box(100, 0, 110, 10)],
        [1, 1, 1],
        false_geometry=[box(0, 20, 10, 30)],
    )
    _write_predictions(
        layout,
        [
            box(0, 0, 10, 10),  # P1: covers A fully
            box(20, 0, 25, 10),  # P2: covers B by 50%
            box(0, 20, 10, 30),  # P3: covers the false object
            box(40, 0, 50, 10),  # P4: no label
        ],
        scores=[0.9, 0.9, 0.9, 0.9],
        class_ids=[1, 1, 1, 1],
    )

    report = evaluate(
        layout,
        "true.shp",
        TIMESTAMP,
        "test1",
        false_objects="false.shp",
    )
    row = report.loc[report["class_id"] == 1].iloc[0]
    assert row["Predicted"] == 4
    assert row["Fully detected"] == 1
    assert row["Partly detected"] == 1
    assert row["Not detected"] == 1
    assert row["Detected with no label"] == 1
    assert row["False detected"] == 1

    overall = report.loc[report["class_id"] == "overall"].iloc[0]
    assert overall["Predicted"] == 4
    assert overall["Fully detected"] == 1

    # Report CSV written at the original location.
    report_path = layout.reports_dir(TIMESTAMP) / "test1.csv"
    assert report_path.exists()
    on_disk = pd.read_csv(report_path)
    assert set(COUNTERS) <= set(on_disk.columns)


def test_evaluate_prob_and_size_cuts(layout: ProjectLayout) -> None:
    _write_annotations(layout, "cut", [box(0, 0, 10, 10)], [1])
    _write_predictions(
        layout,
        [box(0, 0, 10, 10), box(20, 0, 30, 10)],
        scores=[0.9, 0.5],  # second detection below prob_cut 0.8
        class_ids=[1, 1],
    )
    report = evaluate(layout, "true.shp", TIMESTAMP, "cut", prob_cut=0.8)
    row = report.loc[report["class_id"] == 1].iloc[0]
    assert row["Predicted"] == 1
    assert row["Fully detected"] == 1

    report2 = evaluate(layout, "true.shp", TIMESTAMP, "cut", prob_cut=0.8, size_cut=500.0)
    row2 = report2.loc[report2["class_id"] == 1].iloc[0]
    assert row2["Predicted"] == 0  # every detection smaller than the cut


def test_evaluate_multiclass_rows(layout: ProjectLayout) -> None:
    _write_annotations(
        layout,
        "multi",
        [box(0, 0, 10, 10), box(20, 0, 30, 10)],
        [1, 2],
    )
    _write_predictions(
        layout,
        [box(0, 0, 10, 10), box(20, 0, 30, 10)],
        scores=[0.9, 0.9],
        class_ids=[1, 2],
    )
    report = evaluate(layout, "true.shp", TIMESTAMP, "multi")
    classes = report.loc[report["class_id"] != "overall", "class_id"].tolist()
    assert classes == [1, 2]
    for _, row in report.loc[report["class_id"] != "overall"].iterrows():
        assert row["Predicted"] == 1
        assert row["Fully detected"] == 1
        assert row["Not detected"] == 0
    overall = report.loc[report["class_id"] == "overall"].iloc[0]
    assert overall["Predicted"] == 2
    assert overall["Fully detected"] == 2


def test_evaluate_empty_predictions_of_class(layout: ProjectLayout) -> None:
    # Labels of two classes, but only class-1 predictions exist.
    _write_annotations(
        layout,
        "mixed",
        [box(0, 0, 10, 10), box(20, 0, 30, 10)],
        [1, 2],
    )
    _write_predictions(layout, [box(0, 0, 10, 10)], scores=[0.9], class_ids=[1])
    report = evaluate(layout, "true.shp", TIMESTAMP, "mixed")
    class_2 = report.loc[report["class_id"] == 2].iloc[0]
    assert class_2["Predicted"] == 0
    assert class_2["Not detected"] == 1
    assert class_2["Fully detected"] == 0


def test_evaluate_missing_files_raise(layout: ProjectLayout) -> None:
    with pytest.raises(FileNotFoundError, match="label file"):
        evaluate(layout, "absent.shp", TIMESTAMP, "nope")
    _write_annotations(layout, "no_preds", [box(0, 0, 10, 10)], [1])
    with pytest.raises(FileNotFoundError, match="merged prediction shapes"):
        evaluate(layout, "true.shp", TIMESTAMP, "no_preds")


def test_evaluate_aoi_clips_predictions(layout: ProjectLayout) -> None:
    # An explicit AOI excludes one detection entirely.
    _write_annotations(layout, "aoi", [box(0, 0, 10, 10)], [1])
    _write_predictions(
        layout,
        [box(0, 0, 10, 10), box(1000, 1000, 1010, 1010)],
        scores=[0.9, 0.9],
        class_ids=[1, 1],
    )
    _write_shapes(layout.evaluate_annotations_dir("aoi") / "aoi.shp", [box(0, 0, 20, 20)])
    report = evaluate(layout, "true.shp", TIMESTAMP, "aoi", eval_aoi="aoi.shp")
    row = report.loc[report["class_id"] == 1].iloc[0]
    assert row["Predicted"] == 1
    assert row["Detected with no label"] == 0


def test_counts_are_integers(layout: ProjectLayout) -> None:
    _write_annotations(layout, "ints", [box(0, 0, 10, 10)], [1])
    _write_predictions(layout, [box(0, 0, 10, 10)], scores=[1.0], class_ids=[1])
    report = evaluate(layout, "true.shp", TIMESTAMP, "ints")
    for counter in COUNTERS:
        assert report[counter].dtype == np.dtype("int64")
