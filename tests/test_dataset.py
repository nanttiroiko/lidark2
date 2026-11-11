"""Tests for lidark.dataset: spatial sampling, class labels, memmap round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box
from tests.conftest import (
    AREA,
    AREA2,
    AREA2_ORIGIN_X,
    AREA2_ORIGIN_Y,
    ORIGIN_X,
    ORIGIN_Y,
    TILE_SHAPE,
    write_annotation,
)

from lidark.config import DEFAULT_EPSG
from lidark.dataset import (
    LABELS_FILENAME,
    META_FILENAME,
    SAMPLES_FILENAME,
    create_dataset,
    dataset_dir,
    load_dataset,
)
from lidark.io.tiles import tile_index_path

IMG = (16, 16)
FEATURES = ("01_DTM", "02_TPI/radius_20")


def _tile_box(row: int, col: int, margin: float = 0.0):
    w = TILE_SHAPE[1] * 2.0
    h = TILE_SHAPE[0] * 2.0
    x0 = ORIGIN_X + col * w
    y_top = ORIGIN_Y - row * h
    return box(x0 + margin, y_top - h + margin, x0 + w - margin, y_top - margin)


@pytest.fixture()
def annotations(layout) -> None:
    """True labels in tile (0,0) and across the (0,0)-(0,1) seam; false in (1,1)."""
    write_annotation(
        layout,
        "true.shp",
        [
            _tile_box(0, 0, margin=24.0),  # strictly inside tile (0,0)
            box(ORIGIN_X + 50.0, ORIGIN_Y - 20.0, ORIGIN_X + 70.0, ORIGIN_Y - 8.0),
        ],
        class_ids=[1, 1],
    )
    write_annotation(
        layout,
        "false.shp",
        [_tile_box(1, 1, margin=20.0), _tile_box(1, 0, margin=20.0)],
    )


def test_create_dataset_windows_and_meta(
    layout, dtm_product: Path, tpi_layer: Path, annotations
) -> None:
    ds = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        "true.shp",
        "false.shp",
        img_size=IMG,
        dataset_id="d1",
        seed=3,
    )
    # true.shp: object 1 in tile chunk_a (2 windows), object 2 across the
    # chunk_a/survey seam (2 windows per tile after clipping); false.shp:
    # tiles Q + tile-number (2 windows each). 2 + 4 + 4 = 10 windows.
    assert ds.samples.shape == (10, *IMG, 2)
    assert ds.labels.shape == (10, *IMG)
    assert ds.samples.dtype == np.dtype("int8")
    assert ds.labels.dtype == np.dtype("uint8")
    assert ds.num_classes == 1

    meta = ds.meta
    assert meta["feature_list"] == list(FEATURES)
    assert meta["img_size"] == list(IMG)
    assert meta["epsg"] == DEFAULT_EPSG
    assert meta["num_classes"] == 1
    assert meta["seed"] == 3
    assert meta["source_label_files"]["true_objects"] == "true.shp"
    assert meta["excluded_tile_ids"] == []
    assert meta["test_area"] is None
    assert meta["samples_shape"] == list(ds.samples.shape)
    assert meta["labels_dtype"] == "uint8"

    # Files on disk.
    out_dir = dataset_dir(layout, "d1")
    assert (out_dir / SAMPLES_FILENAME).exists()
    assert (out_dir / LABELS_FILENAME).exists()
    assert (out_dir / META_FILENAME).exists()


def test_dataset_records_object_provenance(
    layout, dtm_product: Path, tpi_layer: Path, annotations
) -> None:
    """Object provenance for object-disjoint validation splits: per-window
    object ids and map footprints, the full bounds of every label object,
    and the reference pixel size.

    The seam-crossing true object (true_1) is clipped into two tiles but
    keeps one stable id and one (unclipped) bounds entry.
    """
    ds = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        "true.shp",
        "false.shp",
        img_size=IMG,
        dataset_id="prov",
        seed=3,
    )
    n = len(ds.samples)
    assert n == 10
    assert len(ds.meta["window_objects"]) == n
    assert len(ds.meta["window_bounds"]) == n
    assert ds.pixel_size == 2.0
    objects = ds.objects or {}
    assert set(objects) == {"true_0", "true_1", "false_0", "false_1"}
    # Full (unclipped) bounds: true_1 spans the tile seam in x.
    assert objects["true_1"]["class_id"] == 1
    assert objects["true_1"]["bounds"] == [
        ORIGIN_X + 50.0,
        ORIGIN_Y - 20.0,
        ORIGIN_X + 70.0,
        ORIGIN_Y - 8.0,
    ]
    assert objects["false_0"]["class_id"] == 0
    # The seam object contributed windows in two tiles under one id.
    assert ds.window_objects is not None
    assert ds.window_objects.count("true_1") == 4  # 2 windows per clipped piece
    # Window bounds are map-coordinate footprints of the window rectangles.
    assert ds.window_bounds is not None
    for bounds in ds.window_bounds:
        minx, miny, maxx, maxy = bounds
        assert maxx - minx == IMG[1] * 2.0
        assert maxy - miny == IMG[0] * 2.0


def test_label_windows_match_rasterized_labels(
    layout, dtm_product: Path, tpi_layer: Path, annotations
) -> None:
    ds = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        "true.shp",
        None,
        img_size=IMG,
        dataset_id="d2",
        seed=3,
    )
    # The strictly-interior object guarantees windows fully inside tile (0,0):
    # their labels must contain class 1 pixels; the DTM channel must match the
    # synthetic tile array exactly.
    import rasterio

    interior = [i for i, label in enumerate(ds.labels) if label.max() == 1]
    assert interior
    with rasterio.open(dtm_product / "chunk_a.tif") as src:
        dtm = src.read(1)
    for i in interior:
        assert (ds.labels[i] == 1).any()
        assert (ds.labels[i] <= 1).all()
        # DTM channel equals some 16x16 window of the tile raster.
        found = any(
            np.array_equal(
                ds.samples[i, :, :, 0],
                dtm[r : r + IMG[0], c : c + IMG[1]],
            )
            for r in range(TILE_SHAPE[0] - IMG[0] + 1)
            for c in range(TILE_SHAPE[1] - IMG[1] + 1)
        )
        assert found, f"sample {i} DTM channel is not a tile window"


def test_false_objects_give_background_windows(
    layout, dtm_product: Path, tpi_layer: Path, annotations
) -> None:
    ds = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        None,
        "false.shp",
        img_size=IMG,
        dataset_id="d3",
        seed=3,
    )
    assert len(ds.labels) > 0
    assert not ds.labels.any()  # all-background windows (original behaviour)


def test_class_aware_labels(layout, dtm_product: Path, tpi_layer: Path) -> None:
    write_annotation(
        layout,
        "multi.shp",
        [_tile_box(0, 0, margin=24.0), _tile_box(1, 0, margin=24.0)],
        class_ids=[1, 2],
    )
    ds = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        "multi.shp",
        None,
        img_size=IMG,
        dataset_id="d4",
        seed=3,
    )
    assert ds.num_classes == 2
    assert set(np.unique(ds.labels)) <= {0, 1, 2}
    assert 1 in np.unique(ds.labels)
    assert 2 in np.unique(ds.labels)
    assert ds.meta["class_names"] == ["class_1", "class_2"]

    # Missing class_id column defaults to class 1 (original binary case).
    write_annotation(layout, "plain.shp", [_tile_box(0, 0, margin=24.0)])
    plain = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        "plain.shp",
        None,
        img_size=IMG,
        dataset_id="d5",
        seed=3,
    )
    assert plain.num_classes == 1
    assert set(np.unique(plain.labels)) <= {0, 1}


def test_class_id_validation(layout, dtm_product: Path) -> None:
    write_annotation(layout, "bad.shp", [_tile_box(0, 0, margin=24.0)], class_ids=[0])
    with pytest.raises(ValueError, match="class_id"):
        create_dataset(layout, AREA, list(FEATURES), "bad.shp", None, img_size=IMG)


def test_determinism_and_seed(layout, dtm_product: Path, tpi_layer: Path, annotations) -> None:
    a = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        "true.shp",
        None,
        img_size=IMG,
        dataset_id="a",
        seed=7,
    )
    b = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        "true.shp",
        None,
        img_size=IMG,
        dataset_id="b",
        seed=7,
    )
    np.testing.assert_array_equal(a.samples, b.samples)
    np.testing.assert_array_equal(a.labels, b.labels)
    c = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        "true.shp",
        None,
        img_size=IMG,
        dataset_id="c",
        seed=8,
    )
    # Same objects but different placements -> at least some windows differ.
    assert not np.array_equal(a.samples[:, 0, 0, 0], c.samples[:, 0, 0, 0])


def test_test_area_exclusion(layout, dtm_product: Path, tpi_layer: Path) -> None:
    # Labels inside tiles Q and chunk_a; a test-area polygon layer over tile
    # Q excludes it, chunk_a keeps its windows.
    write_annotation(
        layout,
        "q.shp",
        [_tile_box(1, 0, margin=24.0), _tile_box(0, 0, margin=24.0)],
    )
    write_annotation(layout, "test_area.gpkg", [_tile_box(1, 0, margin=8.0)])
    ds = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        "q.shp",
        None,
        img_size=IMG,
        dataset_id="q",
        test_area="test_area.gpkg",
        seed=3,
    )
    assert ds.meta["excluded_tile_ids"] == ["Q"]
    assert ds.meta["test_area"] == "test_area.gpkg"
    # Only chunk_a windows remain: 1 object x 2 windows.
    assert len(ds.labels) == 2


def test_test_area_missing_layer_raises(layout, dtm_product: Path, tpi_layer: Path) -> None:
    write_annotation(layout, "true.shp", [_tile_box(0, 0, margin=24.0)])
    with pytest.raises(FileNotFoundError, match="polygon layer not found"):
        create_dataset(
            layout,
            AREA,
            list(FEATURES),
            "true.shp",
            None,
            img_size=IMG,
            test_area="absent.gpkg",
        )


def test_load_dataset_round_trip_is_memmap(
    layout, dtm_product: Path, tpi_layer: Path, annotations
) -> None:
    created = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        "true.shp",
        "false.shp",
        img_size=IMG,
        dataset_id="rt",
        seed=3,
    )
    loaded = load_dataset(layout, "rt")
    assert isinstance(loaded.samples, np.memmap)
    assert isinstance(loaded.labels, np.memmap)
    np.testing.assert_array_equal(loaded.samples, created.samples)
    np.testing.assert_array_equal(loaded.labels, created.labels)
    assert loaded.meta == created.meta


def test_load_missing_dataset_raises(layout) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_dataset(layout, "nope")


def test_no_intersecting_labels_raises(layout, dtm_product: Path, tpi_layer: Path) -> None:
    write_annotation(layout, "far.shp", [box(5_000_000.0, 5_000_000.0, 5_000_010.0, 5_000_010.0)])
    with pytest.raises(ValueError, match="no training windows"):
        create_dataset(layout, AREA, list(FEATURES), "far.shp", None, img_size=IMG)


def test_missing_labels_raise(layout, dtm_product: Path, tpi_layer: Path) -> None:
    with pytest.raises(FileNotFoundError, match="label file"):
        create_dataset(layout, AREA, list(FEATURES), "absent.shp", None, img_size=IMG)
    with pytest.raises(ValueError, match="feature_list"):
        create_dataset(layout, AREA, [], "absent.shp", None, img_size=IMG)


def test_feature_misalignment_raises(layout, dtm_product: Path, annotations) -> None:
    # A "feature layer" whose tiles are pixel-offset from the DTM.
    from affine import Affine

    from lidark.io.rasters import write_band
    from lidark.io.tiles import tile_index_path

    feature_dir = layout.lidar_area(AREA) / "02_TPI" / "radius_20"
    feature_dir.mkdir(parents=True, exist_ok=True)
    write_band(
        feature_dir / "chunk_a_TPI.tif",
        np.zeros(TILE_SHAPE, dtype=np.int16),
        transform=Affine(2.0, 0.0, ORIGIN_X + 1.0, 0.0, -2.0, ORIGIN_Y - 1.0),
        crs=DEFAULT_EPSG,
        nodata=None,
    )
    from lidark.io.tiles import load_tile_index

    index = load_tile_index(dtm_product)
    index = index.iloc[[0]].copy()
    index["path"] = [str(feature_dir / "chunk_a_TPI.tif")]
    index.to_file(tile_index_path(feature_dir), layer="tiles")
    with pytest.raises(ValueError, match="pixel-aligned"):
        create_dataset(
            layout,
            AREA,
            list(FEATURES),
            "true.shp",
            None,
            img_size=IMG,
            dataset_id="mis",
            seed=3,
        )


def test_create_dataset_multi_area(
    layout,
    dtm_product: Path,
    tpi_layer: Path,
    dtm_product2: Path,
    tpi_layer2: Path,
) -> None:
    """Several areas build one dataset: the shared labels match tiles
    spatially per area, and every area's windows land in the same
    samples/labels arrays."""
    write_annotation(
        layout,
        "both.shp",
        [
            _tile_box(0, 0, margin=24.0),  # inside AREA tile (0,0)
            box(
                AREA2_ORIGIN_X + 10.0,
                AREA2_ORIGIN_Y - 30.0,
                AREA2_ORIGIN_X + 40.0,
                AREA2_ORIGIN_Y - 10.0,
            ),  # inside AREA2 tile (0,0)
        ],
        class_ids=[1, 1],
    )
    ds = create_dataset(
        layout,
        [AREA, AREA2],
        list(FEATURES),
        "both.shp",
        None,
        img_size=IMG,
        dataset_id="multi",
        seed=3,
    )
    # 2 windows per object per area: the first two from AREA, the last two
    # from AREA2, each with its label pixels burned in.
    assert ds.samples.shape == (4, *IMG, 2)
    assert ds.meta["area"] == AREA
    assert ds.meta["areas"] == [AREA, AREA2]
    assert ds.meta["excluded_tile_ids"] == []
    assert ds.window_objects == ["true_0", "true_0", "true_1", "true_1"]
    assert int(ds.labels[:2].max()) == 1
    assert int(ds.labels[2:].max()) == 1

    # Excluded test tiles are namespaced <area>/<tile> in multi-area datasets.
    write_annotation(
        layout,
        "test_area2.gpkg",
        [
            box(
                AREA2_ORIGIN_X + 10.0,
                AREA2_ORIGIN_Y - 120.0,
                AREA2_ORIGIN_X + 20.0,
                AREA2_ORIGIN_Y - 110.0,
            )
        ],  # inside AREA2 tile (1,0) = "Q"
    )
    ds_ta = create_dataset(
        layout,
        [AREA, AREA2],
        list(FEATURES),
        "both.shp",
        None,
        img_size=IMG,
        dataset_id="multi_ta",
        test_area="test_area2.gpkg",
        seed=3,
    )
    assert ds_ta.meta["excluded_tile_ids"] == [f"{AREA2}/Q"]
    assert len(ds_ta.labels) == 4  # the excluded tile carried no labels

    # Duplicate area names are de-duplicated: identical to a single call.
    dup = create_dataset(
        layout,
        [AREA, AREA],
        list(FEATURES),
        "both.shp",
        None,
        img_size=IMG,
        dataset_id="dup",
        seed=3,
    )
    single = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        "both.shp",
        None,
        img_size=IMG,
        dataset_id="single",
        seed=3,
    )
    assert dup.samples.shape == single.samples.shape == (2, *IMG, 2)


def test_create_dataset_rebuilds_missing_product_artifacts(
    layout, dtm_product: Path, tpi_layer: Path, annotations
) -> None:
    """Missing tile indexes and VRT mosaics of the feature products are
    rebuilt from the tiles on disk before the dataset is created."""
    for product in (dtm_product, tpi_layer):
        for mosaic in ("dtm.vrt", "tpi.vrt", "mosaic.vrt"):
            artifact = product / mosaic
            if artifact.exists():
                artifact.unlink()
        tile_index_path(product).unlink()

    ds = create_dataset(
        layout,
        AREA,
        list(FEATURES),
        "true.shp",
        "false.shp",
        img_size=IMG,
        dataset_id="rebuilt",
        seed=3,
    )
    # Same sampling as the fully built product (see the first test): the
    # auto-rebuilt indexes carry filename-stem tile ids, and layer
    # matching is spatial, so the windows are unchanged.
    assert ds.samples.shape == (10, *IMG, 2)
    assert (dtm_product / "dtm.vrt").exists()
    assert tile_index_path(dtm_product).exists()
    assert (tpi_layer / "tpi.vrt").exists()
    assert tile_index_path(tpi_layer).exists()
