# LIDARK Modernization and Refactor Plan

Source analyzed: `old_code/LIDARK_script.py` (1,344 lines, single file) and `old_code/README.md`.
Goal: a modernized, modular Python package that performs the core tasks of the
original — LiDAR processing (DTM / TPI), training, prediction, and evaluation — on
current dependency versions. The CreateMap module is modernized per the revised
requirements: DTM generation uses PDAL instead of LAStools and runs tiles in parallel,
the MDHS visualization option is dropped, and GDAL VRT mosaics are used to handle the
tiled datasets efficiently. Two further requirements: tile identity must not rely on
mapsheet names being present in filenames (a spatial tile index is used instead), and
the default EPSG (3067) is stored in config. The ML layer is additionally structured
for two anticipated extensions: multi-class detection and alternative ML backends
(Faster R-CNN, YOLO, Mask R-CNN), plus one anticipated training extension:
fine-tuning an existing model (section 3.2). The **Mask R-CNN instance
segmentation backend is implemented** (section 3.1, `models/maskrcnn/`):
torchvision-based, trained with the **mask mAP** metric (AP50 default,
AP75 optional), emitting the Detection IR directly.

---

## 1. What the original program does

| Class | Lines | Responsibility |
|---|---|---|
| `CreateMap` | 29–363 | `DTM`: runs LAStools (`lasindex`, `las2dem`) via `os.system`. `TPI`: moving-window mean-difference filter (numpy, optional torch GPU), writes int16 GeoTIFF, trims the mapsheet buffer. `MDHS`: 6-direction hillshade from DTM gradients, min-combined, uint8. |
| `Training` | 367–862 | Builds 512x512 sample windows around true/false label geometries (2 random windows per object), flip augmentation (H + V), saves `.npy` under timestamps, trains a custom U-Net (Conv-BN-Act blocks, `Add` skip connections) with BCE loss + custom IoU metric, `ModelCheckpoint`/`EarlyStopping`, logs hyperparameters to CSV. |
| `Predict` | 864–1146 | Loads `.h5` model, tiles each mapsheet (512 px window, 128 px stride), stitches with max-probability blending, thresholds at 0.5, polygonizes (connectivity 4), simplifies (0.25 m), attaches mean probability (zonal stats) and area, merges per-area shapefiles and resolves cross-mapsheet overlap with `unary_union` + `overlay` + `dissolve`. |
| `Evaluate` | 1147–1340 | Clips predictions to an evaluation AOI, applies size (1 m2) and probability (0.8) cuts, computes detection counts (predicted / fully detected >= 0.9 overlap / partly detected / not detected / detected-with-no-label / false detected), writes a CSV report. |

Folder convention (preserved, minus MDHS): `Lidar/<area>/00_Original`, `01_DTM`,
`02_TPI/radius_<n>`, and `DeepLearning/{train_annotations, samples, labels,
model/<name>_<timestamp>, evaluate_annotations/<eval_name>, reports}`. The `03_MDHS`
product and folder are removed. Each tiled product additionally gets a GDAL VRT mosaic
written next to its tiles (`01_DTM/dtm.vrt`, `02_TPI/radius_<n>/tpi.vrt`) so downstream
steps can read the whole area as one dataset with windowed reads instead of iterating
over tiles.

## 2. Problems found in the original

### 2.1 Defects (behavior bugs)

| ID | Location | Problem |
|---|---|---|
| B1 | `CreateMap.__init__` | The `else: print('map-type not recognized')` binds only to the `MDHS` branch (`if DTM` / `if TPI` / `if...else`), so the "not recognized" message prints for **valid** map types too, and validation never happens. |
| B2 | `CreateMap.laz2dtm` | LAStools path hard-coded to `C:/lastools/bin/`; return codes of `os.system` are never checked, so a missing binary fails silently. |
| B3 | `Training.create_samples` | First `training_meta.append(...)` call is missing `ignore_index=True` (the false-label branch has it), producing inconsistent indexing. |
| B4 | `Predict.pred_mapsheet` | `zonal_stats` is called on `prob_arr` (the float probability array) — intentional — but `rasterstats.zonal_stats` receives a raw array + transform; fine, yet `pred_arr` name is shadowed by the prediction tile inside the stitching loop, making the code fragile. Also when no polygons are found, no shapefile is written and the merge step must silently skip it (handled by `os.path.exists` check, but implicit). |
| B5 | `Predict.merge_pred_shapes` | `gp.GeoDataFrame(merged_shapes.geometry.unary_union)` + `merged_geos[0]` column juggling is broken on every modern geopandas/shapely and was already brittle. |
| B6 | `Predict` / `Training` | `pred_meta` used for output raster metadata is whichever feature file was opened **last** in the loop — relies on all feature layers being pixel-aligned; unverified at runtime. |
| B7 | Global | Bare `except:` blocks swallow all errors (TPI export, mapsheet prediction); `warnings.filterwarnings("ignore")` globally hides real problems. |

### 2.2 Deprecated / removed APIs (breaks on modern versions)

| ID | API | Status |
|---|---|---|
| D1 | `pandas.DataFrame.append` / `GeoDataFrame.append` | **Removed** in pandas 2.0 — used in `create_samples`, `merge_pred_shapes`, `Evaluate.get_object_data`. |
| D2 | `gp.overlay` (module-level) | Removed in geopandas 1.0 (use `GeoDataFrame.overlay`). |
| D3 | `GeoSeries.unary_union` | Deprecated in shapely 2 / geopandas 1.0 (use `union_all()`). |
| D4 | Keras 2 idioms: `.h5` checkpoint, `tf.keras.backend` metric, `load_model(..., custom_objects=...)` | Keras 3 (default with TF >= 2.16) breaks the custom IoU metric and h5 loading path. |
| D5 | `np.append` in a loop for `pred_samples` | O(n²) reallocation — works, but pathological on real mapsheets. |
| D6 | tf 2.6 / pytorch 1.9 / geopandas 0.9 pin set | All EOL; no Python 3.10+ wheels for several of them. |

### 2.3 Design problems

- All work happens in `__init__` — classes are untestable and unconfigurable (no way to call steps in isolation, no dry-run).
- String path concatenation everywhere; no `pathlib`; paths built 20+ times from the same convention.
- Hard-coded `EPSG:3067`, mapsheet name length (9), LAStools location, hyperparameters.
- Full training dataset materialized in RAM (samples + 4x flip augmentation).
- TPI window loop is a per-offset Python loop over numpy views — same result is a `uniform_filter` / conv2d.
- No tests, no CLI, no seed control, no logging (only prints), no type hints.
- Timestamps used as the only identifier for datasets/models — no metadata attached.

## 3. Target architecture

```
lidark_uusi/
├── pyproject.toml            # packaging, deps, tool config (ruff, mypy, pytest)
├── README.md
├── MODERNIZATION_PLAN.md
├── src/
│   └── lidark/
│       ├── __init__.py
│       ├── config.py          # ProjectLayout (paths, DEFAULT_EPSG=3067) + ModelConfig (backend, classes)
│       ├── io/                # raster/vector helpers, tile index, tiling
│       │   ├── rasters.py     # read/write GeoTIFF, buffer-trim, profile handling
│       │   ├── vectors.py     # polygonize, simplify, one-pass polygon scoring, merge
│       │   ├── detections.py  # Detection IR: geometry + score + class_id schema (see 3.1)
│       │   ├── vrt.py         # build GDAL VRT mosaics over tiled DTM/TPI products
│       │   ├── tiles.py       # tile index (GeoPackage): footprints + opaque tile IDs
│       │   └── tiling.py      # WindowTiler: 512/128 tile generation + max-prob stitching
│       ├── maps/
│       │   ├── dtm.py         # PDAL-based DTM generation, parallel per-tile pipelines
│       │   ├── tpi.py         # vectorized TPI (scipy CPU / torch GPU), VRT windowed reads
│       │   └── api.py         # create_map(...) facade replacing CreateMap (DTM/TPI only)
│       ├── dataset.py         # class-aware training-sample extraction, augmentation, save/load
│       ├── models/            # ML backend abstraction (extensibility seam, see 3.1)
│       │   ├── base.py        # ModelBackend protocol: task, build, fit, predict_tiles, save/load
│       │   ├── registry.py    # backend name -> factory, selected via ModelConfig
│       │   ├── unet/          # default backend (Keras 3): architecture + metrics + trainer
│       │   └── maskrcnn/      # instance segmentation backend (torchvision): mask mAP, sparse IR
│       ├── train.py           # train(...) facade: resolves backend from the registry
│       ├── predict.py         # predict(...) facade: backend-agnostic, consumes Detection IR
│       ├── evaluate.py       # evaluate(...) facade: per-class stats (binary = 1 class)
│       └── cli.py             # argparse subcommands: create-map, train, predict, evaluate
├── tests/
│   ├── conftest.py            # synthetic project tree fixtures (tiny rasters, shapes)
│   ├── test_tpi.py / test_dtm.py / test_vrt.py
│   ├── test_dataset.py / test_models.py (protocol, registry, U-Net) / test_train.py
│   ├── test_predict.py / test_evaluate.py
│   └── test_parity.py         # golden-output checks vs. reference arrays
└── old_code/                  # kept as reference, unchanged
```

Public API parity — callable functions instead of constructor side effects.
`map_type` is now limited to `{"DTM", "TPI"}` (MDHS dropped):

```python
from lidark import create_map, train, predict, evaluate

create_map(project_folder, project_area="Naljanka", map_type="DTM", buffer=50, pixel_size=0.25, cores=8)
train(project_folder, model_name="tar_kilns_v1", true_objects="Tar_kilns.shp", false_objects="OtherFeatures.shp",
      feature_list=["02_TPI/radius_20/", "02_TPI/radius_30/"], backend="unet", ...)
predict(project_folder, project_areas=["Naljanka"], model_id="tar_kilns_v1_20260101-120000",
        pred_type="all")
evaluate(project_folder, true_objects="evaluate_true.shp", model_id=..., eval_name="test1")
```

`predict()` no longer requires `feature_list`: it is read from the trained model's
`model_config.json` (explicit argument still overrides — useful for legacy models).

### 3.1 Extensibility seams (future: multi-class, alternative backends)

- **Backend protocol** (`models/base.py`): every ML backend implements
  `ModelBackend` — `task` (semantic segmentation / instance detection / instance
  segmentation), `build(config)`, `fit(train, val)`, `predict_tiles(tiles)`,
  `save(dir)`, `load(dir)`. Backends register in `models/registry.py` and are
  selected by `ModelConfig.backend` (default `"unet"`). Nothing outside `models/`
  imports Keras/PyTorch — future YOLO / Faster R-CNN backends are new
  subpackages plus one registry entry, with no changes to `train.py`,
  `predict.py`, `evaluate.py`, or the CLI. **The Mask R-CNN backend is
  implemented** (`models/maskrcnn/`, torchvision via `lidark[torch]`):
  `maskrcnn_resnet50_fpn` with an F-channel stem and the transform pinned to
  the window size; instance targets derive from the class maps
  (`instances_from_label`: 4-connected components; touching same-class
  objects merge — the documented class-map limitation); training validates
  with **mask mAP** (`models/maskrcnn/metrics.py`: greedy score-ordered
  matching, all-point AP — **AP50** default, **AP75** via
  `ModelConfig.mask_ap_iou` / `--mask-ap-iou`), which is the monitored
  checkpoint/early-stopping metric (`val_map` in the history/log); weights
  persist as `model.pt` with lazy torch imports, so the core install never
  needs the framework.
- **Detection IR** (`io/detections.py`): the single interchange type consumed by
  vector merge and evaluation — a GeoDataFrame with `geometry`, `score`,
  `class_id`, `class_name`, `area`. Dense backends (U-Net) produce a per-tile score
  raster that is stitched, thresholded, and polygonized into the IR; box/mask
  detectors convert their outputs into the same IR directly. All downstream logic
  (merge, cuts, overlap statistics) is therefore reusable unchanged for any backend.
- **Class-aware labels**: label shapefiles may carry an integer `class_id` column
  (values >= 1). A missing column means single class `class_id = 1`, reproducing the
  original binary behavior exactly. Dataset labels are stored as per-pixel uint8
  class maps; the U-Net output becomes a C+1-channel softmax when C > 1 (binary
  sigmoid when C = 1 — parity), and the metric is mean IoU over classes (binary IoU
  when C = 1 — parity). `dataset_meta.json` / `model_config.json` record
  `num_classes` and `class_names`.
- **Model metadata**: each model dir stores `model_config.json` (backend, task,
  classes, feature_list, img_size, seed, hyperparameters). `predict()` and
  `evaluate()` read it, so switching backends or classes does not change call sites.
- **Optional extras**: alternative backends arrive as optional dependencies
  (`lidark[torch]` — torch + torchvision, covering the implemented Mask
  R-CNN backend and GPU TPI; later `lidark[yolo]` = ultralytics,
  `lidark[detectron2]`); the core install stays TensorFlow-only, and
  unselected backends are never imported.

### 3.2 Potential feature: fine-tuning / warm-start from an existing model

**Implemented** — `train(..., base_model_id=...)` / CLI
`lidark train --base-model` warm-start the weights from an existing model
directory (`train.py` resolution + guards, backends' `load()` keeps a
built config and takes only the weights), covered by
`tests/test_finetune.py` including the plan's acceptance tests below. A
fine-tune-specific learning-rate default was deliberately **not** added:
a lower `--learning-rate` is explicit rather than silently changed.

- **Use case**: training on a new area or a refreshed label set is faster and
  more data-efficient from an existing model than from random initialization,
  and a lineage of models (`tar_kilns_v1` -> `tar_kilns_v2`) mirrors how
  production models actually evolve.
- **API**: `train(..., base_model_id="tar_kilns_v1_20260101-120000")`; CLI
  `lidark train --name tar_kilns_v2 --base-model tar_kilns_v1_...`. The base
  model is the **starting point only**: the fine-tuned run always gets a fresh
  model id and directory — the base model is never modified.
- **Mechanism (design sketch)**: resolve the base dir through
  `ProjectLayout.model_dir`, read its `model_config.json`, and warm-start the
  backend — the `ModelBackend` protocol already covers this with `load(dir)`
  followed by `fit(...)`, so the feature is a `train.py`-level change with no
  protocol growth; every backend (current and future) gains fine-tuning by
  implementing the existing two methods.
- **Compatibility guards** (checked, not assumed): the base model's backend,
  input shape (`img_size` x `len(feature_list)`), and `num_classes` must match
  the new run's config — a mismatch raises a clear error. For the U-Net,
  `unet_size` (its architecture parameter) is guarded too; differing
  feature-layer **names** with the same count load cleanly but are warned
  about. Changing the class count (re-initialized output head) is out of the
  initial scope and a possible
  follow-up.
- **Provenance**: the new model's `model_config.json` and training log record
  `base_model_id` — the id chain carries the full lineage — and the per-epoch
  history makes the warm start visible (epoch-1 loss near the base model's
  last, not a scratch run's high initial loss).
- **Hyperparameters**: fine-tuning typically wants a lower learning rate than
  scratch training; `learning_rate` is already configurable
  (`--learning-rate 0.0001`), and a fine-tune-specific default can arrive with
  the feature.
- **Tests** (with the feature): a warm-started run produces a new model dir
  whose epoch-1 loss is below a scratch run's on the same dataset; the base
  model's files are unchanged afterwards; the compatibility guards raise the
  documented errors.

### 3.3 Feature: multi-class detection support

**Implemented** — MC1-MC7 below are in the code (`predict.py`,
`dataset.py`, `train.py`, `evaluate.py`, `cli.py`, `models/unet/backend.py`),
covered by `tests/test_multiclass.py`; every addition defaults to its
single-class equivalent, so binary behavior is unchanged.

Analysis of the seams sketched in 3.1 against the current code shows the
**data path is already class-aware end to end**: the Detection IR carries
`class_id`/`class_name` (`io/detections.py`), label shapefiles accept an
integer `class_id` column and the dataset stores per-pixel uint8 class maps
with `num_classes`/`class_names` in `dataset_meta.json` (`dataset.py`), the
U-Net emits a C+1-channel softmax head with `CrossEntropyLoss` and
mean-IoU-over-classes (`models/unet/`), predict converts dense channels into
per-class IR rows and sparse backends emit instances with class ids
(`predict.py`), the area merge dissolves within each `class_id` only
(`io/vectors.py`), evaluation reports the six counters per class plus an
overall row (`evaluate.py`), and both detector backends (Faster R-CNN,
Mask R-CNN) are natively multi-class. What remains for **full** multi-class
detection is operational quality, not plumbing — the gaps below.

| ID | Gap | Location | Problem |
|---|---|---|---|
| MC1 | Threshold vs. argmax disagreement | `predict.py` (dense path) | Per-channel `prob > threshold` on softmax probabilities can pass the same pixel in two channels: the per-tile raster mask takes the argmax, but the vector IR polygonizes each channel's mask independently, so IR and raster disagree and cross-class overlaps survive the merge (which never merges across classes). |
| MC2 | Phantom classes | `dataset.py` | `num_classes = max(class_id)`: labels with sparse ids (e.g. {1, 5}) create always-empty classes 2-4 — wasted output channels and a mean IoU dragged by classes that never occur. |
| MC3 | Class imbalance unhandled | `dataset.py` sampling, `models/unet/backend.py` | 2 windows per object regardless of class — a 10-object rare class gets 10x fewer windows than a 100-object common class — and the loss is unweighted BCE/CCE against a background-dominated class map. |
| MC4 | `class_names` not plumbed through | `dataset.py`, `evaluate.py`, `cli.py` | The dataset hard-codes `class_<id>` names, the evaluation report hard-codes `class_{id}` instead of reading `model_config.json`, and there is no CLI flag to supply human-readable names. |
| MC5 | One operating point for all classes | `predict.py`, `evaluate.py` | A single detection `threshold` (0.5) and global `size_cut`/`prob_cut` — classes with different area/score distributions cannot be tuned independently. |
| MC6 | Cross-class confusion invisible | `evaluate.py` | Pairing is per class: a class-2 prediction sitting on a class-1 label surfaces only as "Not detected" + "False detected"; which class was confused with which is not reported. |
| MC7 | Unstratified validation split | `dataset.py` / `train.py` | The val split is random over windows; a rare class can end up absent from validation, making per-class val IoU (the checkpoint metric) meaningless for that class. |

Planned work, in dependency order (binary behavior is untouched: every
addition defaults to its single-class equivalent):

1. **Argmax-constrained dense conversion** (MC1): in the multi-class dense
   path, a pixel belongs to class `argmax(prob)` only when that channel's
   probability `>= threshold`; per-channel masks are intersected with the
   argmax map before polygonization, and the IR `score` is the argmax
   channel's probability. IR and raster then agree and no cross-class
   overlap reaches the merge. Sparse backends are unaffected.
2. **Compact class mapping** (MC2): the dataset remaps the observed ids to
   `1..C` and records `class_id_map` (original id -> compact id) in
   `dataset_meta.json` / `model_config.json`; `class_names` follows the map.
   A gap between observed ids logs a warning (ids are user data, not an
   error).
3. **Imbalance handling** (MC3): optional `class_weights` in
   `ModelConfig`, passed as CCE `weight` (softmax) / BCE `pos_weight` per
   channel (sigmoid); optional class-balanced sampling — windows per object
   normalized per class so every class contributes equally, off by default
   (original 2-per-object parity).
4. **Class names** (MC4): `train(..., class_names=...)` / CLI
   `--class-names`; stored in config, used by `predict()` IR rows, and read
   by `evaluate()` from `model_config.json` instead of the hard-coded
   `class_{id}`.
5. **Per-class operating points** (MC5): `threshold`, `size_cut`, and
   `prob_cut` accept a scalar (all classes — parity) or a per-class mapping;
   per-class cuts are recorded in the report so a run's operating points are
   reproducible.
6. **Confusion reporting** (MC6): `evaluate()` adds a confusion count per
   (true class, best-overlap predicted class) for detected labels, and
   breaks "Detected with no label" out by predicted class; the six-counter
   per-class rows keep their current semantics.
7. **Class-stratified validation** (MC7): the split guarantees every class
   present in the dataset appears in both halves (greedy assignment when a
   class has too few windows).
8. **Tests** (with the feature): a 3-class synthetic fixture (imbalanced
   counts, sparse original ids) asserting: IR/raster argmax agreement (MC1),
   `num_classes` equals the number of observed ids and the map round-trips
   (MC2), weighted vs. unweighted loss differ on the imbalanced fixture and
   balanced sampling equalizes per-class window counts (MC3),
   `class_names` survive train -> predict -> evaluate (MC4), per-class
   thresholds/cuts are honored independently (MC5), and the confusion count
   attributes a misclassified detection to the right class pair (MC6);
   binary outputs stay byte-identical to the current pipeline on the
   existing single-class fixtures.

### Phase 0 — Scaffolding (repo)

- `pyproject.toml` with pinned modern dependency set:
  `python>=3.10`, `numpy>=1.26`, `pandas>=2.2`, `geopandas>=1.0`, `shapely>=2.0`,
  `rasterio>=1.3`, `scipy>=1.12`, `tensorflow>=2.16` (Keras 3),
  `torch>=2.2` (optional, GPU TPI), `pdal>=2.6` + `laspy` (DTM generation),
  `gdal-utils` (gdalbuildvrt for VRT mosaics), `pyogrio` (default geopandas IO
  engine), `tqdm`. `rasterstats` is dropped — replaced by the one-pass polygon
  scorer in `io/vectors.py` (see section 7, P9).
- Tooling: `ruff` (lint + format), `mypy` (strict on `src/`), `pytest`, `pre-commit`.
- Dependency extras so future backends stay optional: the core install needs only
  TensorFlow; `lidark[torch]` adds torch (GPU TPI + any torch-based backends);
  future `lidark[yolo]` (ultralytics) and `lidark[detectron2]` arrive with their
  backends — never in the core dependency set.
- Replace `print` with the `logging` module (`-v` flag for DEBUG).

### Phase 1 — Foundation: config and IO

- `config.ProjectLayout` dataclass as the **single** place that knows the folder
  convention (`lidar_area(area)`, `dtm_dir(area)`, `tpi_dir(area, radius)`,
  `model_dir(timestamp)`, ...). All `'/Lidar/... + '` string building is
  deleted; callers receive `pathlib.Path` objects.
- The default CRS lives in config and only in config:
  `config.DEFAULT_EPSG = "EPSG:3067"`, used as the default of the
  `ProjectLayout.epsg` field (overridable per project/area). No other module hardcodes
  an EPSG; CRS is read from raster/vector metadata when present, with the config value
  as fallback and as the CRS written to new outputs.
- `io/rasters.py`: `read_band()`, `write_band()` (typed profiles, explicit `nodata`),
  `trim_buffer()` (the buffer-slicing + transform-shift logic currently inline in TPI).
- `io/vectors.py`: `polygonize_mask()` (rasterio.features.shapes + simplify),
  `polygon_scores()` (single `rasterize` into a region-id image + one
  `scipy.ndimage.mean` reduction per region — a single pass over the tile,
  replacing the original per-polygon `rasterstats` loops, see 7/P9), with an
  `overlap` strategy parameter for overlapping polygons (`independent` /
  `exclusive` / `priority`, see 7.1),
  `merge_overlapping()` (union_all -> overlay -> dissolve, modern
  geopandas). This isolates the two most fragile parts of the old code (B4, B5).
- `io/tiling.py`: replaces the inline double loop in `pred_mapsheet`; generator-based
  tile production and a `max_prob_stitch` that keeps the exact original semantics
  (max of overlapping tile probabilities per pixel), implemented on preallocated
  buffers with `np.maximum` on window views — no per-window Python arithmetic.
- `io/vrt.py`: `build_vrt(tile_paths, out_path)` shells out to `gdalbuildvrt` via
  `subprocess.run(check=True)` (deliberately **not** `osgeo.gdal`, to avoid a GDAL
  ABI clash with the GDAL bundled inside rasterio's wheels); `open_vrt()` returns an
  open rasterio dataset handle so callers can do single windowed reads across the
  whole mosaic instead of iterating tiles.
- `io/tiles.py` — tile index, replacing the fixed 9-character mapsheet-name
  assumption: `build_tile_index(product_dir)` scans the product's rasters, records
  each tile's footprint, path, and a `tile_id` in `<product_dir>/tiles.gpkg`;
  `load_tile_index()` returns it as a GeoDataFrame. `tile_id` defaults to the
  filename stem but is an **opaque label of arbitrary length** — mapsheet information
  is never parsed from filenames. Tiles from different layers are matched
  **spatially** (footprint intersection), never by name prefix.

### Phase 2 — Maps

- **DTM** (`maps/dtm.py`) — PDAL replaces LAStools:
  - The output tile grid follows the input LAZ tiling: one DTM tile per LAZ file,
    bounds taken from the LAS header and snapped to the pixel grid. Tile identity is
    the LAZ filename stem — an opaque, arbitrary-length `tile_id` registered (with
    footprint and path) in `01_DTM/tiles.gpkg`; no mapsheet prefix is assumed or
    parsed.
  - Per-tile pipeline (Phase 2 remark, implemented): one PDAL pipeline per output
    tile reads the input LAZ tiles whose header bounds intersect the tile extent +
    `buffer` (tile-index neighbour reads — no intermediate point cloud is written
    to disk):
    1. one `readers.las` per intersecting file + `filters.merge` (modern PDAL
       streams do not concatenate reader stages),
    2. `filters.range`, `Classification == 2` (same as `-keep_class 2`),
    3. `filters.crop` to the tile bounds + `buffer` meters (triangulation margin
       from neighbouring tiles, clean seams),
    4. `filters.delaunay` + `filters.faceraster` — a TIN rasterization like the
       original `las2dem` — with the raster grid pinned to the **original tile
       bounds** (`origin_x`/`origin_y`/`width`/`height`; `origin_y` is the bottom
       edge), so the output tile is clipped to the tile itself and adjacent tiles
       abut exactly, without overlap,
    5. `writers.raster` (`gdaldriver=GTiff`, float32, `nodata=-9999`), filename
       `<tile_id>_DTM.tif`.
    The default `buffer` is 2.0 m. A tile with no ground points yields an
    all-nodata tile instead of failing the product.
  - **Remark (revision)**: creating `ground.copc.laz` is slow and takes up a lot
    of disk space. The better overall option is to **avoid storing intermediate
    point clouds to disk altogether**: build a tile index of the input LAZ tiles
    (footprints + paths, `io/tiles.py`) and, per output tile, read the necessary
    points directly from the neighboring LAZ tiles whose footprints intersect
    the tile extent + buffer, keeping the ground filter
    (`Classification == 2`) inside the same per-tile PDAL pipeline. The tile
    index keeps these reads targeted, so only the few tiles actually bordering
    an output tile are opened. This supersedes the two-pass design above (and
    its COPC/EPT ground intermediate) as the default; no intermediate point
    cloud is written to disk.
  - `filters.faceraster` accepts `max_triangle_edge_length` — the direct
    equivalent of the LAStools `-kill 50` TIN edge limit; it is left at PDAL's
    default (no limit).
  - **Parallel processing**: the per-tile pipelines are fully independent, so they are
    executed through `concurrent.futures.ProcessPoolExecutor(max_workers=cores)` —
    each worker process builds and executes its own PDAL pipeline (a single PDAL
    pipeline runs serially; the parallelism is across tiles). Progress is reported
    with `tqdm` over `as_completed`. `cores` retains its original meaning.
  - After all tiles finish, build `01_DTM/dtm.vrt` with `io/vrt.py` and write
    `01_DTM/tiles.gpkg` with `io/tiles.py`, so TPI and inspection tools can treat
    the area as one raster and downstream steps can identify tiles without filename
    conventions.
- **TPI** (`maps/tpi.py`):
  - Replace the per-offset view loop with a vectorized computation of the
    **circular filter** mean: full weight (1.0) up to `radius + 0.1` px, a
    half-weight (0.5) edge ring up to `radius + 0.5` px, zero beyond, center
    excluded. The kernel evaluates the filter as the exact decomposition
    `w = 0.5*[d < r+0.1] + 0.5*[d < r+0.5]` — two hard-disk sums via per-row
    sliding sums over a shared prefix-sum array; the GPU path uses a single
    `torch.nn.functional.conv2d` with the explicit weight kernel. The golden
    parity test proves the kernel equals the per-offset reference
    (`out = z - weighted sum(neighbours)/weighted count`) bit-exactly in
    float64.
  - Keep the post-processing pipeline: `*100`, clip [-100, 100], int16, buffer
    trim, transform shift by `+buffer / -buffer` — with one revision: kernel
    nodata (NaN) maps to the int16 nodata constant `-32768`, written as the
    GeoTIFF nodata tag, instead of the original's `nan->0` (so gaps stay
    distinguishable from flat terrain).
  - **Nodata handling (revision)**: nodata cells (`<= 0` or non-finite) are
    **excluded** from the neighbourhood mean — they do not prevent computation
    (the original propagated NaN from any nodata cell in the window). The
    output is nodata only where the center itself is nodata or the window
    contains no valid neighbour at all.
  - Accumulation runs in float32 by default (bandwidth-bound kernels, half the
    memory traffic of the original float64 temporaries); the kernel accepts a dtype
    argument and the golden parity test runs it in float64 so the bit-parity claim
    against the original stays intact (see 7, P4).
  - Execution mirrors the DTM parallelism (see 7, P3): the CPU path runs tiles
    through the same `ProcessPoolExecutor`; the GPU path **batches** tile windows
    into stacked tensors and applies a single `conv2d` per batch (chunked to
    available VRAM) instead of one kernel-launch set per tile.
  - Source data comes from `01_DTM/dtm.vrt`: for each DTM tile listed in the tile
    index, one windowed read covering the tile footprint + `radius` overlap replaces
    opening the buffered DTM tiles individually — this guarantees the overlap is
    always present regardless of the buffer width used at DTM time, and reads only
    the needed region of the mosaic. The read is **boundless with the DTM nodata
    fill**: cells with no neighbouring tile at the area edge become nodata and are
    excluded by the kernel. The margin is trimmed in exact pixels, so output tiles
    cover the tile footprints exactly. Output tiles (same tile IDs) and the
    transform math are unchanged. After all tiles, `io/vrt.py` builds
    `02_TPI/radius_<n>/tpi.vrt` and `io/tiles.py` writes the TPI tile index with the
    same tile IDs and footprints.
- **MDHS dropped**: the multidirectional-hillshade product is removed per the
  revised requirements. There is no `maps/mdhs.py`, no `03_MDHS` folder, and no
  `map_type='MDHS'` option; `create_map` accepts only `{"DTM", "TPI"}`.
- `maps/api.py`: `create_map(...)` validates `map_type` against the literal set
  `{"DTM", "TPI"}` and raises `ValueError` otherwise (fixes B1); no `__init__` side
  effects.

### Phase 3 — Dataset and training

- `dataset.py`:
  - Tile discovery uses `io/tiles.py`, not filename prefixes: label geometries are
    matched to feature-layer tiles by spatial intersection (`geopandas.sjoin`
    against the tile index), and the corresponding tiles of every layer in
    `feature_list` are resolved spatially (footprint match), never by name.
    **Test areas (revision)**: the held-out region is an **optional polygon
    layer** (e.g. a `.gpkg` file, like the label shapefiles under
    `train_annotations/`; `test_area` argument / CLI `--test-area`) — tiles
    whose footprint intersects the polygons are excluded from sampling and
    recorded as `excluded_tile_ids` in the dataset metadata. This replaces
    the original `test_mapsheets` tile-ID list; no assumption that mapsheet
    names are embedded in filenames or that they have a fixed length.
  - Rebuild `create_samples` with lists + `pd.concat` (fixes D1 and B3). The sampling
    logic (2 random windows per object, window placement bounds, the 4-corner
    clip-fallback for windows crossing raster edges) is preserved verbatim but factored
    into a single `extract_window()` helper used by both the label and feature loops.
  - Class-aware labels: label shapefiles may carry an integer `class_id` column
    (>= 1); missing column = single class 1, which reproduces the original binary
    labels exactly. Label windows are stored as per-pixel uint8 class maps
    (the original binary mask is the C=1 special case).
  - Determinism: all randomness through a `random.Random(seed)` / `np.random.default_rng(seed)`
    passed from config; seed recorded in the dataset metadata.
  - Save format: disk-backed **memmap** (`samples.dat` + `labels.dat`, shape and
    dtype recorded in `dataset_meta.json` together with feature_list, img_size, epsg,
    num_classes, class_names, augmented, seed, excluded tile IDs, source label
    files). `load_dataset()` memory-maps and returns the handles + metadata; the
    training pipeline streams per-index windows straight from the memmap, so RAM is
    bounded by the batch, not the dataset, and there is no graph copy (see 7, P5).
    A one-way `.npy` importer covers legacy datasets.
  - Memory: augmentation done on-the-fly during training — `tf.data` with a parallel
    `map` doing the flips and `prefetch(AUTOTUNE)` — rather than materializing 4x
    copies; the augmented flag is metadata, not duplication.
- `models/` (replaces the old single `model.py` — the extensibility seam):
  - `models/base.py`: the `ModelBackend` protocol (`task`, `build`, `fit`,
    `predict_tiles`, `save`, `load` — see 3.1); typed, minimal, and backend-neutral.
    `models/registry.py`: name -> factory map with a clear error for unknown names.
    Future backends (YOLO, Faster R-CNN, Mask R-CNN) implement the protocol and
    register themselves; `train.py`/`predict.py` never reference a concrete backend.
  - `models/unet/` — the default backend, ported with full parity:
    - U-Net topology unchanged: same depth (4 encoders, bottleneck, 4 decoders), same
      `Conv2D(use_bias=False) -> BN -> Activation` blocks, same `Add`-skip decoders,
      same base features (16). Output layer is task-dependent: 1-channel sigmoid
      when `num_classes == 1` (original behavior), C+1-channel softmax otherwise.
    - IoU implemented as a proper Keras metric class
      (`@tf.keras.utils.register_keras_serializable()`), so `load_model()` needs no
      `custom_objects` dict (fixes D4). Binary threshold stays 0.5; multi-class uses
      mean IoU over classes.
    - Model format: `.keras` (native Keras 3) instead of `.h5`; a small loader keeps
      read-compatibility for legacy `.h5` files with the custom IoU dependency.
    - Every backend writes `model_config.json` into its model dir (backend, task,
      num_classes, class_names, feature_list, img_size, seed) — read by `predict()`
      and `evaluate()` instead of re-prompting the user.
    - Throughput options in `ModelConfig` (off by default, recommended on GPU —
      see 7, P6): mixed precision (`mixed_float16`) and XLA `jit_compile` for
      training and inference; the training log records img/s per epoch so
      throughput changes are visible.
- `train.py`: `train(...)` resolves the backend from `config.ModelConfig.backend`
  (default `"unet"`) via the registry, builds with `ModelConfig`, compiles with
  Adam(lr=0.001) + BCE + IoU, `ModelCheckpoint(save_best_only, monitor='val_iou',
  mode='max')`, `EarlyStopping(patience=10)`, `max_epochs=200`, `batch_size=32`,
  `val_split=0.2` — all configurable, same defaults. **Model naming (revision)**:
  the user supplies a required model name (`model_name` / CLI `--name`); the
  model id is the sanitized name plus the timestamp
  (`<name>_<YYYYMMDD-HHMMSS>`), naming the model dir, the log file, and the
  prediction/report paths — memorable name, unique timestamp. The training log
  is written as **pretty-printed JSON** into the model directory itself
  (`<model_dir>/training_log.json`, so the log lives next to the model:
  the original `create_log` key/value rows plus backend/classes **and the
  full per-epoch history** — `loss`, `iou`, `val_iou`, throughput, one series
  per metric), plus the config JSON copied into the model dir.

### Phase 4 — Prediction

- `predict.py: predict(...)`: facade with `pred_type in {"all", "test"}`, same
  per-area loop. Tiles are iterated from the feature product's tile index
  (`tiles.gpkg`) — not discovered by filename prefix — and per-tile failures are
  **logged** instead of printed. For `pred_type="test"`, the held-out tiles
  are selected by the **test-area polygon layer** (`--test-area`, the same
  convention as training) when given, otherwise by tile-ID patterns
  (`test_tiles`, the original `test_mapsheets` substring semantics); at
  least one selector is required. The backend is loaded through the registry from
  `model_config.json` (same interface for U-Net and any future backend); the raster
  and vector outputs below describe the default (dense segmentation) path.
- Tile-and-stitch via `io/tiling.py`; tile assembly into a preallocated array instead
  of `np.append` (fixes D5). Same window (512), stride (128), max-prob stitching,
  0.5 threshold, connectivity-4 polygonization, 0.25 simplify, zonal mean probability,
  area attribute. Dense per-tile results are converted to the Detection IR
  (`geometry`, `score`, `class_id`, `class_name`, `area`); multi-class dense models
  emit one IR row per class region. **Instance-segmentation backends
  (Mask R-CNN, implemented) take the sparse path**: no score rasters and no
  stitching — each window's instances are converted straight into the IR
  (masks polygonized in window coordinates via `predict_instances`), the
  per-tile raster output becomes the class-id footprint of the predicted
  instances (preserving idempotent reruns), and the sliding-window
  duplicates of one object dissolve in the area-level IR merge — merge and
  evaluation are shared with the dense path.
- Inference throughput (see 7, P7–P9): windows are processed in **batches**
  (`pred_batch_size`, default 32) instead of one forward pass per window; window
  `overlap` is configurable in `ModelConfig` (default 128 preserved for detection
  quality near window edges — note it costs ~16x a disjoint tiling, and 256 halves
  it); polygon mean probability uses the one-pass `polygon_scores()` instead of
  per-polygon `rasterstats`.
- Verify pixel alignment across `feature_list` layers explicitly: assert equal
  transform among the feature rasters of a tile, crop to min-extent as the original
  did (fixes B6).
- Feature layers are read through their VRT mosaics where present
  (`<feature>/tpi.vrt` etc.): one windowed read per tile per feature replaces the
  `glob` + `open` pattern, falling back to individual tiles when no VRT exists.
  Per-tile raster outputs are unchanged (named by tile ID); vector outputs
  are GeoPackages (`.gpkg`).
- `merge_overlapping()` in `io/vectors.py` reimplements the union/overlay/dissolve
  merge using `union_all()` + `GeoDataFrame.overlay` + `dissolve` (fixes B5, D2, D3),
  writing `merged_shapes_<pred_type>.gpkg` in the same location, now operating on the
  Detection IR (detections dissolve together only within the same `class_id`).
  Per-tile Detection IR accumulates **in memory** across the area loop
  (`pd.concat`); per-tile GeoPackages are written only when `write_tile_shapes=True`
  — the original write-then-read-then-merge disk round trip is gone (see 7, P10).
- Skip-if-exists behavior for raster output kept (idempotent reruns), but vector
  output existence made explicit instead of implicit.

### Phase 5 — Evaluation

- `evaluate.py: evaluate(...)` — same semantics: read eval annotations under
  `DeepLearning/evaluate_annotations/<eval_name>/`, optional false objects, AOI from
  file or label bounding box, clip predictions, `size_cut=1`, `prob_cut=0.8`,
  `buffer(0)` geometry repair, overlap statistics:
  `Predicted / Fully detected (>=0.9) / Partly detected / Not detected /
  Detected with no label / False detected`. Labels already carry `label`
  1 = true / 2 = false in the original; this generalizes cleanly to arbitrary
  `class_id` values: the six counters are computed per class and summed into an
  overall row, so the original report is exactly the single-class case.
  Evaluation consumes the Detection IR, so it works for any backend without changes.
- Candidate pairing is done with `sjoin` (STRtree) before any overlap-area
  computation; the pairwise geometry intersection runs only on candidate pairs,
  not all label x prediction combinations (see 7, P11).
- `GeoDataFrame.append` -> `pd.concat`; overlay calls -> instance method; report CSV
  written to `DeepLearning/reports/<model_id>/<eval_name>.csv` as before.

### Phase 6 — CLI and packaging

```
lidark create-map --project-folder C:/project/ --area Naljanka --map-type TPI --buffer 20 --gpu
lidark train     --project-folder ... --name tar_kilns_v1 --true-objects Tar_kilns.shp --feature-list 02_TPI/radius_20/ 02_TPI/radius_30/ --backend unet ...
lidark predict   --project-folder ... --areas Naljanka Kuivaniemi --model-id tar_kilns_v1_20260101-120000 --pred-type all
lidark evaluate  --project-folder ... --model-id tar_kilns_v1_20260101-120000 --eval-name test1
```

- `argparse` (stdlib, no new dep). Every facade function usable directly from Python too.
  All commands accept `--epsg` (default `EPSG:3067`, from `config.DEFAULT_EPSG`).
- `README.md` rewrite: install, folder convention, the four commands, dependency notes
  (PDAL required for DTM — install via conda-forge; torch optional for GPU TPI;
  gdal-utils for VRT mosaics).

### Phase 7 — Testing, parity, CI

- Unit tests on synthetic data (conftest builds a fake project tree in `tmp_path`:
  64x64 float DTM rasters, a few label polygons):
  - TPI: compare against a brute-force reference implementation for small windows;
    verify buffer trim and transform math.
  - DTM (PDAL): synthetic LAS written with laspy -> pipeline -> DTM tile with the
    expected extent, resolution, nodata, and ground-only content; seam test: adjacent
    tiles cover each other's buffer; parallel run (2+ tiles, `ProcessPoolExecutor`)
    produces identical tiles to a serial run.
  - VRT: `dtm.vrt`/`tpi.vrt` open in rasterio; a windowed read across a tile seam
    returns the expected pixels; mosaic extent equals the tile-union extent.
  - Tile index: built from rasters with arbitrary (non-mapsheet) filenames of
    varying lengths; spatial matching pairs the same tile across two feature layers;
    tile-ID exclusion (`test_mapsheets`) works; nothing in the pipeline depends on
    filename content or length.
  - Dataset: window placement/edge clipping, augmentation equivalence, save/load round-trip;
    `class_id` column honored, absent column defaults to class 1.
  - Models: registry resolves `"unet"` by default and raises a clear error for
    unknown backends; `ModelBackend` protocol conformance check for the U-Net;
    output shape, IoU metric sanity (perfect prediction -> 1.0), single-class
    (binary parity) and multi-class (softmax, mean IoU) output shapes; training
    runs 1 epoch on a tiny dataset without error; `model_config.json` round-trips.
    Mask R-CNN: mask mAP metric cases (perfect, missed, AP50/AP75 threshold
    split, greedy matching — pure numpy, always run) plus backend
    build/fit/checkpoint/save-load/`predict_instances` tests (skipped without
    `lidark[torch]`); the sparse predict path is exercised with a stub instance
    backend (no torch required).
  - Detection IR: schema validation, dense-to-IR conversion produces expected
    geoms/scores on a synthetic mask.
  - Performance guards: batched inference returns predictions identical (within
    float tolerance) to single-window inference; the memmap dataset streams windows
    without loading the full array into RAM; `polygon_scores()` matches a
    `rasterstats` reference on a synthetic case; the tile-index DTM
    (per-output-tile neighbor reads, Phase 2 remark) produces the same tiles as
    a naive read-everything pipeline on a small fixture.
  - Predict: tiler stitching equals reference max-blend; polygonize produces expected
    geoms on a synthetic mask; merge resolves overlap.
  - Evaluate: counts on a constructed label/prediction pair.
- End-to-end smoke test: tiny synthetic DTM -> TPI -> dataset -> 1-epoch train ->
  predict -> evaluate, asserting the folder/file layout.
- Golden parity tests where feasible: run original vs new TPI on the same
  synthetic input and require identical arrays (both are deterministic). DTM is
  excluded: PDAL's IDW gridding is intentionally different from LAStools' TIN
  rasterization (see Risks).
- CI (GitHub Actions): lint + mypy + pytest on one windows and one ubuntu runner;
  tensorflow CPU.
- **Real-data samples (opt-in only)**: `test_data/laz/` holds a small sample of
  real LiDAR data (if present) that can be used for more realistic testing when
  synthetic data is inadequate — e.g. verifying DTM interpolation quality on
  genuine point density and classification noise, or profiling the DTM pipeline's
  tile-index neighbor reads on real LAZ decompression costs. **Synthetic data remains the default
  for all tests**: it is deterministic, fast, and proves the same logic. Tests
  that need real data are marked `@pytest.mark.realdata` and use the
  `real_laz_files` fixture, which skips automatically when `test_data/laz/` is
  empty or missing, so CI never depends on the sample being present.

## 5. Implementation order and acceptance criteria

| Step | Deliverable | Acceptance criteria |
|---|---|---|
| 1 | Scaffolding + `config.py` + `io/` (incl. tile index) | `ruff`/`mypy` clean; layout dataclass resolves all convention paths; default EPSG 3067 comes from config only; tile index builds/loads from arbitrary filenames. |
| 2 | `maps/tpi.py` + `io/vrt.py` | TPI bit-identical to the per-offset reference on synthetic input (revised nodata semantics, float64 mode); nodata cells excluded from the mean and mapped to the int16 nodata constant; missing neighbouring tiles become nodata; VRT mosaics build, open, and read correctly across seams; tiles run in parallel (CPU) / batches (GPU). |
| 3 | `maps/dtm.py` (PDAL, tile-index neighbor reads, parallel) | Synthetic LAS -> correct DTM tiles (extent, resolution, ground-only, nodata -9999); tiles clipped to the original tile bounds (adjacent tiles abut exactly); parallel run matches serial run; no intermediate point cloud written to disk — each output tile reads only intersecting neighbor LAZ tiles (Phase 2 remark); ground-free tiles yield all-nodata tiles; `dtm.vrt` built. |
| 4 | `dataset.py` + `models/` (protocol, registry, U-Net) | Round-trip tests pass; IoU metric registered; U-Net trains 1 epoch on synthetic data (binary and multi-class shapes); registry resolves default backend and rejects unknown names; `model_config.json` round-trips; memmap dataset streams per-index windows. |
| 5 | `train.py` | Checkpoint + log + metadata written to expected folders; early stopping honored. |
| 6 | `io/tiling.py` + `predict.py` | Stitching test passes; e2e predict on synthetic area produces raster + shapes + merged output; predictions carry the Detection IR schema (score/class_id); batched inference equals single-window predictions. |
| 7 | `evaluate.py` | Report CSV on synthetic pair has expected six counts (single class); per-class rows when labels carry >1 class. |
| 8 | `cli.py` + README | All four commands run end-to-end on the synthetic fixture tree. |
| 9 | Tests/CI complete | Full suite green in CI. |
| 10 | Multi-class completion (3.3: MC1-MC7) | 3-class synthetic fixture green: IR and raster agree under argmax; no phantom classes; weighted/balanced options change training as documented; `class_names` and per-class cuts round-trip through train -> predict -> evaluate; confusion counts correct; binary outputs unchanged on single-class fixtures. |

## 6. Risks and notes

- **Keras 3 vs original TF 2.6**: model *weights* cannot be migrated losslessly between
  the two, but this refactor retrains from data, so no model migration is needed.
  Legacy `.h5` loading is best-effort only.
- **TPI vectorization** is the only place where "identical output" must be proven
  rather than assumed; the golden test is mandatory before deleting the reference
  implementation.
- **PDAL vs LAStools gridding**: both rasterize a TIN now (`filters.delaunay` +
  `filters.faceraster` vs `las2dem`); `faceraster`'s `max_triangle_edge_length` is
  the direct equivalent of `-kill 50` but is left unlimited by default, so values
  near data gaps and in sparse areas can still differ. Acceptance is equivalent
  coverage and interpolation quality (verified visually/statistically on real
  data), not bit parity; downstream steps are unaffected because models are
  retrained on the new DTMs.
- **PDAL packaging**: conda-forge is the reliable install channel (pip wheels are
  platform-dependent); CI installs PDAL via conda and the README points Windows
  users to conda.
- **VRT vs rasterio GDAL ABI**: VRTs are created with the standalone `gdalbuildvrt`
  binary (`gdal-utils`) and only read through rasterio (ABI-safe); the plan
  deliberately avoids importing `osgeo.gdal` alongside rasterio.
- **Parallel execution memory**: each worker loads its tile's points in memory;
  `cores` is exposed as the pool size so users can throttle it for large LAZ files.
  With tile-index neighbor reads (Phase 2 remark), per-worker memory is bounded
  by the tile + buffer region, not by whole neighbor files.
- **COPC/EPT intermediate**: pass 1 needs disk for the ground-point intermediate
  (roughly the size of the ground-class points). COPC is preferred because its
  octree clustering enables true spatially bounded reads; EPT is the
  directory-based fallback. Both ship with PDAL — no extra dependency.
  Superseded by the Phase 2 remark: no intermediate is written to disk; a LAZ
  tile index directs per-tile neighbor reads instead.
- **Mixed precision**: `mixed_float16` numerics differ slightly from float32; it is
  optional and off by default, and all parity tests run with it disabled.
- **pandas/geopandas semantics**: `overlay` results and `dissolve` row ordering may
  differ in column order between versions; tests assert on geometry and attributes,
  not row order.
- Dropped assumptions: the fixed 9-character mapsheet name prefix is gone — tile
  identity comes from the tile index and spatial matching, so filenames carry no
  mapsheet information and no length convention. Assumptions kept (now checked, not
  assumed): all feature layers pixel-aligned per tile, EPSG uniform across the
  project (default `EPSG:3067` from config).
- **Backend abstraction cost**: the protocol/registry/IR indirection is added now for
  backends that do not exist yet. The seams are deliberately thin: YOLO, Faster R-CNN,
  and Mask R-CNN differ mainly in their output type (boxes/masks vs. score rasters),
  which the Detection IR already models, so their integration should be confined to
  `models/<backend>/`. Torch-based backends add a torch dependency only through the
  `lidark[torch]` extra and are never imported unless selected in config. The default
  U-Net path keeps exact parity with the original.

## 7. Performance bottlenecks and optimizations

Analysis of the planned design. Backwards compatibility is not the primary
concern, so the faster variant is adopted as the default behavior wherever the
phase text above references an entry here.

| ID | Stage | Bottleneck | Adopted improvement |
|---|---|---|---|
| P1 | DTM | Per-tile pipelines each re-read and re-decompress every intersecting neighbor LAZ: total decompression work is O(tiles x neighbors), and LAZ decompression dominates DTM runtime. | Two-pass design (Phase 2): one read per input file into a COPC/EPT ground intermediate; tiles then issue windowed `bounds` queries — neighbor data is fetched, never re-decompressed. `--no-intermediate` retains the naive mode for small inputs. **Revised (Phase 2 remark)**: the on-disk ground intermediate is dropped — a tile index over the input LAZ tiles directs each output tile to read only its intersecting neighbors. |
| P2 | DTM | Each worker holds every intersecting point cloud in RAM. | Windowed queries bound per-worker memory to the tile + buffer region; `cores` bounds total. |
| P3 | TPI | Plan initially ran tiles serially; per-tile kernel launches (GPU) or single-thread scipy (CPU) leave cores idle. | CPU: same `ProcessPoolExecutor` as DTM. GPU: stack tile windows and run one batched `conv2d` per chunk. |
| P4 | TPI | float64 temporaries double memory bandwidth; `uniform_filter` is already O(H x W) (separable), so bandwidth is the limit. | float32 accumulation by default; float64 mode kept for the bit-parity golden test. |
| P5 | Dataset / training | `np.load` + `tf.data.from_tensor_slices` materializes the full array and then **copies it again** into the TF graph — multi-GB datasets OOM or thrash. | Disk-backed memmap (`samples.dat`/`labels.dat`); the pipeline reads per-index windows from the memmap with `prefetch(AUTOTUNE)` — RAM bounded by batch size, no graph copy. |
| P6 | Training | Keras defaults underutilize the GPU at 512x512/batch 32. | Parallel `map` (flips) + prefetch; optional mixed precision and XLA `jit_compile` behind `ModelConfig` flags; img/s per epoch logged. |
| P7 | Prediction | Stride 128 = 75% window overlap: ~16x the inference of a disjoint tiling; cost is per-window, GPU-bound. | `overlap` configurable in `ModelConfig` (default 128 kept for edge quality; 256 halves cost with near-identical results on dense targets). |
| P8 | Prediction | One forward pass per window (original behavior) starves the GPU with tiny batches. | Windows processed in batches (`pred_batch_size`, default 32); optional mixed-precision inference. |
| P9 | Prediction | Stitching loop does Python arithmetic per window; `rasterstats.zonal_stats` rasterizes the tile once **per polygon** — O(polygons x pixels). | Stitching via `np.maximum` on preallocated window views; polygon scores in one pass: `rasterize` all polygons into a region-id image, one `scipy.ndimage.mean` reduction — O(pixels). `rasterstats` removed from dependencies. |
| P10 | Merge | Per-tile shapefile write -> re-read all -> merge is a full disk round trip for data already in memory. | Detection IR accumulates in memory (`pd.concat`); per-tile GeoPackages only on request; merge dissolves directly on the in-memory IR. |
| P11 | Evaluation | `overlay(intersection)` computes pairwise overlaps for all label x prediction combinations. | `sjoin` (STRtree) selects candidate pairs first; geometry intersection only on candidates — typically O(N + hits) instead of O(N x M). |
| P12 | Vector IO | fiona engine is the slowest geopandas backend. | `engine="pyogrio"` everywhere (already in the dependency set). |

Notes:

- The two highest-impact items are P1 (DTM I/O amplification — it changes the
  asymptotics of the whole LiDAR stage) and P5+P7+P8 (prediction/training
  throughput — batch size and overlap dominate wall time there).
- Parity is preserved deliberately where it is cheap (TPI float64 mode, overlap
  default 128) and abandoned where the cost is structural (memmap dataset,
  one-pass polygon scoring, in-memory merge) — matching the "compatibility not
  primary" constraint.
- The smoke test logs tiles/s (DTM, TPI) and img/s (train, predict) so performance
  regressions are visible across changes; no hard CI threshold is enforced.

### 7.1 Polygon scores under overlapping detections

Dense-segmentation detections are disjoint after polygonization, so the
one-pass scorer (P9) is exact for the default U-Net path. Overlapping
geometries still arise in three expected places: multi-class dense models
(class regions may overlap), sparse detector backends (YOLO / Faster / Mask
R-CNN emit overlapping boxes/masks), and user-supplied labels. The scorer's
behavior there is defined by an `overlap` strategy:

| Strategy | Semantics | Cost |
|---|---|---|
| `independent` (default) | Each polygon is scored over its **full geometry**; contested pixels count for every covering polygon — identical to the original `zonal_stats`. A detection's confidence reflects its whole footprint, so evaluation cuts (`prob_cut`) behave as before. | O(sum of polygon areas): per-polygon rasterization into a bounding-box window. Typically *cheaper* than the full-tile pass for sparse detections; worse only when polygons tile the raster with heavy overlap. |
| `priority` | The current one-pass behavior, formalized: later-rasterized polygons claim contested pixels; others' means exclude them. Fast but arbitrary — only valid for provably disjoint inputs (the post-merge dense path). | O(tile pixels). |
| `exclusive` | Contested pixels are assigned to the **nearest** polygon (boundary distance via a multi-label distance transform), giving every detection a well-defined "own" region; the score is the mean over that region. Best when adjacent detections bleed into each other and instance-level areas matter. | O(tile pixels + label distance transform). |

Adaptive dispatch: `polygon_scores()` starts with the O(pixels) one-pass
rasterization; a cheap STRtree self-join detects whether any pair of polygons
actually overlaps, and only then re-runs with the requested strategy
(`independent` by default). Disjoint inputs (the common dense case) never pay
the per-polygon path.

For overlapping detections the IR may carry both views: `score` (strategy
result) plus optional `score_inclusive` (full-footprint mean) and
`overlap_frac`, so evaluation can filter on either without re-rasterizing.

Adjacent-but-fused features are a different problem: when the thresholded mask
merges two touching objects into one connected region, no scoring strategy
separates them — instance separation (watershed on the probability surface, or
a detector backend) is the intended future remedy, and the Detection IR schema
already accommodates it.
