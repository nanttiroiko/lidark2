# lidark^2

lidark^2 detects archaeological features (such as tar production kilns) from
airborne laser scanning data using a deep learning pipeline. It is the
modernized implementation of the original
[LIDARK](old_code/) script: LiDAR point clouds are processed into elevation
and terrain models, a U-Net is trained to recognize the features, and
trained models produce georeferenced detection polygons.

## Setup

lidark trains and predicts on the GPU with PyTorch, so start by
installing the NVIDIA driver for your GPU - `nvidia-smi` must run before
you continue. On WSL2, install the driver on Windows only - do **not**
install a driver inside WSL.

### Choosing one installation route

lidark combines two binary (compiled) stacks: the LiDAR/geospatial stack
(GDAL, PDAL, rasterio, geopandas) and the PyTorch stack. conda-forge
packages and pip wheels each bundle their own low-level runtimes
(OpenMP, BLAS, CUDA), and mixing the two in one environment loads
incompatible duplicates into one process - typical symptoms are torch DLL
load failures on Windows (`WinError 127` on torch's `shm.dll`) and OpenMP
aborts at the first BLAS call. Pick **one route per environment** and do
not install the other stack's binaries into it:

- **Conda route** (recommended): every binary package comes from
  conda-forge. Works the same on native Windows and Linux/WSL2 and needs
  no system packages.
- **Pip route** (Linux/WSL2 only): everything comes from pip/PyPI. PDAL
  and GDAL publish no wheels on PyPI - their packages build from source -
  so the LiDAR stack first needs the matching system dev packages. On
  native Windows this route is not practical; use the conda route there.

### Conda route (recommended)

```console
conda create -n lidark -c conda-forge python=3.11 gdal python-pdal \
    pytorch-gpu numpy scipy geopandas pyogrio shapely rasterio tqdm
conda activate lidark
pip install -e . --no-deps
python -c "import torch; print(torch.cuda.is_available())"
```

pip is used only for the pure-python lidark package itself: the
`--no-deps` flag is important, because it stops pip from pulling any
binary dependency as a wheel - a pip torch wheel next to the conda
stack is exactly the mixing this route avoids. `gdalbuildvrt` comes
with the conda GDAL package (the PyPI `gdal-utils` package is not
needed), and conda-forge resolves the `pytorch-gpu` CUDA build matching
the installed driver.

Notes:

- Use `pytorch-cpu` instead of `pytorch-gpu` on machines without an
  NVIDIA GPU. For the newest GPU architectures (RTX 50xx, compute
  capability 12.0), check that the resolved build contains `sm_120`
  kernels - a build without them fails at the first GPU op with
  `cudaErrorNoKernelImageForDevice`.
- lidark sets `KMP_DUPLICATE_LIB_OK=TRUE` itself (in
  `lidark/__init__.py`), so a residual OpenMP duplicate is tolerated
  instead of aborting.
- Always run lidark and pytest from an activated environment
  (`conda activate lidark`). The conda-forge numpy in this stack
  delay-loads its BLAS runtime from the environment's `Library\bin`,
  which is only on the DLL search path in an activated env; calling the
  env's `python.exe` directly can hard-abort on the first BLAS call
  (Windows exit code `0xC06D007F`).
- An environment that already contains pip-installed torch or numpy
  next to the conda stack cannot be repaired reliably in place -
  `pip uninstall` the pip copies or recreate the environment.

### Pip route (Linux/WSL2 only)

Install the system dev packages first (Ubuntu/Debian names; they vary by
distribution) - the PDAL and GDAL Python packages compile against them:

```console
sudo apt install gdal-bin libgdal-dev libpdal-dev
```

Then create the environment and install lidark:

```console
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"
```

The base `pip install -e .` pulls the CPU torch wheel; to replace it
with the GPU build, install the exact version with its `+cuXXX` local
tag from a CUDA wheel index - use the index currently listed at
[pytorch.org](https://pytorch.org/get-started/locally/) for your driver's
CUDA version. A bare `pip install torch --index-url ...` is a no-op here:
pip sees the already-installed CPU build as satisfying the requirement
and keeps it. `pip index versions torch --index-url <index>` lists the
available CUDA builds.

Notes:

- For the newest GPU architectures (RTX 50xx, compute capability 12.0)
  use a `cu128`-or-newer index instead - a `cu126` build has no `sm_120`
  kernels and every GPU op fails with `cudaErrorNoKernelImageForDevice`.
- `gdal-utils` (installed with lidark) provides the `gdalbuildvrt`
  command the VRT mosaics are built with; it needs the GDAL Python
  bindings, so also run `pip install gdal` (it compiles against the
  `libgdal-dev` installed above). Without a working `gdalbuildvrt`,
  `create-map --map-type TPI` fails - train and predict fall back to
  per-tile reads with a warning.

### Optional extras

Both extras exist in both routes - only the tool differs:

- torchvision (the Faster R-CNN and Mask R-CNN backends; torch itself
  is a core dependency - U-Net, GPU TPI): conda route -
  `conda install torchvision`; pip route - `pip install -e ".[torch]"`,
  using the torchvision build from the same `--index-url` as the torch
  build when training on the GPU.
- dev tools (pytest, ruff, mypy): conda route -
  `conda install pytest ruff mypy pre-commit`; pip route -
  `pip install -e ".[dev]"`.

This provides the `lidark` command. `train` and `predict` switch to the
GPU automatically when torch finds one; `create-map --map-type TPI --gpu`
selects the torch TPI path. On GPU, `train` and `predict` can be sped up
further with `--mixed-precision` and `--jit-compile` (both off by
default).

## Running the modules

Store your project folder once; every command uses it from then on:

```console
lidark set-project-folder C:/project
```

All commands accept `--project-folder` to override the stored folder for a
single run, and `--epsg` (default `EPSG:3067`). Add `-v` for verbose
logging.

### 1. create-map: LiDAR processing

Turns the point clouds in `Lidar/<area>/00_Original/` into data maps used
for training and prediction.

```console
lidark create-map --area Naljanka \
    --map-type DTM --pixel-size 0.25 --cores 8
lidark create-map --area Naljanka \
    --map-type TPI --buffer 20 --gpu
```

- `DTM` - ground elevation model per input LAZ tile (PDAL TIN
  rasterization, parallel per-tile pipelines). Each tile reads only the
  intersecting neighbouring LAZ tiles; no intermediate point cloud is
  written to disk. `--buffer` is the point-read margin in metres (default
  2). Output tiles are clipped to the original tile bounds; gaps (no
  ground points, e.g. water) are nodata (`-9999`).
- `TPI` - topographic position index computed from the DTM product.
  `--buffer` is the TPI radius in metres (default 30)
  (output: `Lidar/<area>/02_TPI/radius_<buffer>/`). The TPI uses a circular
  filter of the given radius (full weight, half-weight soft edge ring).
  Nodata cells in the DTM are excluded from the TPI calculation and stay
  nodata (`-32768`) in the output; cells with no neighbouring tile at the
  area edge are handled the same way.
- A failing tile never aborts a create-map run: an input file whose header
  cannot be read is skipped, a tile whose processing fails (PDAL pipeline,
  window read, kernel, write — on the CPU, worker-pool, or GPU path) is
  written as an all-nodata tile, and the VRT mosaic and tile index are
  built over the tiles that were actually written. Every failure is logged
  as a warning; the run only fails if no tile at all could be written.
- Re-running `create-map` resumes instead of restarting: a tile whose output
  already exists (and is a readable raster at the current `--pixel-size`)
  is skipped, with one log line per skipped tile. A changed `--pixel-size`
  reprocesses the product automatically; delete the product folder to force
  a full reprocess otherwise (e.g. after changing `--buffer`).
- Large areas (thousands of tiles) work: the VRT mosaic is built by passing
  the tile list to `gdalbuildvrt` via `-input_file_list` instead of the
  command line, which Windows limits to 32767 characters (WinError 206).

### 2. train: train a model

Builds a training dataset from the label shapefiles in
`DeepLearning/train_annotations/` and trains a model (default backend:
U-Net).

```console
lidark train --area Naljanka \
    --name tar_kilns_v1 \
    --true-objects Tar_kilns.shp --false-objects OtherFeatures.shp \
    --raster-type 02_TPI/radius_20 02_TPI/radius_30 \
    --backend unet --test-area Test_area.gpkg
```

`--name` is required: it names the model so it is easy to remember which
model was which. The model id is the sanitized name plus the timestamp
(`<name>_<YYYYMMDD-HHMMSS>`, e.g. `tar_kilns_v1_20260101-120000`), naming
`DeepLearning/model/<name>_<timestamp>/` — the training log
(`training_log.json`) is written inside that directory, alongside the
predictions/reports below; the command prints the id used by predict.
Invalid filename characters and spaces in the name become `_`.
Optionally pass `--test-area <layer>` — a polygon layer (e.g. a `.gpkg`
file) in `DeepLearning/train_annotations/` — to hold out a test region:
every tile whose footprint intersects the polygons is excluded from
training (and recorded in the dataset metadata).
Hyperparameters such as `--max-epochs`, `--batch-size`, `--learning-rate`,
`--img-size`, and `--seed` can be tuned; the defaults match the original
program. `--max-samples-per-epoch N` caps the training windows processed
per epoch — each epoch draws a fresh seeded random subset of the training
windows, so long datasets can be iterated on quickly; validation is never
capped. `--dataset-id` reuses an existing dataset instead of creating a
new one.

#### Training on multiple areas

`--area` accepts several areas: one dataset is built from all of their
tiles and one model is trained on it — useful when each area holds few
labelled objects.

```console
lidark train --area Naljanka Kuivaniemi \
    --name tar_kilns_v1 \
    --true-objects Tar_kilns.shp --false-objects OtherFeatures.shp \
    --raster-type 02_TPI/radius_20 02_TPI/radius_30 \
    --test-area Test_area.gpkg
```

- Every listed area must contain the feature layers. The label
  shapefiles (and the optional test-area / val-area polygons) are
  shared: labels match tiles spatially per area, so a label simply
  places windows in whichever areas it intersects.
- All areas must share one pixel size (build the DTM/TPI products with
  one `--pixel-size`); a mismatch is a clear error, because training
  windows are placed in pixels.
- The dataset metadata records all areas (`areas`); excluded test tiles
  are recorded as `<area>/<tile>` so equal tile ids in different areas
  stay distinguishable. Duplicate area names are de-duplicated.
- A model trained this way is used exactly like any other: `predict
  --areas ... --model-id <id>` works unchanged.

#### Validation split

The validation split is **object-disjoint by default** (`--val-split-mode
group`): label objects within the group-buffer distance — default: the
training window diagonal — form one spatial group, and whole groups are
assigned to one side, so the same object, or a cluster of neighbouring
objects, never appears in both training and validation. The split is
**class-stratified**: every class with at least two groups keeps windows
on both sides, so per-class validation metrics stay meaningful. With the
default buffer, a window placed around one group cannot even contain
another group's pixels; a smaller buffer is warned about if any window
still crosses the split.

The knobs:

- `--val-split` — validation fraction per class (default 0.2). Because
  whole groups move, the achieved fraction is approximate; the training
  log records the achieved counts (`validation split details`).
- `--val-split-mode window` — the original window-level shuffle + split
  (parity with the old program; objects may appear in both halves).
  Datasets created before object provenance was recorded fall back to
  this mode with a warning — re-create the dataset to use group mode.
- `--val-group-buffer <m>` — the cluster radius in metres (default: the
  training window diagonal, e.g. about 181 m for a 512-px window at
  0.25 m/px).
- `--val-area <layer>` — a polygon layer in
  `DeepLearning/train_annotations/` (like the test area, but for
  validation): every window whose footprint intersects it is validation
  only, together with its whole group.
- `--val-per-class 1=0.3 2=0.1` — per-class validation fractions;
  unlisted classes use `--val-split`.

Three backends ship. `--backend unet` (default, torch) is the semantic
segmentation of the original program. `--backend maskrcnn` trains a Mask
R-CNN instance segmentation model instead (torchvision, requires
`lidark[torch]`): instance targets come from the label class maps, and
training is validated with **mask mAP** — AP50 by default, or AP75 with
`--mask-ap-iou 0.75` (the monitored checkpoint/early-stopping metric).
Mask R-CNN predictions are per-instance polygons in the same output
GeoPackage as the U-Net's, so `predict` and `evaluate` work unchanged.
`--backend fasterrcnn` trains a Faster R-CNN object detection model
(torchvision, requires `lidark[torch]`): the same instance targets become
bounding boxes, and training is validated with **box mAP@0.5**
(PASCAL VOC) by default — or AP75 with `--det-ap-iou 0.75`. Each
detection is one box polygon in the same output GeoPackage, so `predict`
and `evaluate` work unchanged there as well.

A Mask R-CNN training run uses the same data and flags as the U-Net —
only the backend (and optionally the mAP IoU threshold) differ:

```console
pip install -e ".[torch]"  # torchvision for the maskrcnn backend
# conda route instead: conda install torchvision (see Setup)

lidark train --area Naljanka \
    --name tar_kilns_rcnn_v1 \
    --true-objects Tar_kilns.shp --false-objects OtherFeatures.shp \
    --raster-type 02_TPI/radius_20 02_TPI/radius_30 \
    --backend maskrcnn --mask-ap-iou 0.75 --test-area Test_area.gpkg
```

Omit `--mask-ap-iou` for the default AP50; `--mask-ap-iou 0.75` monitors
AP75 instead. A Faster R-CNN training run is the same command with
`--backend fasterrcnn` (omit `--det-ap-iou` for the PASCAL VOC mAP@0.5
default):

```console
lidark train --area Naljanka \
    --name tar_kilns_frcnn_v1 \
    --true-objects Tar_kilns.shp --false-objects OtherFeatures.shp \
    --raster-type 02_TPI/radius_20 02_TPI/radius_30 \
    --backend fasterrcnn --test-area Test_area.gpkg
```

#### Multi-class detection

Label shapefiles may carry an integer **`class_id` column** (values >= 1)
to train one model on several feature types at once — e.g. tar kilns and
charcoal pits — instead of one binary model per type. A shapefile without
the column means a single class, which reproduces the original binary
behavior exactly, so existing projects are unaffected.

```console
lidark train --area Naljanka \
    --name features_multiclass_v1 \
    --true-objects Features.shp \
    --raster-type 02_TPI/radius_20 02_TPI/radius_30 \
    --class-names tar_kiln charcoal_pit \
    --class-weights 1.0 2.0 --balanced-sampling
```

How the classes flow through training:

- The observed label ids are remapped to **compact ids 1..C** — sparse
  ids such as {1, 5} do not create empty intermediate classes (the gap is
  warned about, since ids are your data). The mapping is recorded as
  `class_id_map` in `model_config.json`, so predictions (compact ids)
  stay traceable to your original ids.
- `--class-names` gives one human-readable name per class (compact-id
  order); the names appear in the prediction and evaluation outputs.
  Omit it for automatic `class_1`, `class_2`, ...
- `--class-weights 1.0 2.0` weights the loss per class (compact-id
  order) when one class is rarer than the other; `--balanced-sampling`
  instead normalizes the windows per object per class so every class
  contributes about equally. Both are off by default.
- The U-Net grows a (C+1)-channel softmax head for multiple classes
  (single class keeps the original sigmoid), validated with IoU over
  the foreground classes; the validation split is class-stratified so
  per-class metrics stay meaningful. The detector backends (Mask R-CNN,
  Faster R-CNN) are natively multi-class: instance targets derive from
  the same class maps.
- Every multi-class option is a no-op for single-class runs — binary
  behavior is unchanged.

#### Fine-tuning an existing model

Training on a new area or a refreshed label set is faster and more
data-efficient when warm-started from an existing model. Pass
`--base-model <id>` with the id of an earlier trained model (as printed by
`train`): its **weights** become the starting point, everything else —
the dataset, the hyperparameters, the fresh model id and directory —
comes from the new run. The base model is never modified; each fine-tuned
model is a new entry under `DeepLearning/model/`, so a lineage such as
`tar_kilns_v1` → `tar_kilns_v2` stays fully traceable.

```console
lidark train --area Naljanka \
    --name tar_kilns_v2 \
    --true-objects Tar_kilns.shp --false-objects OtherFeatures.shp \
    --raster-type 02_TPI/radius_20 02_TPI/radius_30 \
    --base-model tar_kilns_v1_20260101-120000 \
    --learning-rate 0.0001
```

Notes:

- A **lower `--learning-rate` than scratch training is typically wanted**
  (e.g. `0.0001` instead of the `0.001` default): the weights already sit
  near a good solution and large steps destroy that. The learning rate is
  never changed automatically — it stays explicit.
- The base model's **backend, input shape** (`--img-size` x the number of
  `--raster-type` layers) **, and class count must match** the new run —
  for the U-Net, `--unet-size` must match too. A mismatch raises a clear
  error before any training starts; changing the class count is not
  supported by warm start (train from scratch instead). Feature layers
  with the same count but different names load cleanly but are warned
  about — warm start assumes the same data products.
- The new model records `base_model_id` in its `model_config.json` and
  training log, and the per-epoch history shows the warm start: epoch-1
  loss starts near the base model's last, not at a scratch run's high
  initial loss.
- `--dataset-id` combines with `--base-model` for the fastest iteration:
  warm-start from a prior model on an existing dataset.

Fine-tuning works for every backend (`unet`, `maskrcnn`, `fasterrcnn`)
with the same flag. From Python, pass `base_model_id` to
`lidark.train` (the learning rate is part of `ModelConfig`):

```python
from lidark import train
from lidark.config import ModelConfig, ProjectLayout

layout = ProjectLayout(project_folder="C:/project")
train(layout, "Naljanka", model_name="tar_kilns_v2",
      true_objects="Tar_kilns.shp", false_objects="OtherFeatures.shp",
      feature_list=["02_TPI/radius_20", "02_TPI/radius_30"],
      model_config=ModelConfig(learning_rate=0.0001),
      base_model_id="tar_kilns_v1_20260101-120000")
```

### 3. predict: detect features

Predicts detections with a trained model over the same feature layers the
model was trained on.

```console
lidark predict --areas Naljanka Kuivaniemi \
    --model-id tar_kilns_v1_20260101-120000 --pred-type all
```

`--model-id` takes the id printed by train (`--model-timestamp` is kept as
an alias). Per-tile probability rasters are written under
`DeepLearning/model/<name>_<timestamp>/predictions/<area>/raster/`, and the
merged detections (with probability and area attributes) to
`predictions/<area>/merged_shapes_<pred-type>.gpkg`. Use `--pred-type test`
to predict only the held-out test tiles: pass `--test-area <layer>` (the
same test-area polygon layer as in training) or `--test-tile <id>`.

Predicting with a Mask R-CNN or Faster R-CNN model is the same command
with its model id — the backend is read from the model's
`model_config.json`:

```console
lidark predict --areas Naljanka \
    --model-id tar_kilns_rcnn_v1_20260101-120000 --pred-type all \
    --threshold 0.5
```

A Mask R-CNN model emits **one polygon per detected instance** (no
probability-raster stitching); `--threshold` drops instances below the
detection score. The merged detections and the per-tile raster output
(the class-id footprint of the predicted instances) land in the same
locations as the U-Net's, so `evaluate` works unchanged.

**Multi-class models** work through the same command. Every detection
polygon carries `class_id` and `class_name` attributes in the merged
GeoPackage, and the per-tile raster output is a class-id map
(0 = background). A dense (U-Net) multi-class model decides per pixel:
the pixel belongs to its highest-scoring class, but only when that
class's own probability passes the threshold — the vector output and the
raster always agree, and no pixel is claimed by two classes. Thresholds
can be tuned per class instead of one value for everything:

```console
# one threshold for every class (default 0.5)
lidark predict --areas Naljanka --model-id features_multiclass_v1_... --threshold 0.5

# per-class operating points: class 1 at 0.4, class 2 at 0.6
lidark predict --areas Naljanka --model-id features_multiclass_v1_... \
    --threshold 1=0.4 2=0.6
```

A per-class threshold mapping must cover every class of the model; a
missing class raises a clear error before any tile is processed.

The modules are also usable directly from Python:

```python
from lidark import create_map, train, predict
```

### 4. evaluate: detection quality per class

Scores a model's merged predictions against evaluation annotations and
writes the report to `DeepLearning/reports/<model_id>/<eval_name>.csv`.
The labels live in `DeepLearning/evaluate_annotations/<eval_name>/` and
may carry the same `class_id` column as the training labels.

```console
lidark evaluate --model-id tar_kilns_v1_20260101-120000 \
    --eval-name test1 --true-objects Tar_kilns_eval.shp
```

Multi-class specifics:

- The report has **one row per class plus an `overall` row**, each with
  the six counters (`Predicted / Fully detected / Partly detected /
  Not detected / Detected with no label / False detected`); a
  single-class report is exactly the original one-row report.
- Label ids may be your **original** ids: they are remapped through the
  model's `class_id_map` automatically. A class the model never saw keeps
  its id and shows up as `Not detected` (truthful, not silently
  dropped); row `class_name`s come from the model's `--class-names`.
- `--size-cut` and `--prob-cut` accept one value for all classes
  (defaults 1.0 m2 / 0.8) or per-class operating points:
  `--prob-cut 1=0.8 2=0.6`.
- A **confusion report** is written next to the main report as
  `<eval_name>_confusion.csv`: for every detected label, the class of its
  best-overlap prediction (so misclassifications are visible as
  cross-class pairs), plus the detections overlapping no label grouped
  by predicted class (`true_class_id` = `none`).

See [MODERNIZATION_PLAN.md](MODERNIZATION_PLAN.md) for the full design
and workflow details.
