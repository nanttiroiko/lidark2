# lidark

lidark detects archaeological features (such as tar production kilns) from
airborne laser scanning data using a deep learning pipeline. It is the
modernized implementation of the original
[LIDARK](old_code/) script: LiDAR point clouds are processed into elevation
and terrain models, a U-Net is trained to recognize the features, and
trained models produce georeferenced detection polygons.

## Setup

lidark trains and predicts on the GPU with PyTorch. The instructions
below install a CUDA-enabled torch on both native Windows and Linux/WSL2,
so start by installing the NVIDIA driver for your GPU - `nvidia-smi` must
run before you continue. The geospatial stack (GDAL/rasterio, PDAL)
installs most reliably through conda-forge.

### Windows (native)

Install the normal NVIDIA driver, then confirm the GPU is visible from
PowerShell:

```console
nvidia-smi
```

Create the environment and install lidark plus the CUDA-enabled torch
build. The base `pip install -e .` pulls the CPU torch wheel; to replace
it with the GPU build, install the exact version with its `+cuXXX` local
tag from a CUDA wheel index - use the index currently listed at
[pytorch.org](https://pytorch.org/get-started/locally/) for your driver's
CUDA version. A bare `pip install torch --index-url ...` is a no-op here:
pip sees the already-installed CPU build as satisfying the requirement
and keeps it. `pip index versions torch --index-url <index>` lists the
available CUDA builds:

```console
conda create -n lidark -c conda-forge python=3.11 gdal pdal
conda activate lidark
pip install -e .
pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
```

Verify the GPU is visible to torch:

```console
python -c "import torch; print(torch.cuda.is_available())"
```

Note: conda-forge numpy/scipy and the pip torch wheels each bundle an
OpenMP runtime; with both loaded, BLAS calls can hard-abort. lidark sets
`KMP_DUPLICATE_LIB_OK=TRUE` itself (in `lidark/__init__.py`), so
torch-based paths work next to numpy out of the box.

Also always run lidark and pytest from an activated environment
(`conda activate lidark`). The conda-forge numpy in this stack delay-loads
its BLAS runtime from the environment's `Library\bin`, which is only on
the DLL search path in an activated env; calling the env's `python.exe`
directly can hard-abort on the first BLAS call (Windows exit code
`0xC06D007F`).

### Linux / WSL2

On WSL2, install the normal NVIDIA driver on Windows only - do **not**
install a driver inside WSL - and confirm the GPU is visible from the
shell:

```console
nvidia-smi
```

The install is otherwise identical to native Windows:

```console
conda create -n lidark -c conda-forge python=3.11 gdal pdal
conda activate lidark
pip install -e .
pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"
```

### Optional extras

- `pip install -e ".[torch]"` - the torchvision-based model backends
  (Faster R-CNN, Mask R-CNN); torch itself is a core dependency
  (U-Net, GPU TPI)
- `pip install -e ".[dev]"` - pytest, ruff, mypy for development

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

### 2. train: train a model

Builds a training dataset from the label shapefiles in
`DeepLearning/train_annotations/` and trains a model (default backend:
U-Net).

```console
lidark train --area Naljanka \
    --name tar_kilns_v1 \
    --true-objects Tar_kilns.shp --false-objects OtherFeatures.shp \
    --feature-list 02_TPI/radius_20 02_TPI/radius_30 \
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
program. `--dataset-id` reuses an existing dataset instead of creating a
new one.

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

lidark train --area Naljanka \
    --name tar_kilns_rcnn_v1 \
    --true-objects Tar_kilns.shp --false-objects OtherFeatures.shp \
    --feature-list 02_TPI/radius_20 02_TPI/radius_30 \
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
    --feature-list 02_TPI/radius_20 02_TPI/radius_30 \
    --backend fasterrcnn --test-area Test_area.gpkg
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

The modules are also usable directly from Python:

```python
from lidark import create_map, train, predict
```

An `evaluate` command is available as well; see
[MODERNIZATION_PLAN.md](MODERNIZATION_PLAN.md) for the full design and
workflow details.
