"""Command-line interface: the four LIDARK workflows as argparse subcommands.

Every facade stays usable directly from Python; this module only maps
arguments onto :func:`lidark.create_map`, :func:`lidark.train`,
:func:`lidark.predict`, and :func:`lidark.evaluate`. The default project
folder comes from :mod:`lidark.user_config` (``lidark set-project-folder``);
``--project-folder`` overrides it for one run, and ``--epsg`` defaults to
``EPSG:3067`` from config. ``-v``/``--verbose`` switches logging to DEBUG.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lidark.config import DEFAULT_EPSG, ModelConfig, ProjectLayout
from lidark.logging import configure_logging, get_logger
from lidark.user_config import get_project_folder, set_project_folder

LOGGER = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    """The ``lidark`` argument parser (four subcommands)."""
    parser = argparse.ArgumentParser(
        prog="lidark",
        description=(
            "LiDAR-based detection of archaeological features with deep "
            "learning (modernized LIDARK)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose (DEBUG) logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--project-folder",
            default=None,
            help="project root path (default: the folder stored by `lidark set-project-folder`)",
        )
        p.add_argument(
            "--epsg", default=DEFAULT_EPSG, help=f"default CRS (default: {DEFAULT_EPSG})"
        )

    # --------------------------------------------------- set-project-folder
    spf = subparsers.add_parser(
        "set-project-folder",
        help="store the default project folder used by the other commands",
    )
    spf.add_argument("project_folder", help="path to the project root (must exist)")

    # ------------------------------------------------------------- create-map
    create = subparsers.add_parser(
        "create-map",
        help="generate a DTM or TPI data-map product",
        description="Generate a DTM or TPI data-map product. Tiles whose output "
        "already exists are skipped (one log line each), so re-running the "
        "command resumes an interrupted run.",
    )
    common(create)
    create.add_argument("--area", required=True, help="project area name")
    create.add_argument(
        "--map-type",
        required=True,
        help="DTM (PDAL, from LAZ originals) or TPI (from the DTM product)",
    )
    create.add_argument(
        "--buffer",
        type=float,
        default=None,
        help="DTM: point-read margin in m (default 2). TPI: radius in m (default 30)",
    )
    create.add_argument("--pixel-size", type=float, default=0.25, help="raster resolution (m/px)")
    create.add_argument("--cores", type=int, default=8, help="worker processes for tile pipelines")
    create.add_argument("--gpu", action="store_true", help="TPI: batched torch conv2d path")

    # ----------------------------------------------------------------- train
    train_parser = subparsers.add_parser(
        "train", help="build a training dataset and/or train a model"
    )
    common(train_parser)
    train_parser.add_argument(
        "--area", nargs="+", required=True, metavar="AREA", help="project areas"
    )
    train_parser.add_argument(
        "--name",
        required=True,
        help="model name; the model id becomes <name>_<timestamp>",
    )
    train_parser.add_argument("--true-objects", help="true-object shapefile name")
    train_parser.add_argument("--false-objects", help="false-object shapefile name")
    train_parser.add_argument(
        "--raster-type",
        nargs="+",
        metavar="LAYER",
        help="feature layers relative to Lidar/<area>/ (e.g. 01_DTM 02_TPI/radius_20)",
    )
    train_parser.add_argument(
        "--test-area",
        help="test-area polygon layer (e.g. .gpkg) in DeepLearning/train_annotations/; "
        "tiles intersecting it are excluded from training",
    )
    train_parser.add_argument(
        "--base-model",
        help="fine-tune: warm-start the weights from this model id "
        "(<name>_<timestamp>); the base model is never modified and a "
        "lower --learning-rate is typically wanted",
    )
    train_parser.add_argument(
        "--dataset-id", help="load an existing dataset instead of creating one"
    )
    train_parser.add_argument("--backend", default="unet", help="model backend")
    train_parser.add_argument(
        "--mask-ap-iou",
        type=float,
        default=0.5,
        help="maskrcnn backend: IoU threshold of the monitored mask mAP "
        "(0.5 = AP50 default, 0.75 = AP75)",
    )
    train_parser.add_argument(
        "--det-ap-iou",
        type=float,
        default=0.5,
        help="fasterrcnn backend: IoU threshold of the monitored box mAP "
        "(0.5 = PASCAL VOC mAP@0.5 default, 0.75 = AP75)",
    )
    train_parser.add_argument(
        "--img-size",
        nargs=2,
        type=int,
        default=[512, 512],
        metavar=("H", "W"),
        help="training window size in px (default: 512 512)",
    )
    train_parser.add_argument("--max-epochs", type=int, default=200)
    train_parser.add_argument(
        "--max-samples-per-epoch",
        type=int,
        default=None,
        metavar="N",
        help="cap on training windows per epoch: a fresh seeded subset each "
        "epoch, validation never capped (default: all windows)",
    )
    train_parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        metavar="N",
        help="early stopping: stop after N consecutive epochs without improvement "
        "of the monitored metric (default: 10; >= --max-epochs disables it)",
    )
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--val-split", type=float, default=0.2)
    train_parser.add_argument(
        "--val-split-mode",
        choices=["group", "window"],
        default="group",
        help="validation split strategy: 'group' (default) keeps objects and "
        "clusters out of both halves; 'window' is the original window-level split",
    )
    train_parser.add_argument(
        "--val-group-buffer",
        type=float,
        default=None,
        metavar="M",
        help="group mode: objects within this many metres share a split group "
        "(default: the training window diagonal)",
    )
    train_parser.add_argument(
        "--val-area",
        help="group mode: polygon layer in DeepLearning/train_annotations/; "
        "windows intersecting it are validation only",
    )
    train_parser.add_argument(
        "--val-per-class",
        nargs="+",
        metavar="CLASS=FRACTION",
        help="per-class validation fractions (compact ids); unlisted classes use --val-split",
    )
    train_parser.add_argument("--unet-size", type=int, default=16)
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument(
        "--pred-stride",
        type=int,
        default=128,
        help="prediction window stride in px (default: 128)",
    )
    train_parser.add_argument("--pred-batch-size", type=int, default=32)
    train_parser.add_argument(
        "--class-names",
        nargs="+",
        metavar="NAME",
        help="human-readable class names, one per class in compact-id order",
    )
    train_parser.add_argument(
        "--class-weights",
        nargs="+",
        type=float,
        metavar="W",
        help="loss weight per class in compact-id order (default: unweighted)",
    )
    train_parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="class-balanced window sampling (windows per object normalized "
        "per class; default off)",
    )
    train_parser.add_argument("--no-augment", action="store_true", help="disable flip augmentation")
    train_parser.add_argument(
        "--mixed-precision", action="store_true", help="mixed_float16 training"
    )
    train_parser.add_argument("--jit-compile", action="store_true", help="XLA-compile the model")

    # --------------------------------------------------------------- predict
    predict_parser = subparsers.add_parser(
        "predict", help="predict detections with a trained model"
    )
    common(predict_parser)
    predict_parser.add_argument(
        "--areas", nargs="+", required=True, metavar="AREA", help="project areas"
    )
    predict_parser.add_argument(
        "--model-id",
        "--model-timestamp",
        dest="model_id",
        required=True,
        help="model id (<name>_<timestamp>, as printed by train)",
    )
    predict_parser.add_argument("--pred-type", default="all", help='"all" (default) or "test"')
    predict_parser.add_argument(
        "--test-tile",
        nargs="*",
        default=[],
        metavar="ID",
        help="tile IDs for --pred-type test (substring match)",
    )
    predict_parser.add_argument(
        "--test-area",
        help="test-area polygon layer (e.g. .gpkg) in DeepLearning/train_annotations/; "
        "for --pred-type test, predicts tiles intersecting it",
    )
    predict_parser.add_argument(
        "--raster-type",
        nargs="*",
        metavar="LAYER",
        help="override the model's recorded feature layers",
    )
    predict_parser.add_argument(
        "--threshold",
        nargs="+",
        default=["0.5"],
        metavar="T",
        help="detection threshold: one float for all classes, or CLASS=T "
        "per class (e.g. 1=0.4 2=0.6)",
    )
    predict_parser.add_argument(
        "--batch-size", type=int, default=None, help="windows per forward pass"
    )
    predict_parser.add_argument("--stride", type=int, default=None)
    predict_parser.add_argument(
        "--write-tile-shapes",
        action="store_true",
        help="write per-tile GeoPackage vector files",
    )

    # -------------------------------------------------------------- evaluate
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate predictions against evaluation annotations"
    )
    common(evaluate_parser)
    evaluate_parser.add_argument(
        "--model-id",
        "--model-timestamp",
        dest="model_id",
        required=True,
        help="model id (<name>_<timestamp>, as printed by train)",
    )
    evaluate_parser.add_argument("--eval-name", required=True)
    evaluate_parser.add_argument("--true-objects", required=True)
    evaluate_parser.add_argument("--false-objects")
    evaluate_parser.add_argument("--pred-type", default="all")
    evaluate_parser.add_argument("--eval-aoi", help="AOI shapefile (default: label bbox)")
    evaluate_parser.add_argument(
        "--size-cut",
        nargs="+",
        default=["1.0"],
        metavar="A",
        help="min detection area (m2): one float, or CLASS=A per class",
    )
    evaluate_parser.add_argument(
        "--prob-cut",
        nargs="+",
        default=["0.8"],
        metavar="P",
        help="min detection score: one float, or CLASS=P per class",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging(verbose=args.verbose)
    try:
        if args.command == "set-project-folder":
            stored = set_project_folder(args.project_folder)
            print(f"project folder set: {stored}")
            return 0
        project_folder = _resolve_project_folder(args.project_folder)
        layout = ProjectLayout(project_folder=project_folder, epsg=args.epsg)
        if args.command == "create-map":
            _run_create_map(layout, args)
        elif args.command == "train":
            _run_train(layout, args)
        elif args.command == "predict":
            _run_predict(layout, args)
        elif args.command == "evaluate":
            _run_evaluate(layout, args)
        else:  # pragma: no cover - argparse enforces the choices
            parser.error(f"unknown command {args.command!r}")
        return 0
    except Exception:  # noqa: BLE001 - the CLI reports and exits non-zero
        LOGGER.exception("command failed")
        return 1


def _resolve_project_folder(override: str | None) -> Path:
    """The run's project folder: the flag overrides, else the stored default.

    Raises:
        ValueError: If neither a flag nor a stored folder is available.
    """
    if override is not None:
        return Path(override)
    stored = get_project_folder()
    if stored is None:
        raise ValueError(
            "no project folder: pass --project-folder or run `lidark set-project-folder <path>`"
        )
    return stored


def _run_create_map(layout: ProjectLayout, args: argparse.Namespace) -> None:
    from lidark.maps.api import create_map

    tiles = create_map(
        layout.project_folder,
        args.area,
        args.map_type,
        buffer=args.buffer,
        pixel_size=args.pixel_size,
        cores=args.cores,
        gpu=args.gpu,
        epsg=args.epsg,
    )
    print(f"{args.map_type} product written: {len(tiles)} tiles")


def _class_spec(tokens: Sequence[str], flag: str) -> float | dict[int, float]:
    """Parse a scalar or per-class value list (plan 3.3/MC5).

    A single plain token is a scalar for every class; otherwise every token
    must be ``CLASS=VALUE`` and the result is a per-class mapping.
    """
    if len(tokens) == 1 and "=" not in tokens[0]:
        return float(tokens[0])
    spec: dict[int, float] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"{flag}: expected CLASS=VALUE, got {token!r}")
        key, value = token.split("=", 1)
        spec[int(key)] = float(value)
    return spec


def _per_class_fractions(tokens: Sequence[str]) -> tuple[tuple[int, float], ...]:
    """Parse per-class fraction tokens (``CLASS=FRACTION``) for the split."""
    spec: dict[int, float] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"--val-per-class: expected CLASS=FRACTION, got {token!r}")
        key, value = token.split("=", 1)
        spec[int(key)] = float(value)
    return tuple(sorted(spec.items()))


def _run_train(layout: ProjectLayout, args: argparse.Namespace) -> None:
    from lidark.train import train

    if not args.dataset_id and not (args.true_objects and args.raster_type):
        raise ValueError(
            "train requires --true-objects and --raster-type "
            "(or --dataset-id for an existing dataset)"
        )
    config = ModelConfig(
        backend=args.backend,
        feature_list=tuple(args.raster_type) if args.raster_type else (),
        img_size=(args.img_size[0], args.img_size[1]),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        val_split=args.val_split,
        val_split_mode=args.val_split_mode,
        val_group_buffer=args.val_group_buffer,
        val_area=args.val_area,
        val_per_class=_per_class_fractions(args.val_per_class) if args.val_per_class else None,
        max_epochs=args.max_epochs,
        max_samples_per_epoch=args.max_samples_per_epoch,
        early_stop_patience=args.early_stop_patience,
        unet_size=args.unet_size,
        dropout=args.dropout,
        seed=args.seed,
        pred_stride=args.pred_stride,
        pred_batch_size=args.pred_batch_size,
        mixed_precision=args.mixed_precision,
        jit_compile=args.jit_compile,
        mask_ap_iou=args.mask_ap_iou,
        det_ap_iou=args.det_ap_iou,
    )
    model_id = train(
        layout,
        args.area,
        feature_list=args.raster_type,
        true_objects=args.true_objects,
        false_objects=args.false_objects,
        model_name=args.name,
        dataset_id=args.dataset_id,
        test_area=args.test_area,
        model_config=config,
        class_names=args.class_names,
        class_weights=args.class_weights,
        class_balanced=args.balanced_sampling,
        base_model_id=args.base_model,
        augment=not args.no_augment,
    )
    print(f"model trained: {model_id}")


def _run_predict(layout: ProjectLayout, args: argparse.Namespace) -> None:
    from lidark.predict import predict

    results = predict(
        layout,
        args.areas,
        args.model_id,
        pred_type=args.pred_type,
        test_tiles=args.test_tile,
        test_area=args.test_area,
        feature_list=args.raster_type or None,
        write_tile_shape_files=args.write_tile_shapes,
        threshold=_class_spec(args.threshold, "--threshold"),
        pred_batch_size=args.batch_size,
        pred_stride=args.stride,
    )
    for area, merged in results.items():
        path = layout.predictions_dir(args.model_id, area) / f"merged_shapes_{args.pred_type}.gpkg"
        print(f"{area}: {len(merged)} detections (merged shapes: {path})")


def _run_evaluate(layout: ProjectLayout, args: argparse.Namespace) -> None:
    from lidark.evaluate import evaluate

    report = evaluate(
        layout,
        args.true_objects,
        args.model_id,
        args.eval_name,
        false_objects=args.false_objects,
        pred_type=args.pred_type,
        eval_aoi=args.eval_aoi,
        size_cut=_class_spec(args.size_cut, "--size-cut"),
        prob_cut=_class_spec(args.prob_cut, "--prob-cut"),
    )
    path = layout.reports_dir(args.model_id) / f"{args.eval_name}.csv"
    print(report.to_string(index=False))
    print(f"report written: {path}")


__all__ = ["build_parser", "main"]
