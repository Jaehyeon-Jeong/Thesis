#!/usr/bin/env python3
"""Export Stage 1 predictions and evaluation metrics.

Examples:
    Local smoke evaluation after `run_stage1_baseline.py`:
        python scripts/evaluate_stage1_predictions.py \
          --config configs/env_local.yaml \
          --horizon stage1_i20_r20 \
          --run-seed 42 \
          --split validation \
          --max-rows 4

    Kaggle full test evaluation:
        python scripts/evaluate_stage1_predictions.py \
          --config configs/env_kaggle.yaml \
          --horizon stage1_i20_r20 \
          --run-seed 42 \
          --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def add_stage1_src_to_path() -> Path:
    """Add local Stage 1 `src/` directory to `sys.path`.

    This makes local source imports work when running the script directly:
    `from stage1_reimage.evaluation import predict_loader`.
    """

    stage_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage_root / "src"))
    return stage_root


def parse_args(stage_root: Path) -> argparse.Namespace:
    """Parse CLI arguments.

    Important modes:
        normal mode loads one checkpoint and writes seed-level predictions.
        `--average-seed-predictions` skips model loading and averages existing
        prediction CSV files.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=stage_root / "configs" / "env_local.yaml",
        help="Stage 1 environment config path.",
    )
    parser.add_argument("--horizon", default="stage1_i20_r20")
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for smoke evaluation. Do not use for reproduction metrics.",
    )
    parser.add_argument(
        "--normalization-max-images",
        type=int,
        default=None,
        help="Fallback normalization cap if checkpoint metadata is unavailable.",
    )
    parser.add_argument(
        "--average-seed-predictions",
        nargs="*",
        type=int,
        default=None,
        help="Average existing seed prediction CSVs for the selected horizon/split.",
    )
    return parser.parse_args()


def main() -> int:
    """Run seed-level prediction export or average existing seed predictions.

    Seed-level path:
        checkpoint -> model -> DataLoader -> prediction DataFrame -> CSV/JSON.

    Averaging path:
        seed prediction CSVs -> mean probabilities -> averaged CSV/JSON.
    """

    stage_root = add_stage1_src_to_path()
    args = parse_args(stage_root)

    from stage1_reimage.config import load_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.data import (  # pylint: disable=import-outside-toplevel
        HORIZON_SPECS,
        TARGET_COLUMNS,
        HorizonSplitImageDataset,
        PixelNormalizationStats,
        assign_splits,
        build_base_metadata,
        build_dataset_from_config,
        build_horizon_frame,
        compute_pixel_normalization,
        normalization_settings_from_config,
        split_settings_from_config,
    )
    from stage1_reimage.evaluation import (  # pylint: disable=import-outside-toplevel
        average_seed_predictions,
        compute_classification_metrics,
        compute_correlation_metrics,
        evaluation_settings_from_config,
        load_checkpoint_into_model,
        predict_loader,
        write_evaluation_outputs,
    )
    from stage1_reimage.models import StockCNNI20  # pylint: disable=import-outside-toplevel
    from stage1_reimage.paths import (  # pylint: disable=import-outside-toplevel
        build_stage1_paths,
        ensure_stage1_output_dirs,
    )
    from stage1_reimage.runtime import select_device  # pylint: disable=import-outside-toplevel

    # Config controls data paths, device, split settings, and evaluation rule.
    config = load_config(args.config)
    paths = build_stage1_paths(config)
    ensure_stage1_output_dirs(paths)
    settings = evaluation_settings_from_config(config)

    if args.average_seed_predictions is not None:
        # This branch does not run the CNN. It only reads existing seed
        # prediction CSV files and averages their softmax probabilities.
        summary = _run_average_predictions(
            paths=paths,
            horizon=args.horizon,
            split_name=args.split,
            run_seeds=args.average_seed_predictions,
            settings=settings,
            compute_classification_metrics=compute_classification_metrics,
            compute_correlation_metrics=compute_correlation_metrics,
            write_evaluation_outputs=write_evaluation_outputs,
            average_seed_predictions=average_seed_predictions,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.horizon not in TARGET_COLUMNS:
        raise KeyError(f"Unknown horizon: {args.horizon}")

    # By default, evaluation uses the best validation-loss checkpoint created by
    # `run_stage1_baseline.py`.
    checkpoint_path = args.checkpoint_path or (
        paths.checkpoint_root / args.horizon / f"seed_{args.run_seed}" / "best.pt"
    )
    device = select_device(config)

    # Create an empty model object first, then load learned weights from the
    # checkpoint into it.
    model = StockCNNI20()
    checkpoint = load_checkpoint_into_model(model, checkpoint_path, device)

    # Training saved the exact mean/std used for image normalization. Reusing it
    # keeps evaluation data transformed exactly like validation during training.
    normalization_stats = _normalization_stats_from_checkpoint(
        checkpoint=checkpoint,
        target_return_name=TARGET_COLUMNS[args.horizon],
    )

    # Rebuild the same row index used in training so prediction rows align with
    # original Date/StockID/return metadata.
    base_dataset = build_dataset_from_config(config)
    base_metadata = build_base_metadata(base_dataset.shards)
    horizon_frame = build_horizon_frame(base_metadata, args.horizon)
    split_frame = assign_splits(horizon_frame, split_settings_from_config(config))
    if normalization_stats is None:
        normalization_stats = compute_pixel_normalization(
            dataset=base_dataset,
            split_frame=split_frame,
            settings=normalization_settings_from_config(config),
            target_return_name=TARGET_COLUMNS[args.horizon],
            max_images=args.normalization_max_images,
        )

    # Evaluation dataset returns normalized images `(1,64,60)`, labels, and
    # metadata. DataLoader stacks them into batches `(B,1,64,60)`.
    dataset = HorizonSplitImageDataset(
        base_dataset=base_dataset,
        split_frame=split_frame,
        split_name=args.split,
        normalization_stats=normalization_stats,
        max_rows=args.max_rows,
    )
    loader = _build_eval_loader(config=config, dataset=dataset, batch_size=settings.batch_size)
    horizon_spec = HORIZON_SPECS[args.horizon]
    # Run the checkpoint over the selected split and build one prediction row
    # per image.
    predictions = predict_loader(
        model=model,
        data_loader=loader,
        checkpoint_path=checkpoint_path,
        experiment_name=args.horizon,
        image_window=horizon_spec["image_window"],
        target_horizon=_target_horizon_from_name(args.horizon),
        run_seed=args.run_seed,
        split_name=args.split,
        settings=settings,
        device=device,
    )
    # Metrics are computed from the prediction DataFrame, not from the model
    # directly. This makes saved CSVs and metrics auditable.
    classification_metrics = compute_classification_metrics(predictions)
    correlation_metrics = compute_correlation_metrics(
        predictions,
        min_group_size=settings.min_correlation_group_size,
    )
    # Write CSV/JSON files under outputs/predictions and outputs/metrics.
    written = write_evaluation_outputs(
        predictions=predictions,
        classification_metrics=classification_metrics,
        correlation_metrics=correlation_metrics,
        predictions_dir=paths.predictions_root / args.horizon / f"seed_{args.run_seed}",
        metrics_dir=paths.metrics_root / args.horizon / f"seed_{args.run_seed}",
        split_name=args.split,
    )
    summary = {
        "status": "ok",
        "horizon": args.horizon,
        "split": args.split,
        "run_seed": args.run_seed,
        "checkpoint_path": str(checkpoint_path),
        "num_predictions": int(len(predictions)),
        "accuracy": classification_metrics["accuracy"],
        "positive_rate": classification_metrics["positive_rate"],
        "predicted_positive_rate": classification_metrics["predicted_positive_rate"],
        "written": written,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _build_eval_loader(
    config: dict[str, Any],
    dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> DataLoader:
    """Build a deterministic evaluation DataLoader.

    Evaluation must keep row order stable because prediction CSV rows should
    align with metadata and later seed averaging.
    """

    from stage1_reimage.config import get_config_section  # pylint: disable=import-outside-toplevel

    runtime_config = get_config_section(config, "runtime")
    num_workers = int(runtime_config.get("num_workers", 0))
    persistent_workers = bool(runtime_config.get("persistent_workers", False)) and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=bool(runtime_config.get("pin_memory", False)),
        persistent_workers=persistent_workers,
    )


def _normalization_stats_from_checkpoint(
    checkpoint: dict[str, Any],
    target_return_name: str,
) -> Any:
    """Restore normalization stats from checkpoint metadata when available.

    Output:
        `PixelNormalizationStats` used by `HorizonSplitImageDataset`, or `None`
        if an older checkpoint did not store normalization metadata.
    """

    from stage1_reimage.data import PixelNormalizationStats  # pylint: disable=import-outside-toplevel

    metadata = checkpoint.get("normalization_metadata") or {}
    required = [
        "train_pixel_mean",
        "train_pixel_std",
        "pixel_scale",
        "epsilon",
        "num_train_images_available",
        "num_train_images_used",
        "num_pixels_used",
        "sampled_for_smoke",
    ]
    if not all(key in metadata for key in required):
        return None
    return PixelNormalizationStats(
        target_return_name=str(metadata.get("target_return_name", target_return_name)),
        train_pixel_mean=float(metadata["train_pixel_mean"]),
        train_pixel_std=float(metadata["train_pixel_std"]),
        pixel_scale=float(metadata["pixel_scale"]),
        epsilon=float(metadata["epsilon"]),
        num_train_images_available=int(metadata["num_train_images_available"]),
        num_train_images_used=int(metadata["num_train_images_used"]),
        num_pixels_used=int(metadata["num_pixels_used"]),
        sampled_for_smoke=bool(metadata["sampled_for_smoke"]),
    )


def _target_horizon_from_name(horizon_name: str) -> str:
    """Map `stage1_i20_r20` to `R20` for prediction metadata."""

    return horizon_name.rsplit("_", maxsplit=1)[-1].upper()


def _run_average_predictions(
    paths: Any,
    horizon: str,
    split_name: str,
    run_seeds: list[int],
    settings: Any,
    compute_classification_metrics: Any,
    compute_correlation_metrics: Any,
    write_evaluation_outputs: Any,
    average_seed_predictions: Any,
) -> dict[str, Any]:
    """Average already-written seed prediction files.

    This is used after five independent training runs. It checks that all seed
    files describe the same rows, then averages `prob_up` and recomputes metrics.
    """

    # Expected file pattern:
    # outputs/predictions/<horizon>/seed_<seed>/<split>_predictions.csv
    prediction_paths = [
        paths.predictions_root / horizon / f"seed_{run_seed}" / f"{split_name}_predictions.csv"
        for run_seed in run_seeds
    ]
    missing = [str(path) for path in prediction_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing seed prediction file(s): {missing}")

    averaged = average_seed_predictions(
        prediction_paths=prediction_paths,
        run_seeds=run_seeds,
        settings=settings,
    )
    classification_metrics = compute_classification_metrics(
        averaged,
        probability_column="mean_prob_up",
    )
    correlation_metrics = compute_correlation_metrics(
        averaged,
        probability_column="mean_prob_up",
        min_group_size=settings.min_correlation_group_size,
    )
    written = write_evaluation_outputs(
        predictions=averaged,
        classification_metrics=classification_metrics,
        correlation_metrics=correlation_metrics,
        predictions_dir=paths.predictions_root / horizon / "averaged",
        metrics_dir=paths.metrics_root / horizon / "averaged",
        split_name=split_name,
    )
    return {
        "status": "ok",
        "mode": "average_seed_predictions",
        "horizon": horizon,
        "split": split_name,
        "run_seeds": run_seeds,
        "num_predictions": int(len(averaged)),
        "accuracy": classification_metrics["accuracy"],
        "positive_rate": classification_metrics["positive_rate"],
        "predicted_positive_rate": classification_metrics["predicted_positive_rate"],
        "written": written,
    }


if __name__ == "__main__":
    raise SystemExit(main())
