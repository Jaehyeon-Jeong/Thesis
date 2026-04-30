#!/usr/bin/env python3
"""Smoke-check Stage 1 label, split, and normalization implementation.

This script verifies the data-preparation steps before model training:
future-return labels, deterministic split assignment, and train-only pixel
normalization. It writes small JSON audit outputs under `outputs/metrics/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_stage1_src_to_path() -> Path:
    """Add local Stage 1 `src/` directory to `sys.path`."""

    stage_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage_root / "src"))
    return stage_root


def parse_args(stage_root: Path) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=stage_root / "configs" / "env_local.yaml",
        help="Stage 1 environment config path.",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        default=["stage1_i20_r5", "stage1_i20_r20", "stage1_i20_r60"],
        help="Horizon names to check.",
    )
    parser.add_argument(
        "--normalization-max-images",
        type=int,
        default=2048,
        help=(
            "Maximum training images for local normalization smoke check. "
            "Use 0 to compute on all training images."
        ),
    )
    parser.add_argument(
        "--normalization-chunk-size",
        type=int,
        default=4096,
        help="Images per memmap chunk while computing pixel statistics.",
    )
    parser.add_argument(
        "--write-split-index",
        action="store_true",
        help="Write split_index.csv in addition to JSON summaries.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the label/split/normalization smoke check.

    Data path:
        monthly_20d shards -> base metadata DataFrame -> horizon labels ->
        split frame -> train-only normalization statistics.
    """

    add_stage1_src_to_path()
    args = parse_args(Path(__file__).resolve().parents[1])

    from stage1_reimage.config import load_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.data import (  # pylint: disable=import-outside-toplevel
        TARGET_COLUMNS,
        assign_splits,
        build_base_metadata,
        build_dataset_from_config,
        build_horizon_frame,
        compute_pixel_normalization,
        make_split_summary,
        normalization_settings_from_config,
        split_settings_from_config,
        write_horizon_metadata,
    )
    from stage1_reimage.paths import (  # pylint: disable=import-outside-toplevel
        build_stage1_paths,
        ensure_stage1_output_dirs,
    )

    config = load_config(args.config)
    paths = build_stage1_paths(config)
    ensure_stage1_output_dirs(paths)
    # Dataset reads images; base_metadata holds Date/StockID/future returns and
    # row ids. No CNN tensor batch is built in this script.
    dataset = build_dataset_from_config(config)
    base_metadata = build_base_metadata(dataset.shards)
    split_settings = split_settings_from_config(config)
    normalization_settings = normalization_settings_from_config(config)
    max_images = None if args.normalization_max_images <= 0 else args.normalization_max_images

    horizon_results = {}
    for horizon_name in args.horizons:
        if horizon_name not in TARGET_COLUMNS:
            raise KeyError(f"Unknown horizon: {horizon_name}")

        # Build labels for one target horizon, e.g. Ret_20d -> label.
        horizon_frame = build_horizon_frame(base_metadata, horizon_name)
        split_frame = assign_splits(horizon_frame, split_settings)
        split_summary = make_split_summary(split_frame, split_settings, horizon_name)

        # This computes scalar train mean/std from image pixels. The smoke cap
        # keeps local execution small.
        normalization_stats = compute_pixel_normalization(
            dataset=dataset,
            split_frame=split_frame,
            settings=normalization_settings,
            target_return_name=TARGET_COLUMNS[horizon_name],
            max_images=max_images,
            chunk_size=args.normalization_chunk_size,
        )

        output_dir = paths.metrics_root / horizon_name
        written_files = write_horizon_metadata(
            output_dir=output_dir,
            split_summary=split_summary,
            normalization_stats=normalization_stats,
            split_frame=split_frame,
            write_split_index=args.write_split_index,
        )
        horizon_results[horizon_name] = {
            "target_return_name": TARGET_COLUMNS[horizon_name],
            "split_summary": split_summary,
            "normalization": normalization_stats.as_dict(),
            "written_files": written_files,
        }

    summary = {
        "status": "ok",
        "config_path": str(args.config),
        "num_base_rows": int(len(base_metadata)),
        "normalization_max_images": max_images,
        "horizon_results": horizon_results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
