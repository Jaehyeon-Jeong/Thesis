#!/usr/bin/env python3
"""Smoke-check Stage 1 monthly_20d data loading.

This validates shard discovery, row alignment, memmap-backed image loading, and
sample tensor shape. It does not construct horizon labels, splits, normalization
statistics, model inputs beyond the image tensor, or training dataloaders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def add_stage1_src_to_path() -> Path:
    """Add local Stage 1 `src/` directory to `sys.path`.

    This lets the smoke script import local Stage 1 modules without installing
    the package into the Python environment.
    """

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
        "--sample-indices",
        nargs="*",
        type=int,
        default=[0, -1],
        help="Global row indices to inspect after building the dataset.",
    )
    return parser.parse_args()


def summarize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe summary of one dataset sample.

    Input sample format:
        `{"image": tensor(1,64,60), "metadata": {...}}`.
    The printed summary shows shape and min/max pixel values without dumping the
    whole image tensor.
    """

    image = sample["image"]
    metadata = sample["metadata"]
    return {
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "image_min": float(image.min().item()),
        "image_max": float(image.max().item()),
        "year": metadata["year"],
        "local_row": metadata["local_row"],
        "date": metadata["Date"],
        "stock_id": metadata["StockID"],
        "metadata_columns": sorted(metadata.keys()),
    }


def main() -> int:
    """Run the data-loading smoke check and print a JSON summary."""

    add_stage1_src_to_path()
    args = parse_args(Path(__file__).resolve().parents[1])

    from stage1_reimage.config import load_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.data import (  # pylint: disable=import-outside-toplevel
        build_dataset_from_config,
    )

    # Build the memmap-backed dataset. At this point images are still read
    # lazily from disk; only label metadata is loaded into memory.
    config = load_config(args.config)
    dataset = build_dataset_from_config(config)
    inspected_samples = []
    for index in args.sample_indices:
        # Accessing `dataset[index]` reads exactly one image row and returns a
        # tensor `(1,64,60)` plus metadata for that row.
        inspected_samples.append(
            {
                "requested_index": index,
                "sample": summarize_sample(dataset[index]),
            }
        )

    summary = {
        "status": "ok",
        "config_path": str(args.config),
        "num_shards": len(dataset.shards),
        "num_rows": len(dataset),
        "first_shard": dataset.shards[0].as_dict(),
        "last_shard": dataset.shards[-1].as_dict(),
        "inspected_samples": inspected_samples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
