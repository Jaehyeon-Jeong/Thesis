#!/usr/bin/env python3
"""Smoke-check the Stage 1 shared scaffold.

This script verifies only the 1-I1 scaffold: package import, config loading,
path construction, optional output-directory creation, seed setting, and device
selection. It intentionally does not load `.dat` images or train a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_stage1_src_to_path() -> Path:
    """Add the local Stage 1 `src/` directory to `sys.path`."""

    stage_root = Path(__file__).resolve().parents[1]
    src_root = stage_root / "src"
    sys.path.insert(0, str(src_root))
    return stage_root


def parse_args(stage_root: Path) -> argparse.Namespace:
    """Parse CLI arguments for the scaffold smoke check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=stage_root / "configs" / "env_local.yaml",
        help="Stage 1 environment config path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to apply for the current smoke-check process.",
    )
    parser.add_argument(
        "--create-output-dirs",
        action="store_true",
        help="Create configured output directories if they do not exist.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the scaffold check and print a compact JSON summary."""

    stage_root = add_stage1_src_to_path()
    args = parse_args(stage_root)

    from stage1_reimage import (  # pylint: disable=import-outside-toplevel
        build_stage1_paths,
        ensure_stage1_output_dirs,
        load_config,
        select_device,
        set_global_seed,
    )

    config = load_config(args.config)
    paths = build_stage1_paths(config)
    created_or_verified_dirs = []
    if args.create_output_dirs:
        created_or_verified_dirs = [
            str(path) for path in ensure_stage1_output_dirs(paths)
        ]

    seed_info = set_global_seed(args.seed)
    device = select_device(config)

    summary = {
        "status": "ok",
        "stage_root": str(stage_root),
        "config_path": str(args.config),
        "environment": config["environment"]["name"],
        "selected_device": device,
        "data_root_exists": paths.data_root.exists(),
        "output_root": str(paths.output_root),
        "created_or_verified_dirs": created_or_verified_dirs,
        "seed_info": seed_info,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
