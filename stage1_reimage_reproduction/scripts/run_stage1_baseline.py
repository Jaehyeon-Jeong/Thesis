#!/usr/bin/env python3
"""Run Stage 1 baseline training from local or Kaggle config.

Examples:
    Local smoke:
        python scripts/run_stage1_baseline.py \
          --config configs/env_local.yaml \
          --run-mode smoke \
          --horizons stage1_i20_r20 \
          --max-train-rows 8 \
          --max-val-rows 4 \
          --normalization-max-images 128 \
          --max-epochs 1

    Kaggle single-seed:
        python scripts/run_stage1_baseline.py \
          --config configs/env_kaggle.yaml \
          --run-mode full_single_seed
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
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=stage_root / "configs" / "env_local.yaml",
        help="Stage 1 environment config path.",
    )
    parser.add_argument(
        "--run-mode",
        choices=["smoke", "full_single_seed", "full_paper_style"],
        default=None,
        help="Run mode. Defaults to config run.default_run_mode.",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        default=None,
        help="Horizon names. Defaults to stage1_i20_r20 for smoke, all horizons for full modes.",
    )
    parser.add_argument("--run-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--normalization-max-images", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--allow-local-full",
        action="store_true",
        help="Allow non-smoke mode when config environment.full_run_target is false.",
    )
    return parser.parse_args()


def main() -> int:
    """Run Stage 1 baseline runner and print a JSON summary."""

    stage_root = add_stage1_src_to_path()
    args = parse_args(stage_root)

    from stage1_reimage.config import load_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.data import TARGET_COLUMNS  # pylint: disable=import-outside-toplevel
    from stage1_reimage.paths import build_stage1_paths  # pylint: disable=import-outside-toplevel
    from stage1_reimage.runners import (  # pylint: disable=import-outside-toplevel
        RunSelection,
        run_stage1_baseline,
    )

    config = load_config(args.config)
    paths = build_stage1_paths(config)
    run_config = config["run"]
    run_mode = args.run_mode or str(run_config["default_run_mode"])
    horizons = tuple(args.horizons or _default_horizons(run_mode, TARGET_COLUMNS))
    run_seeds = tuple(args.run_seeds or _default_run_seeds(run_mode, run_config))

    selection = RunSelection(
        run_mode=run_mode,
        horizons=horizons,
        run_seeds=run_seeds,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        normalization_max_images=args.normalization_max_images,
        max_epochs=args.max_epochs,
        allow_local_full=bool(args.allow_local_full),
    )
    summary = run_stage1_baseline(config=config, paths=paths, selection=selection)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _default_horizons(run_mode: str, target_columns: dict[str, str]) -> list[str]:
    """Return safe default horizons for the selected mode."""

    if run_mode == "smoke":
        return ["stage1_i20_r20"]
    return list(target_columns)


def _default_run_seeds(run_mode: str, run_config: dict[str, object]) -> list[int]:
    """Return run seeds from config for the selected mode."""

    key = {
        "smoke": "smoke_run_seeds",
        "full_single_seed": "full_single_seed_run_seeds",
        "full_paper_style": "full_paper_style_run_seeds",
    }[run_mode]
    return [int(value) for value in run_config[key]]  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
