#!/usr/bin/env python
"""2-I9/2-I10: Check required Stage 2 output files exist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_src_to_path() -> Path:
    stage_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage_root / "src"))
    return stage_root


def main() -> None:
    stage_root = add_src_to_path()
    from stage2_btc import build_stage2_paths, experiment_output_roots, load_config, make_experiment_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=20)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--gradcam-samples-per-class", type=int, default=2)
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    paths = build_stage2_paths(config)
    experiment_name = make_experiment_name(args.image_window, args.image_spec, args.return_horizon)
    output_roots = experiment_output_roots(paths, experiment_name, args.run_seed)
    checks = {
        "best_checkpoint": output_roots["checkpoint"] / "best.pt",
        "train_history": output_roots["metrics"] / "train_history.csv",
        "train_metadata": output_roots["metrics"] / "train_metadata.json",
        "predictions": output_roots["predictions"] / f"{args.split}_predictions.csv",
        "classification_metrics": output_roots["metrics"] / f"{args.split}_metrics.json",
        "trading_metrics": output_roots["metrics"] / f"{args.split}_trading_metrics.json",
        "gradcam": output_roots["figures"] / "gradcam" / args.split / f"btc_gradcam_{args.split}_{args.gradcam_samples_per_class}perclass.png",
    }
    result = {name: {"path": str(path), "exists": path.exists()} for name, path in checks.items()}
    status = "ok" if all(item["exists"] for item in result.values()) else "missing"
    print(json.dumps({"status": status, "checks": result}, indent=2))
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
