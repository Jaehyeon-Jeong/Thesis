#!/usr/bin/env python
"""3-I6/3-I7: Check required Stage 3 output files exist."""

from __future__ import annotations

import argparse
import json
import sys

from _stage3_script_utils import add_stage3_and_stage2_src_from_argv


def main() -> None:
    """한 Stage 3 experiment/seed의 checkpoint, metrics, Grad-CAM 존재를 확인한다."""

    stage_root = add_stage3_and_stage2_src_from_argv(sys.argv)
    from stage2_btc import build_stage2_paths, experiment_output_roots, load_config
    from stage3_linear.config import stage3_run_context_base

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=60)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--gradcam-samples-per-class", type=int, default=2)
    parser.add_argument("--require-comparison-gradcam", action="store_true")
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    paths = build_stage2_paths(config)
    run_context = stage3_run_context_base(
        config,
        args.image_window,
        args.image_spec,
        args.return_horizon,
        args.run_seed,
    )
    experiment_name = run_context["stage3_experiment_name"]
    output_roots = experiment_output_roots(paths, experiment_name, args.run_seed)
    checks = {
        "best_checkpoint": output_roots["checkpoint"] / "best.pt",
        "train_history": output_roots["metrics"] / "train_history.csv",
        "train_metadata": output_roots["metrics"] / "train_metadata.json",
        "predictions": output_roots["predictions"] / f"{args.split}_predictions.csv",
        "classification_metrics": output_roots["metrics"] / f"{args.split}_metrics.json",
        "trading_metrics": output_roots["metrics"] / f"{args.split}_trading_metrics.json",
        "linear_gradcam": (
            output_roots["figures"]
            / "gradcam"
            / args.split
            / f"btc_linear_gradcam_{args.split}_{args.gradcam_samples_per_class}perclass.png"
        ),
    }
    if args.require_comparison_gradcam:
        checks["comparison_gradcam"] = (
            output_roots["figures"]
            / "gradcam_comparison"
            / args.split
            / f"stage2_vs_stage3_linear_{args.split}_{args.gradcam_samples_per_class}perclass.png"
        )
    result = {name: {"path": str(path), "exists": path.exists()} for name, path in checks.items()}
    status = "ok" if all(item["exists"] for item in result.values()) else "missing"
    print(json.dumps({"status": status, "experiment_name": experiment_name, "checks": result}, indent=2))
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

