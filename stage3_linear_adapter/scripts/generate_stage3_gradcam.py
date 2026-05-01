#!/usr/bin/env python
"""3-I5: Generate Stage 3 Linear Grad-CAM and optional Stage2-vs-Stage3 comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from _stage3_script_utils import add_stage3_and_stage2_src_from_argv


def main() -> None:
    """Stage 3 Grad-CAM figure와, 가능하면 Stage 2 baseline 비교 figure를 만든다."""

    stage_root = add_stage3_and_stage2_src_from_argv(sys.argv)
    from stage2_btc import build_stage2_paths, experiment_output_roots, load_config, select_device
    from stage2_btc.evaluation.prediction import load_checkpoint_into_model
    from stage2_btc.interpretability import generate_gradcam_figure
    from stage2_btc.models import build_stock_cnn_for_window
    from stage2_btc.runners.btc_baseline import prepare_btc_experiment_data
    from stage3_linear.config import (
        get_stage2_dependency_config,
        make_stage2_baseline_experiment_name,
        stage3_run_context_base,
    )
    from stage3_linear.interpretability import generate_baseline_linear_comparison_figure
    from stage3_linear.models import build_linear_stock_cnn_for_window

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=60)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--write-report-copy", action="store_true")
    parser.add_argument("--skip-baseline-comparison", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    paths = build_stage2_paths(config)
    device = select_device(config)
    run_context = stage3_run_context_base(
        config,
        args.image_window,
        args.image_spec,
        args.return_horizon,
        args.run_seed,
    )
    experiment_name = run_context["stage3_experiment_name"]
    output_roots = experiment_output_roots(paths, experiment_name, args.run_seed)
    prediction_path = output_roots["predictions"] / f"{args.split}_predictions.csv"
    checkpoint_path = output_roots["checkpoint"] / "best.pt"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Stage 3 prediction CSV not found: {prediction_path}")

    predictions = pd.read_csv(prediction_path)
    prepared = prepare_btc_experiment_data(
        config,
        paths,
        args.image_window,
        args.image_spec,
        args.return_horizon,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        max_test_rows=args.max_test_rows,
    )
    linear_model = build_linear_stock_cnn_for_window(config, args.image_window)
    load_checkpoint_into_model(linear_model, checkpoint_path, device)

    figure_path = (
        output_roots["figures"]
        / "gradcam"
        / args.split
        / f"btc_linear_gradcam_{args.split}_{args.samples_per_class}perclass.png"
    )
    written_figure, selected = generate_gradcam_figure(
        linear_model,
        prepared.datasets[args.split],
        predictions,
        figure_path,
        samples_per_class=args.samples_per_class,
        device=device,
    )
    samples_path = written_figure.parent / "samples.csv"
    selected.to_csv(samples_path, index=False)
    result = {
        "status": "ok",
        "linear_figure": str(written_figure),
        "linear_samples": str(samples_path),
        "baseline_comparison": "skipped",
    }

    if args.write_report_copy:
        report_dir = paths.reports_root / "figures" / "gradcam"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_copy = report_dir / f"{experiment_name}_seed_{args.run_seed}_{args.split}_gradcam.png"
        report_copy.write_bytes(written_figure.read_bytes())
        result["report_copy"] = str(report_copy)

    if not args.skip_baseline_comparison:
        comparison = _try_write_baseline_comparison(
            config=config,
            linear_model=linear_model,
            dataset=prepared.datasets[args.split],
            linear_predictions=predictions,
            image_window=args.image_window,
            image_spec=args.image_spec,
            return_horizon=args.return_horizon,
            run_seed=args.run_seed,
            split=args.split,
            samples_per_class=args.samples_per_class,
            output_roots=output_roots,
            device=device,
        )
        result["baseline_comparison"] = comparison

    print(json.dumps(result, indent=2, default=str))


def _try_write_baseline_comparison(
    config,
    linear_model,
    dataset,
    linear_predictions: pd.DataFrame,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    run_seed: int,
    split: str,
    samples_per_class: int,
    output_roots,
    device,
) -> dict[str, str]:
    """Stage 2 baseline output이 있으면 같은 sample Grad-CAM 비교를 저장한다."""

    from stage2_btc.evaluation.prediction import load_checkpoint_into_model
    from stage2_btc.models import build_stock_cnn_for_window

    from stage3_linear.config import get_stage2_dependency_config, make_stage2_baseline_experiment_name
    from stage3_linear.interpretability import generate_baseline_linear_comparison_figure

    dependency = get_stage2_dependency_config(config)
    baseline_root = Path(str(dependency.get("baseline_output_root", ""))).expanduser()
    baseline_experiment = make_stage2_baseline_experiment_name(
        image_window,
        image_spec,
        return_horizon,
    )
    seed_name = f"seed_{int(run_seed)}"
    baseline_checkpoint = baseline_root / "checkpoints" / baseline_experiment / seed_name / "best.pt"
    baseline_predictions_path = (
        baseline_root
        / "predictions"
        / baseline_experiment
        / seed_name
        / f"{split}_predictions.csv"
    )
    if not baseline_checkpoint.exists() or not baseline_predictions_path.exists():
        return {
            "status": "missing_stage2_outputs",
            "baseline_checkpoint": str(baseline_checkpoint),
            "baseline_predictions": str(baseline_predictions_path),
        }

    baseline_model = build_stock_cnn_for_window(config, image_window)
    load_checkpoint_into_model(baseline_model, baseline_checkpoint, device)
    baseline_predictions = pd.read_csv(baseline_predictions_path)
    comparison_path = (
        output_roots["figures"]
        / "gradcam_comparison"
        / split
        / f"stage2_vs_stage3_linear_{split}_{samples_per_class}perclass.png"
    )
    comparison_figure, comparison_samples = generate_baseline_linear_comparison_figure(
        baseline_model,
        linear_model,
        dataset,
        linear_predictions,
        baseline_predictions,
        comparison_path,
        samples_per_class=samples_per_class,
        device=device,
    )
    comparison_samples_path = comparison_figure.parent / "comparison_samples.csv"
    comparison_samples.to_csv(comparison_samples_path, index=False)
    return {
        "status": "ok",
        "figure": str(comparison_figure),
        "samples": str(comparison_samples_path),
    }


if __name__ == "__main__":
    main()

