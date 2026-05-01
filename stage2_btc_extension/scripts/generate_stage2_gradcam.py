#!/usr/bin/env python
"""2-I8: Generate BTC Grad-CAM figure for one Stage 2 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def add_src_to_path() -> Path:
    stage_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage_root / "src"))
    return stage_root


def main() -> None:
    stage_root = add_src_to_path()
    from stage2_btc import (
        build_stage2_paths,
        experiment_output_roots,
        load_config,
        make_experiment_name,
        select_device,
    )
    from stage2_btc.evaluation.prediction import load_checkpoint_into_model
    from stage2_btc.interpretability import generate_gradcam_figure
    from stage2_btc.models import build_stock_cnn_for_window
    from stage2_btc.runners.btc_baseline import prepare_btc_experiment_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=20)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--write-report-copy", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    paths = build_stage2_paths(config)
    device = select_device(config)
    experiment_name = make_experiment_name(args.image_window, args.image_spec, args.return_horizon)
    output_roots = experiment_output_roots(paths, experiment_name, args.run_seed)
    prediction_path = output_roots["predictions"] / f"{args.split}_predictions.csv"
    checkpoint_path = output_roots["checkpoint"] / "best.pt"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_path}")
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
    model = build_stock_cnn_for_window(config, args.image_window)
    load_checkpoint_into_model(model, checkpoint_path, device)
    figure_path = (
        output_roots["figures"]
        / "gradcam"
        / args.split
        / f"btc_gradcam_{args.split}_{args.samples_per_class}perclass.png"
    )
    written_figure, selected = generate_gradcam_figure(
        model,
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
        "figure": str(written_figure),
        "samples": str(samples_path),
    }
    if args.write_report_copy:
        report_dir = paths.reports_root / "figures" / "gradcam"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_copy = report_dir / f"{experiment_name}_seed_{args.run_seed}_{args.split}_gradcam.png"
        report_copy.write_bytes(written_figure.read_bytes())
        result["report_copy"] = str(report_copy)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
