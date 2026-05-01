#!/usr/bin/env python
"""2-I6: Export Stage 2 BTC predictions and classification metrics."""

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
    from stage2_btc import (
        build_stage2_paths,
        experiment_output_roots,
        load_config,
        make_experiment_name,
        select_device,
    )
    from stage2_btc.evaluation import compute_classification_metrics, predict_loader, write_json
    from stage2_btc.evaluation.prediction import load_checkpoint_into_model
    from stage2_btc.models import build_stock_cnn_for_window
    from stage2_btc.runners.btc_baseline import build_dataloaders, prepare_btc_experiment_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=20)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    paths = build_stage2_paths(config)
    device = select_device(config)
    experiment_name = make_experiment_name(args.image_window, args.image_spec, args.return_horizon)
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
    loaders = build_dataloaders(prepared.datasets, config, shuffle_train=False)
    output_roots = experiment_output_roots(paths, experiment_name, args.run_seed)
    checkpoint_path = output_roots["checkpoint"] / "best.pt"
    model = build_stock_cnn_for_window(config, args.image_window)
    load_checkpoint_into_model(model, checkpoint_path, device)
    run_context = {
        "experiment_name": experiment_name,
        "image_window": int(args.image_window),
        "image_spec": str(args.image_spec),
        "return_horizon": int(args.return_horizon),
        "run_seed": int(args.run_seed),
    }
    predictions = predict_loader(
        model,
        loaders[args.split],
        config,
        device=device,
        run_context=run_context,
        checkpoint_path=checkpoint_path,
        split_name=args.split,
    )
    prediction_path = output_roots["predictions"] / f"{args.split}_predictions.csv"
    metrics_path = output_roots["metrics"] / f"{args.split}_metrics.json"
    output_roots["predictions"].mkdir(parents=True, exist_ok=True)
    output_roots["metrics"].mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    metrics = compute_classification_metrics(predictions)
    write_json(metrics_path, metrics)
    result = {
        "status": "ok",
        "predictions": str(prediction_path),
        "metrics": str(metrics_path),
        "num_predictions": int(len(predictions)),
        "accuracy": metrics["accuracy"],
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
