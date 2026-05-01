#!/usr/bin/env python
"""3-I4: Export Stage 3 predictions and classification metrics."""

from __future__ import annotations

import argparse
import json
import sys

from _stage3_script_utils import add_stage3_and_stage2_src_from_argv


def main() -> None:
    """학습된 Stage 3 checkpoint를 prediction CSV와 metric JSON으로 변환한다."""

    stage_root = add_stage3_and_stage2_src_from_argv(sys.argv)
    from stage2_btc import build_stage2_paths, experiment_output_roots, load_config, select_device
    from stage2_btc.evaluation import compute_classification_metrics, predict_loader, write_json
    from stage2_btc.evaluation.prediction import load_checkpoint_into_model
    from stage2_btc.runners.btc_baseline import build_dataloaders, prepare_btc_experiment_data
    from stage3_linear.config import stage3_run_context_base
    from stage3_linear.models import build_linear_stock_cnn_for_window

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=60)
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
    run_context = stage3_run_context_base(
        config,
        args.image_window,
        args.image_spec,
        args.return_horizon,
        args.run_seed,
    )
    experiment_name = run_context["stage3_experiment_name"]
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
    model = build_linear_stock_cnn_for_window(config, args.image_window)
    load_checkpoint_into_model(model, checkpoint_path, device)
    prediction_context = {
        "experiment_name": experiment_name,
        "image_window": int(args.image_window),
        "image_spec": str(args.image_spec),
        "return_horizon": int(args.return_horizon),
        "run_seed": int(args.run_seed),
        "stage": "stage3",
        "adapter_dim": int(run_context["adapter_dim"]),
    }
    predictions = predict_loader(
        model,
        loaders[args.split],
        config,
        device=device,
        run_context=prediction_context,
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

