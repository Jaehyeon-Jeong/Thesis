#!/usr/bin/env python
"""Export Stage 4 predictions and classification metrics."""

from __future__ import annotations

import argparse
import json
import sys

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from stage2_btc.evaluation import compute_classification_metrics, write_json
from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import (
    CONTEXT_METHODS,
    get_stage4_model_config,
    stage4_run_context_base,
    validate_context_method,
)
from stage4_film.evaluation import load_stage4_checkpoint_into_model, predict_context_loader
from stage4_film.paths import experiment_output_roots
from stage4_film.runners import (
    build_stage4_context_dataloaders,
    build_stage4_context_model,
    prepare_stage4_context_experiment_data,
)
from stage4_film.runtime import select_device


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=0)
    parser.add_argument("--image-spec", default="")
    parser.add_argument("--return-horizon", type=int, default=0)
    parser.add_argument(
        "--context-method",
        default="concat",
        choices=list(CONTEXT_METHODS),
    )
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run Stage 4 prediction and classification metric export."""

    args = parse_args()
    config = load_config(args.config)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    stage4_model = get_stage4_model_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
    image_spec = str(args.image_spec or stage4_model["primary_image_spec"])
    return_horizon = int(args.return_horizon or stage4_model["primary_return_horizon"])
    context_method = validate_context_method(args.context_method)
    device = select_device(config)

    run_context = stage4_run_context_base(
        config,
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        context_method=context_method,
        run_seed=int(args.run_seed),
    )
    experiment_name = str(run_context["stage4_experiment_name"])
    run_context["experiment_name"] = experiment_name
    output_roots = experiment_output_roots(paths, experiment_name, int(args.run_seed))
    checkpoint_path = output_roots["checkpoint"] / "best.pt"

    prepared = prepare_stage4_context_experiment_data(
        config=config,
        paths=paths,
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        run_seed=int(args.run_seed),
        max_train_rows=_optional_positive_int(args.max_train_rows),
        max_validation_rows=_optional_positive_int(args.max_validation_rows),
        max_test_rows=_optional_positive_int(args.max_test_rows),
    )
    loaders = build_stage4_context_dataloaders(prepared.datasets, config, shuffle_train=False)
    model = build_stage4_context_model(
        config=config,
        image_window=image_window,
        context_method=context_method,
    )
    checkpoint = load_stage4_checkpoint_into_model(model, checkpoint_path, device)
    context_feature_names = [
        f"{feature}_normalized" for feature in prepared.context_scaler.feature_order
    ]
    predictions = predict_context_loader(
        model,
        loaders[args.split],
        config,
        device=device,
        run_context=run_context,
        checkpoint_path=checkpoint_path,
        split_name=args.split,
        context_feature_names=context_feature_names,
    )

    output_roots["predictions"].mkdir(parents=True, exist_ok=True)
    output_roots["metrics"].mkdir(parents=True, exist_ok=True)
    prediction_path = output_roots["predictions"] / f"{args.split}_predictions.csv"
    metrics_path = output_roots["metrics"] / f"{args.split}_metrics.json"
    predictions.to_csv(prediction_path, index=False)
    metrics = compute_classification_metrics(predictions)
    write_json(metrics_path, metrics)

    print(
        json.dumps(
            {
                "status": "ok",
                "experiment_name": experiment_name,
                "context_method": context_method,
                "split": args.split,
                "run_seed": int(args.run_seed),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "num_predictions": int(len(predictions)),
                "accuracy": metrics["accuracy"],
                "positive_rate": metrics["positive_rate"],
                "predicted_positive_rate": metrics["predicted_positive_rate"],
                "written": {
                    "predictions": str(prediction_path),
                    "classification_metrics": str(metrics_path),
                },
            },
            indent=2,
        )
    )


def _optional_positive_int(value: int) -> int | None:
    """Convert CLI row limit values to optional positive integers."""

    integer = int(value)
    return integer if integer > 0 else None


if __name__ == "__main__":
    main()
