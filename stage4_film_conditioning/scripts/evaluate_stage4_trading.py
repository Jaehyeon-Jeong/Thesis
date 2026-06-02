#!/usr/bin/env python
"""Export Stage 4 BTC trading metrics from prediction CSV."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from stage2_btc.evaluation import compute_trading_metrics, write_json
from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import (
    CONTEXT_METHODS,
    get_stage4_model_config,
    stage4_run_context_base,
    validate_context_method,
)
from stage4_film.paths import experiment_output_roots


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
    return parser.parse_args()


def main() -> None:
    """Compute Stage 4 trading metrics."""

    args = parse_args()
    config = load_config(args.config)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    stage4_model = get_stage4_model_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
    image_spec = str(args.image_spec or stage4_model["primary_image_spec"])
    return_horizon = int(args.return_horizon or stage4_model["primary_return_horizon"])
    context_method = validate_context_method(args.context_method)
    run_context = stage4_run_context_base(
        config,
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        context_method=context_method,
        run_seed=int(args.run_seed),
    )
    experiment_name = str(run_context["stage4_experiment_name"])
    output_roots = experiment_output_roots(paths, experiment_name, int(args.run_seed))
    prediction_path = output_roots["predictions"] / f"{args.split}_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_path}")

    predictions = pd.read_csv(prediction_path)
    trading_config = config["trading"]
    metrics = compute_trading_metrics(
        predictions,
        annualization_periods=int(trading_config.get("annualization_periods", 365)),
        transaction_cost_bps=float(trading_config.get("transaction_cost_bps", 10.0)),
    )
    output_path = output_roots["metrics"] / f"{args.split}_trading_metrics.json"
    write_json(output_path, metrics)
    print(
        json.dumps(
            {
                "status": "ok",
                "experiment_name": experiment_name,
                "context_method": context_method,
                "split": args.split,
                "run_seed": int(args.run_seed),
                "prediction_path": str(prediction_path),
                "trading_metrics": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
