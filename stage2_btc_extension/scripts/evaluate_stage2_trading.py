#!/usr/bin/env python
"""2-I7: Export Stage 2 BTC trading metrics."""

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
    from stage2_btc import build_stage2_paths, experiment_output_roots, load_config, make_experiment_name
    from stage2_btc.evaluation import compute_trading_metrics, write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=20)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    paths = build_stage2_paths(config)
    experiment_name = make_experiment_name(args.image_window, args.image_spec, args.return_horizon)
    output_roots = experiment_output_roots(paths, experiment_name, args.run_seed)
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
    print(json.dumps({"status": "ok", "trading_metrics": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
