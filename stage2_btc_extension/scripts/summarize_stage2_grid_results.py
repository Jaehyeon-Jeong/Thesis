#!/usr/bin/env python
"""Summarize Stage 2 BTC grid outputs into CSV tables.

역할:
    Stage 2 grid run이 만든 classification metric JSON과 trading metric JSON을
    읽어서 seed-level table과 mean/std table을 만든다.

지원 위치:
    1. 현재 `/kaggle/working/stage2_btc_extension/outputs/stage2` output tree
    2. `/kaggle/working/stage2_saved_outputs`에 저장된 backup zip

이 script는 모델을 다시 실행하지 않는다. 이미 저장된 결과만 읽는다.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


IMAGE_WINDOWS = (5, 20, 60)
RETURN_HORIZONS = (5, 20, 60)
IMAGE_SPECS = ("ohlc", "ohlc_vb", "ohlc_ma", "ohlc_ma_vb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_kaggle.yaml")
    parser.add_argument("--image-windows", nargs="+", type=int, default=list(IMAGE_WINDOWS))
    parser.add_argument("--return-horizons", nargs="+", type=int, default=list(RETURN_HORIZONS))
    parser.add_argument("--image-specs", nargs="+", default=list(IMAGE_SPECS))
    parser.add_argument("--run-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split", default="test")
    parser.add_argument("--backup-root", default="/kaggle/working/stage2_saved_outputs")
    parser.add_argument("--include-backup-zips", action="store_true")
    parser.add_argument("--output-prefix", default="stage2_grid")
    return parser.parse_args()


def experiment_name(image_window: int, image_spec: str, return_horizon: int) -> str:
    return f"stage2_i{image_window}_{image_spec}_r{return_horizon}"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_from_zips(backup_root: Path, experiment: str, seed: int, suffix: str) -> dict[str, Any] | None:
    """backup zip들에서 suffix가 일치하는 JSON을 찾아 읽는다."""

    if not backup_root.exists():
        return None
    zip_files = sorted(backup_root.glob(f"{experiment}_seed{seed}_*_outputs.zip"), reverse=True)
    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                normalized = member.replace("\\", "/")
                if normalized.endswith(suffix):
                    with archive.open(member) as handle:
                        return json.loads(handle.read().decode("utf-8"))
    return None


def flatten_classification(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """classification metric JSON에서 table column으로 쓸 값만 꺼낸다."""

    if metrics is None:
        return {"classification_available": False}
    accuracy = metrics.get("accuracy")
    majority = metrics.get("majority_class_accuracy")
    return {
        "classification_available": True,
        "num_samples": metrics.get("num_samples"),
        "accuracy": accuracy,
        "majority_class_accuracy": majority,
        "accuracy_minus_majority": None if accuracy is None or majority is None else accuracy - majority,
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "f1": metrics.get("f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "brier_score": metrics.get("brier_score"),
        "log_loss": metrics.get("log_loss"),
        "positive_rate": metrics.get("positive_rate"),
        "predicted_positive_rate": metrics.get("predicted_positive_rate"),
        "probability_return_pearson": metrics.get("probability_return_pearson"),
    }


def flatten_trading(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """trading metric JSON에서 long/flat, long/short 핵심 값만 꺼낸다."""

    if metrics is None:
        return {"trading_available": False}
    row: dict[str, Any] = {"trading_available": True}
    for strategy_name in ("long_flat", "long_short"):
        values = metrics.get(strategy_name, {})
        prefix = strategy_name
        for key in (
            "mean_trade_return_gross",
            "mean_trade_return_net",
            "annualized_return_gross",
            "annualized_return_net",
            "annualized_volatility",
            "sharpe_gross",
            "sharpe_net",
            "max_drawdown_gross",
            "max_drawdown_net",
            "turnover_mean",
            "mean_prob_up",
        ):
            row[f"{prefix}_{key}"] = values.get(key)
    return row


def main() -> int:
    args = parse_args()
    stage_root = Path(__file__).resolve().parents[1]
    config = load_yaml(stage_root / args.config)
    output_root = Path(config["paths"]["output_root"])
    tables_root = Path(config["paths"].get("tables_root", stage_root / "reports" / "tables"))
    tables_root.mkdir(parents=True, exist_ok=True)
    backup_root = Path(args.backup_root)

    rows: list[dict[str, Any]] = []
    for seed in args.run_seeds:
        for image_window in args.image_windows:
            for return_horizon in args.return_horizons:
                for image_spec in args.image_specs:
                    experiment = experiment_name(image_window, image_spec, return_horizon)
                    seed_name = f"seed_{seed}"
                    metrics_path = output_root / "metrics" / experiment / seed_name / f"{args.split}_metrics.json"
                    trading_path = output_root / "metrics" / experiment / seed_name / f"{args.split}_trading_metrics.json"
                    classification = read_json(metrics_path)
                    trading = read_json(trading_path)

                    if classification is None and args.include_backup_zips:
                        classification = read_json_from_zips(
                            backup_root,
                            experiment,
                            seed,
                            f"metrics/{experiment}/{seed_name}/{args.split}_metrics.json",
                        )
                    if trading is None and args.include_backup_zips:
                        trading = read_json_from_zips(
                            backup_root,
                            experiment,
                            seed,
                            f"metrics/{experiment}/{seed_name}/{args.split}_trading_metrics.json",
                        )

                    row = {
                        "experiment_name": experiment,
                        "image_window": image_window,
                        "return_horizon": return_horizon,
                        "image_spec": image_spec,
                        "run_seed": seed,
                    }
                    row.update(flatten_classification(classification))
                    row.update(flatten_trading(trading))
                    rows.append(row)

    seed_table = pd.DataFrame(rows)
    available = seed_table[
        (seed_table["classification_available"] == True)  # noqa: E712 - pandas boolean filter.
        & (seed_table["trading_available"] == True)
    ].copy()

    metric_columns = [
        column
        for column in available.columns
        if column
        not in {
            "experiment_name",
            "image_window",
            "return_horizon",
            "image_spec",
            "run_seed",
            "classification_available",
            "trading_available",
        }
        and pd.api.types.is_numeric_dtype(available[column])
    ]
    if available.empty:
        summary_table = pd.DataFrame()
    else:
        summary_table = (
            available.groupby(["image_window", "return_horizon", "image_spec"], as_index=False)[metric_columns]
            .agg(["mean", "std", "count"])
        )
        summary_table.columns = [
            "_".join([str(item) for item in column if item])
            for column in summary_table.columns.to_flat_index()
        ]
        summary_table = summary_table.reset_index()

    seed_csv = tables_root / f"{args.output_prefix}_seed_results.csv"
    summary_csv = tables_root / f"{args.output_prefix}_mean_std_results.csv"
    seed_table.to_csv(seed_csv, index=False)
    summary_table.to_csv(summary_csv, index=False)

    print(
        json.dumps(
            {
                "status": "ok",
                "num_expected_rows": int(len(seed_table)),
                "num_available_rows": int(len(available)),
                "seed_results_csv": str(seed_csv),
                "mean_std_results_csv": str(summary_csv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
