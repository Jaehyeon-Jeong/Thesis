"""Build Stage 4 structured context features and train-only scaler.

The output table contains raw context values, missing flags, transformed values,
and normalized values. Scaling statistics are fit on the train split only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from stage2_btc.data import build_btc_samples, build_btc_splits, find_btc_ohlcv_source
from stage2_btc.data import load_btc_ohlcv
from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import get_context_config, get_stage4_model_config
from stage4_film.context import build_context_feature_table, find_fear_greed_source
from stage4_film.context import fit_transform_context_features, load_fear_greed_index
from stage4_film.context.features import build_context_feature_audit
from stage4_film.context.features import make_context_output_name


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=None)
    parser.add_argument("--image-spec", default=None)
    parser.add_argument("--return-horizon", type=int, default=None)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument(
        "--write-report-copy",
        action="store_true",
        help="Also write small audit/summary files under reports/tables.",
    )
    return parser.parse_args()


def main() -> None:
    """Build context feature artifacts."""

    args = parse_args()
    config = load_config(args.config)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    stage4_model = get_stage4_model_config(config)
    context_config = get_context_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
    image_spec = str(args.image_spec or stage4_model["primary_image_spec"])
    return_horizon = int(args.return_horizon or stage4_model["primary_return_horizon"])
    context_window = int(context_config["context_window"])
    primary_features = [str(feature) for feature in context_config["primary_features"]]

    btc_source = find_btc_ohlcv_source(config, paths)
    ohlcv = load_btc_ohlcv(btc_source)
    samples = build_btc_samples(
        ohlcv,
        config,
        image_window=image_window,
        return_horizon=return_horizon,
    )
    splits = build_btc_splits(samples, config)

    fg_source = find_fear_greed_source(config, paths)
    fg_config = context_config.get("fear_greed", {})
    fear_greed = load_fear_greed_index(
        fg_source,
        date_column=str(fg_config.get("date_column", "date")),
        value_column=str(fg_config.get("value_column", "value")),
        classification_column=str(fg_config.get("classification_column", "classification")),
    )

    raw_table = build_context_feature_table(
        ohlcv=ohlcv,
        samples_by_split=splits,
        fear_greed=fear_greed,
        config=config,
    )
    context_table, scaler = fit_transform_context_features(raw_table, config)
    audit = build_context_feature_audit(context_table, primary_features)
    summary = _build_feature_summary(context_table, primary_features)

    context_name = make_context_output_name(
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        context_window=context_window,
    )
    output_dir = paths.context_root / context_name / f"seed_{int(args.run_seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    context_table_path = output_dir / "context_features.csv"
    scaler_path = output_dir / "context_scaler.json"
    audit_path = output_dir / "context_feature_audit.json"
    summary_path = output_dir / "context_feature_summary.csv"

    context_table.to_csv(context_table_path, index=False)
    scaler_path.write_text(json.dumps(_jsonable(scaler.as_dict()), indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
    summary.to_csv(summary_path, index=False)

    report_files: dict[str, str] = {}
    if args.write_report_copy:
        paths.tables_root.mkdir(parents=True, exist_ok=True)
        report_prefix = f"{context_name}_seed{int(args.run_seed)}"
        report_audit = paths.tables_root / f"{report_prefix}_context_feature_audit.json"
        report_summary = paths.tables_root / f"{report_prefix}_context_feature_summary.csv"
        report_audit.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
        summary.to_csv(report_summary, index=False)
        report_files = {
            "report_audit": str(report_audit),
            "report_summary": str(report_summary),
        }

    print(
        json.dumps(
            {
                "status": "ok",
                "context_name": context_name,
                "image_window": image_window,
                "image_spec": image_spec,
                "return_horizon": return_horizon,
                "context_window": context_window,
                "run_seed": int(args.run_seed),
                "split_counts": audit["split_counts"],
                "missing_warnings": audit["warnings"],
                "written": {
                    "context_features": str(context_table_path),
                    "context_scaler": str(scaler_path),
                    "context_feature_audit": str(audit_path),
                    "context_feature_summary": str(summary_path),
                    **report_files,
                },
            },
            indent=2,
        )
    )


def _build_feature_summary(context_table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Build a compact split/feature summary table."""

    rows: list[dict[str, Any]] = []
    for split, split_frame in context_table.groupby("split", sort=True):
        for feature in features:
            raw = pd.to_numeric(split_frame[feature], errors="coerce")
            normalized = pd.to_numeric(split_frame[f"{feature}_normalized"], errors="coerce")
            rows.append(
                {
                    "split": str(split),
                    "feature": feature,
                    "num_rows": int(len(split_frame)),
                    "raw_missing_rate": float(raw.isna().mean()),
                    "raw_mean": _safe_float(raw.mean()),
                    "raw_std": _safe_float(raw.std(ddof=0)),
                    "raw_min": _safe_float(raw.min()),
                    "raw_max": _safe_float(raw.max()),
                    "normalized_mean": _safe_float(normalized.mean()),
                    "normalized_std": _safe_float(normalized.std(ddof=0)),
                    "normalized_min": _safe_float(normalized.min()),
                    "normalized_max": _safe_float(normalized.max()),
                }
            )
    return pd.DataFrame(rows)


def _safe_float(value: Any) -> float | None:
    """Return None for NaN-like values, otherwise float."""

    if pd.isna(value):
        return None
    return float(value)


def _jsonable(value: Any) -> Any:
    """Convert common pandas/numpy scalar values to JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


if __name__ == "__main__":
    main()
