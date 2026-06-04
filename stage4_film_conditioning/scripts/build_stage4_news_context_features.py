"""Build normalized Stage 4 news context features from TF-IDF/SVD vectors.

This is the 4-N5 news-context step. It reads the 4-N4 sample-level
TF-IDF/SVD vector table and writes a Stage4 context artifact layout that later
model runners can consume:

    context_features.csv
    context_scaler.json
    context_feature_audit.json
    context_feature_summary.csv

The first news context vector is headline-only and leakage-safe:

    news_svd_7d + news_svd_20d + news_svd_60d
    + log1p(news count / unique-source count for 7d, 20d, 60d)

All normalization statistics are fit on the train split only.

Marker: sample-level news context feature builder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from audit_stage4_news_alignment import NEWS_WINDOWS, SPLIT_ORDER
from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import get_stage4_model_config, sanitize_name_suffix


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=None)
    parser.add_argument("--image-spec", default=None)
    parser.add_argument("--return-horizon", type=int, default=None)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--window-days", type=int, nargs="+", default=list(NEWS_WINDOWS))
    parser.add_argument(
        "--input-prefix",
        default="stage4_news_tfidf_svd",
        help="4-N4 output prefix under outputs/stage4/news.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Optional explicit news_tfidf_svd_features parquet/csv path.",
    )
    parser.add_argument(
        "--output-prefix",
        default="stage4_news_context",
        help="Prefix for context artifact and report names.",
    )
    parser.add_argument(
        "--feature-set-name",
        default="tfidf_svd_w7_20_60",
        help="Filesystem-safe suffix describing the news context feature set.",
    )
    parser.add_argument("--clip-lower-quantile", type=float, default=0.01)
    parser.add_argument("--clip-upper-quantile", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0e-8)
    parser.add_argument(
        "--write-report-copy",
        action="store_true",
        help="Also write small audit/summary files under reports/tables.",
    )
    return parser.parse_args()


def main() -> None:
    """Build normalized news context artifacts."""

    args = parse_args()
    validate_args(args)

    config = load_config(args.config)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    stage4_model = get_stage4_model_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
    image_spec = str(args.image_spec or stage4_model["primary_image_spec"])
    return_horizon = int(args.return_horizon or stage4_model["primary_return_horizon"])
    window_days = tuple(sorted({int(day) for day in args.window_days}))

    input_path = resolve_input_path(
        paths.output_root,
        input_prefix=str(args.input_prefix),
        explicit_path=Path(args.input_path) if args.input_path else None,
        image_window=image_window,
        return_horizon=return_horizon,
    )
    vector_frame = read_table(input_path)
    vector_frame = sort_samples(vector_frame)
    svd_dim = infer_svd_dim(vector_frame, window_days)
    feature_order = build_feature_order(window_days, svd_dim)
    validate_input_frame(vector_frame, feature_order, window_days)

    context_table, scaler = fit_transform_news_context_features(
        vector_frame,
        feature_order=feature_order,
        lower_q=float(args.clip_lower_quantile),
        upper_q=float(args.clip_upper_quantile),
        epsilon=float(args.epsilon),
    )
    audit = build_audit(
        context_table,
        scaler=scaler,
        input_path=input_path,
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        window_days=window_days,
        svd_dim=svd_dim,
    )
    summary = build_feature_summary(context_table, feature_order)

    context_name = make_news_context_output_name(
        output_prefix=str(args.output_prefix),
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        feature_set_name=str(args.feature_set_name),
    )
    output_dir = paths.context_root / context_name / f"seed_{int(args.run_seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_root.mkdir(parents=True, exist_ok=True)

    context_csv = output_dir / "context_features.csv"
    context_parquet = output_dir / "context_features.parquet"
    scaler_path = output_dir / "context_scaler.json"
    audit_path = output_dir / "context_feature_audit.json"
    summary_path = output_dir / "context_feature_summary.csv"
    manifest_path = output_dir / "news_context_manifest.json"

    context_table.to_csv(context_csv, index=False)
    context_table.to_parquet(context_parquet, index=False)
    scaler_path.write_text(json.dumps(_jsonable(scaler), indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
    summary.to_csv(summary_path, index=False)
    manifest_path.write_text(json.dumps(_jsonable(build_manifest(audit, scaler)), indent=2), encoding="utf-8")

    report_files: dict[str, str] = {}
    if args.write_report_copy:
        report_prefix = f"{context_name}_seed{int(args.run_seed)}"
        report_audit = paths.tables_root / f"{report_prefix}_news_context_feature_audit.json"
        report_summary = paths.tables_root / f"{report_prefix}_news_context_feature_summary.csv"
        report_manifest = paths.tables_root / f"{report_prefix}_news_context_manifest.json"
        report_audit.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
        summary.to_csv(report_summary, index=False)
        report_manifest.write_text(
            json.dumps(_jsonable(build_manifest(audit, scaler)), indent=2),
            encoding="utf-8",
        )
        report_files = {
            "report_audit": str(report_audit),
            "report_summary": str(report_summary),
            "report_manifest": str(report_manifest),
        }

    print(
        json.dumps(
            {
                "status": "ok",
                "stage": "4-N5",
                "context_name": context_name,
                "image_window": image_window,
                "image_spec": image_spec,
                "return_horizon": return_horizon,
                "run_seed": int(args.run_seed),
                "window_days": list(window_days),
                "svd_dim_per_window": int(svd_dim),
                "context_dim": len(feature_order),
                "split_counts": audit["split_counts"],
                "missing_warnings": audit["warnings"],
                "written": {
                    "context_features": str(context_csv),
                    "context_features_parquet": str(context_parquet),
                    "context_scaler": str(scaler_path),
                    "context_feature_audit": str(audit_path),
                    "context_feature_summary": str(summary_path),
                    "news_context_manifest": str(manifest_path),
                    **report_files,
                },
            },
            indent=2,
        )
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before reading files."""

    if args.clip_lower_quantile < 0 or args.clip_upper_quantile > 1:
        raise ValueError("Clip quantiles must be between 0 and 1.")
    if args.clip_lower_quantile > args.clip_upper_quantile:
        raise ValueError("--clip-lower-quantile must be <= --clip-upper-quantile.")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")


def resolve_input_path(
    output_root: Path,
    *,
    input_prefix: str,
    explicit_path: Path | None,
    image_window: int,
    return_horizon: int,
) -> Path:
    """Resolve the 4-N4 sample-level news vector path."""

    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {explicit_path}")
        return explicit_path

    base = output_root / "news" / f"{input_prefix}_i{image_window}_r{return_horizon}"
    candidates = [
        base / "news_tfidf_svd_features.parquet",
        base / "news_tfidf_svd_features.csv.gz",
        base / "news_tfidf_svd_features.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "4-N4 news TF-IDF/SVD feature table is missing. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def read_table(path: Path) -> pd.DataFrame:
    """Read parquet/csv/csv.gz table."""

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".gz", ".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input table format: {path}")


def sort_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort sample rows in a stable split/date/sample order."""

    result = frame.copy()
    if "split" not in result.columns:
        raise KeyError("Input table must contain split.")
    if "sample_index" not in result.columns:
        raise KeyError("Input table must contain sample_index.")
    result["_split_order"] = result["split"].map(SPLIT_ORDER).fillna(99).astype(int)
    if "image_end_date" in result.columns:
        result["image_end_date"] = pd.to_datetime(result["image_end_date"]).dt.date.astype(str)
        sort_columns = ["_split_order", "image_end_date", "sample_index"]
    else:
        sort_columns = ["_split_order", "sample_index"]
    result = result.sort_values(sort_columns).drop(columns=["_split_order"])
    return result.reset_index(drop=True)


def infer_svd_dim(frame: pd.DataFrame, window_days: tuple[int, ...]) -> int:
    """Infer the common SVD dimension from column names."""

    dims: list[int] = []
    for window in window_days:
        prefix = f"{window}d"
        columns = [
            column for column in frame.columns
            if column.startswith(f"news_svd_{prefix}_")
        ]
        if not columns:
            raise ValueError(f"No SVD columns found for window {prefix}.")
        dims.append(len(columns))
    if len(set(dims)) != 1:
        raise ValueError(f"SVD dimensions differ by window: {dict(zip(window_days, dims, strict=True))}")
    return int(dims[0])


def build_feature_order(window_days: tuple[int, ...], svd_dim: int) -> list[str]:
    """Return raw feature order for the first news context vector."""

    features: list[str] = []
    for window in window_days:
        prefix = f"{window}d"
        features.extend(f"news_svd_{prefix}_{index:02d}" for index in range(svd_dim))
    for window in window_days:
        prefix = f"{window}d"
        features.append(f"news_count_log1p_{prefix}")
        features.append(f"unique_source_count_log1p_{prefix}")
    return features


def validate_input_frame(
    frame: pd.DataFrame,
    feature_order: list[str],
    window_days: tuple[int, ...],
) -> None:
    """Validate required metadata and feature columns."""

    required = {
        "split",
        "sample_index",
        "image_end_date",
        "label_end_date",
        *feature_order,
    }
    for window in window_days:
        prefix = f"{window}d"
        required.update(
            {
                f"news_count_{prefix}",
                f"unique_source_count_{prefix}",
                f"headline_text_hash_{prefix}",
                f"news_svd_norm_{prefix}",
            }
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Input news vector table is missing columns: " + ", ".join(missing))
    if "train" not in set(frame["split"].astype(str)):
        raise ValueError("Input news vector table has no train split rows.")
    duplicates = frame["sample_index"].duplicated()
    if duplicates.any():
        examples = frame.loc[duplicates, "sample_index"].head().tolist()
        raise ValueError(f"Input news vector table has duplicate sample_index rows: {examples}")


def fit_transform_news_context_features(
    frame: pd.DataFrame,
    *,
    feature_order: list[str],
    lower_q: float,
    upper_q: float,
    epsilon: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Impute, clip, and z-score news features with train-only statistics."""

    metadata_columns = [
        "split",
        "sample_index",
        "image_end_date",
        "label_end_date",
        "strict_policy",
        "same_day_excluded",
    ]
    passthrough_prefixes = (
        "news_count_",
        "unique_source_count_",
        "headline_text_hash_",
        "news_svd_norm_",
    )
    passthrough_columns = [
        column for column in frame.columns
        if column.startswith(passthrough_prefixes) and column not in feature_order
    ]
    context = frame[[column for column in metadata_columns if column in frame.columns]].copy()
    for column in passthrough_columns:
        context[column] = frame[column].values

    train_mask = frame["split"].astype(str).eq("train")
    if not train_mask.any():
        raise ValueError("Cannot fit news context scaler because split == train is empty.")

    missing_rate_by_split = missing_rates(frame, feature_order)
    medians: dict[str, float] = {}
    q01: dict[str, float] = {}
    q99: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    feature_columns: dict[str, Any] = {}

    for feature in feature_order:
        raw = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        train_raw = raw.loc[train_mask]
        median = float(train_raw.median())
        if np.isnan(median):
            raise ValueError(f"Cannot fit all-missing news feature: {feature}")
        imputed = raw.fillna(median)
        lower = float(imputed.loc[train_mask].quantile(lower_q))
        upper = float(imputed.loc[train_mask].quantile(upper_q))
        if lower > upper:
            lower, upper = upper, lower
        clipped = imputed.clip(lower=lower, upper=upper)
        train_clipped = clipped.loc[train_mask]
        mean = float(train_clipped.mean())
        std = float(train_clipped.std(ddof=0))
        if not np.isfinite(std) or std < epsilon:
            std = 1.0

        feature_columns[feature] = raw
        feature_columns[f"{feature}_missing"] = raw.isna()
        feature_columns[f"{feature}_imputed_clipped"] = clipped
        feature_columns[f"{feature}_normalized"] = (clipped - mean) / max(std, epsilon)
        medians[feature] = median
        q01[feature] = lower
        q99[feature] = upper
        means[feature] = mean
        stds[feature] = std

    context = pd.concat([context, pd.DataFrame(feature_columns, index=context.index)], axis=1)

    scaler = {
        "feature_order": feature_order,
        "normalized_feature_columns": [f"{feature}_normalized" for feature in feature_order],
        "feature_groups": feature_groups(feature_order),
        "transforms": {feature: "identity_or_log1p_precomputed" for feature in feature_order},
        "medians": medians,
        "q01": q01,
        "q99": q99,
        "means": means,
        "stds": stds,
        "epsilon": float(epsilon),
        "missing_rate_by_split": missing_rate_by_split,
        "fit_on": "train",
        "normalization": "train_median_impute_train_quantile_clip_train_zscore",
    }
    return context, scaler


def feature_groups(feature_order: list[str]) -> dict[str, list[str]]:
    """Group news context features for reporting."""

    groups: dict[str, list[str]] = {
        "news_svd_7d": [],
        "news_svd_20d": [],
        "news_svd_60d": [],
        "news_log_counts": [],
    }
    for feature in feature_order:
        if feature.startswith("news_svd_7d_"):
            groups["news_svd_7d"].append(feature)
        elif feature.startswith("news_svd_20d_"):
            groups["news_svd_20d"].append(feature)
        elif feature.startswith("news_svd_60d_"):
            groups["news_svd_60d"].append(feature)
        elif "count_log1p" in feature:
            groups["news_log_counts"].append(feature)
    return groups


def missing_rates(frame: pd.DataFrame, feature_order: list[str]) -> dict[str, dict[str, float]]:
    """Raw feature missing rate by split."""

    result: dict[str, dict[str, float]] = {}
    for split, split_frame in frame.groupby("split", sort=True):
        result[str(split)] = {
            feature: float(pd.to_numeric(split_frame[feature], errors="coerce").isna().mean())
            for feature in feature_order
        }
    return result


def build_audit(
    context_table: pd.DataFrame,
    *,
    scaler: dict[str, Any],
    input_path: Path,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    window_days: tuple[int, ...],
    svd_dim: int,
) -> dict[str, Any]:
    """Build compact audit metadata for the news context table."""

    split_counts = {
        str(split): int(count)
        for split, count in context_table["split"].value_counts(sort=False).items()
    }
    feature_order = list(scaler["feature_order"])
    warnings: list[str] = []
    for split, rates in scaler["missing_rate_by_split"].items():
        high_missing = [
            feature for feature, rate in rates.items()
            if float(rate) > 0.05
        ]
        if high_missing:
            warnings.append(
                f"{split}: high missing rate features >5%: "
                + ", ".join(high_missing[:10])
            )
    for feature, std in scaler["stds"].items():
        if float(std) == 1.0 and float(scaler["q01"][feature]) == float(scaler["q99"][feature]):
            warnings.append(f"{feature}: near-constant after train clipping")

    normalized_columns = scaler["normalized_feature_columns"]
    finite_normalized = bool(
        np.isfinite(context_table[normalized_columns].to_numpy(dtype="float64")).all()
    )
    if not finite_normalized:
        warnings.append("normalized feature matrix contains non-finite values")

    return {
        "status": "ok",
        "stage": "4-N5",
        "input_path": str(input_path),
        "image_window": int(image_window),
        "image_spec": str(image_spec),
        "return_horizon": int(return_horizon),
        "window_days": list(window_days),
        "svd_dim_per_window": int(svd_dim),
        "context_dim": int(len(feature_order)),
        "feature_order": feature_order,
        "feature_group_counts": {
            group: int(len(features))
            for group, features in scaler["feature_groups"].items()
        },
        "split_counts": split_counts,
        "date_min": str(context_table["image_end_date"].min()),
        "date_max": str(context_table["image_end_date"].max()),
        "fit_on": "train",
        "normalization": scaler["normalization"],
        "finite_normalized": finite_normalized,
        "warnings": warnings,
    }


def build_feature_summary(context_table: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    """Build split/feature raw and normalized summary."""

    rows: list[dict[str, Any]] = []
    for split, split_frame in context_table.groupby("split", sort=True):
        for feature in feature_order:
            raw = pd.to_numeric(split_frame[feature], errors="coerce")
            normalized = pd.to_numeric(split_frame[f"{feature}_normalized"], errors="coerce")
            rows.append(
                {
                    "split": str(split),
                    "feature": feature,
                    "feature_group": feature_group_name(feature),
                    "num_rows": int(len(split_frame)),
                    "raw_missing_rate": float(raw.isna().mean()),
                    "raw_mean": safe_float(raw.mean()),
                    "raw_std": safe_float(raw.std(ddof=0)),
                    "raw_min": safe_float(raw.min()),
                    "raw_max": safe_float(raw.max()),
                    "normalized_mean": safe_float(normalized.mean()),
                    "normalized_std": safe_float(normalized.std(ddof=0)),
                    "normalized_min": safe_float(normalized.min()),
                    "normalized_max": safe_float(normalized.max()),
                }
            )
    return pd.DataFrame(rows)


def build_manifest(audit: dict[str, Any], scaler: dict[str, Any]) -> dict[str, Any]:
    """Build compact manifest for model-runner handoff."""

    return {
        "status": "ok",
        "stage": "4-N5",
        "context_source": "headline_only_news_tfidf_svd",
        "alignment_policy": "strict_t_minus_1_calendar_date",
        "context_dim": audit["context_dim"],
        "feature_order": scaler["feature_order"],
        "normalized_feature_columns": scaler["normalized_feature_columns"],
        "feature_group_counts": audit["feature_group_counts"],
        "split_counts": audit["split_counts"],
        "fit_on": "train",
        "normalization": scaler["normalization"],
    }


def make_news_context_output_name(
    *,
    output_prefix: str,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    feature_set_name: str,
) -> str:
    """Return filesystem-safe Stage 4 news context artifact name."""

    suffix = sanitize_name_suffix(feature_set_name)
    base = f"{output_prefix}_i{int(image_window)}_{image_spec}_r{int(return_horizon)}"
    return f"{base}_{suffix}" if suffix else base


def feature_group_name(feature: str) -> str:
    """Return feature group label."""

    if feature.startswith("news_svd_7d_"):
        return "news_svd_7d"
    if feature.startswith("news_svd_20d_"):
        return "news_svd_20d"
    if feature.startswith("news_svd_60d_"):
        return "news_svd_60d"
    if "count_log1p" in feature:
        return "news_log_counts"
    return "other"


def safe_float(value: Any) -> float | None:
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
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


if __name__ == "__main__":
    main()
