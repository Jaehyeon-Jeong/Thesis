"""Build Stage 4 OFR FSI context features.

This is the 4-N13-1 macro-regime context step. It reads the official OFR
Financial Stress Index CSV, aligns it to Stage 4 BTC samples with an as-of
policy, and writes a prebuilt context artifact that the existing Stage 4
runner can consume:

    context_features.csv
    context_scaler.json
    context_feature_audit.json
    context_feature_summary.csv

OFR FSI is not a direct RORO index. It is used here as an official
financial-stress / risk-off proxy:

    higher OFR FSI -> higher financial stress -> risk-off context

The BTC direction is not hard-coded. The FiLM/context model learns how this
stress signal should modulate the frozen Stage 2 chart features.

Marker: OFR FSI risk-off proxy context builder
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

from stage2_btc.data import build_btc_samples, build_btc_splits, find_btc_ohlcv_source
from stage2_btc.data import load_btc_ohlcv
from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import get_stage4_model_config, sanitize_name_suffix


DEFAULT_FSI_URL = "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
BASE_FSI_FEATURES = [
    "ofr_fsi_value",
    "ofr_fsi_mean_20",
    "ofr_fsi_mean_60",
    "ofr_fsi_delta_20",
    "ofr_fsi_delta_60",
    "ofr_fsi_std_60",
]
CATEGORY_FEATURES = [
    "ofr_credit",
    "ofr_equity_valuation",
    "ofr_funding",
    "ofr_safe_assets",
    "ofr_volatility",
]
OFR_RENAME_MAP = {
    "OFR FSI": "ofr_fsi_value",
    "Credit": "ofr_credit",
    "Equity valuation": "ofr_equity_valuation",
    "Safe assets": "ofr_safe_assets",
    "Funding": "ofr_funding",
    "Volatility": "ofr_volatility",
    "United States": "ofr_region_united_states",
    "Other advanced economies": "ofr_region_other_advanced_economies",
    "Emerging markets": "ofr_region_emerging_markets",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=None)
    parser.add_argument("--image-spec", default=None)
    parser.add_argument("--return-horizon", type=int, default=None)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument(
        "--fsi-csv",
        default=DEFAULT_FSI_URL,
        help="Official OFR FSI CSV URL or local CSV path.",
    )
    parser.add_argument(
        "--asof-lag-days",
        type=int,
        default=1,
        help=(
            "Conservative lag between BTC image end date t and the FSI value "
            "used for context. Default 1 means use FSI as of t-1."
        ),
    )
    parser.add_argument(
        "--include-category-features",
        action="store_true",
        help="Include OFR FSI component categories beside the headline FSI features.",
    )
    parser.add_argument(
        "--output-prefix",
        default="stage4_fsi_context",
        help="Prefix for context artifact and report names.",
    )
    parser.add_argument(
        "--feature-set-name",
        default="ofr_fsi_lag1_w20_60",
        help="Filesystem-safe suffix describing this FSI context feature set.",
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
    """Build normalized OFR FSI context artifacts."""

    args = parse_args()
    validate_args(args)

    config = load_config(args.config)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    stage4_model = get_stage4_model_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
    image_spec = str(args.image_spec or stage4_model["primary_image_spec"])
    return_horizon = int(args.return_horizon or stage4_model["primary_return_horizon"])

    btc_source = find_btc_ohlcv_source(config, paths)
    ohlcv = load_btc_ohlcv(btc_source)
    samples = build_btc_samples(
        ohlcv,
        config,
        image_window=image_window,
        return_horizon=return_horizon,
    )
    splits = build_btc_splits(samples, config)
    sample_table = build_sample_table(splits)

    fsi = load_ofr_fsi(args.fsi_csv)
    raw_context = build_fsi_sample_context(
        samples=sample_table,
        fsi=fsi,
        asof_lag_days=int(args.asof_lag_days),
        include_category_features=bool(args.include_category_features),
    )
    feature_order = BASE_FSI_FEATURES + (
        CATEGORY_FEATURES if bool(args.include_category_features) else []
    )
    context_table, scaler = fit_transform_fsi_context_features(
        raw_context,
        feature_order=feature_order,
        lower_q=float(args.clip_lower_quantile),
        upper_q=float(args.clip_upper_quantile),
        epsilon=float(args.epsilon),
    )
    audit = build_audit(
        context_table,
        scaler=scaler,
        fsi=fsi,
        fsi_source=str(args.fsi_csv),
        btc_source=str(btc_source),
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        asof_lag_days=int(args.asof_lag_days),
    )
    summary = build_feature_summary(context_table, feature_order)

    context_name = make_fsi_context_output_name(
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
    manifest_path = output_dir / "fsi_context_manifest.json"

    context_table.to_csv(context_csv, index=False)
    parquet_written = write_parquet_if_available(context_table, context_parquet)
    scaler_path.write_text(json.dumps(_jsonable(scaler), indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
    summary.to_csv(summary_path, index=False)
    manifest_path.write_text(
        json.dumps(_jsonable(build_manifest(audit, scaler)), indent=2),
        encoding="utf-8",
    )

    report_files: dict[str, str] = {}
    if args.write_report_copy:
        report_prefix = f"{context_name}_seed{int(args.run_seed)}"
        report_audit = paths.tables_root / f"{report_prefix}_fsi_context_feature_audit.json"
        report_summary = paths.tables_root / f"{report_prefix}_fsi_context_feature_summary.csv"
        report_manifest = paths.tables_root / f"{report_prefix}_fsi_context_manifest.json"
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
                "stage": "4-N13-1",
                "context_name": context_name,
                "context_source": "ofr_fsi_risk_off_proxy",
                "image_window": image_window,
                "image_spec": image_spec,
                "return_horizon": return_horizon,
                "run_seed": int(args.run_seed),
                "asof_lag_days": int(args.asof_lag_days),
                "context_dim": int(len(feature_order)),
                "split_counts": audit["split_counts"],
                "missing_warnings": audit["warnings"],
                "written": {
                    "context_features": str(context_csv),
                    "context_features_parquet": str(context_parquet) if parquet_written else None,
                    "context_scaler": str(scaler_path),
                    "context_feature_audit": str(audit_path),
                    "context_feature_summary": str(summary_path),
                    "fsi_context_manifest": str(manifest_path),
                    **report_files,
                },
            },
            indent=2,
        )
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before reading files."""

    if args.asof_lag_days < 0:
        raise ValueError("--asof-lag-days must be >= 0.")
    if args.clip_lower_quantile < 0 or args.clip_upper_quantile > 1:
        raise ValueError("Clip quantiles must be between 0 and 1.")
    if args.clip_lower_quantile > args.clip_upper_quantile:
        raise ValueError("--clip-lower-quantile must be <= --clip-upper-quantile.")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")


def load_ofr_fsi(source: str | Path) -> pd.DataFrame:
    """Load and standardize the official OFR Financial Stress Index table."""

    frame = pd.read_csv(source)
    if "Date" not in frame.columns:
        raise KeyError("OFR FSI table must contain a Date column.")
    missing = [column for column in OFR_RENAME_MAP if column not in frame.columns]
    if missing:
        raise KeyError("OFR FSI table is missing expected columns: " + ", ".join(missing))

    frame = frame.rename(columns=OFR_RENAME_MAP)
    frame["ofr_date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    keep_columns = ["ofr_date", *OFR_RENAME_MAP.values()]
    result = frame.loc[:, keep_columns].copy()
    for column in keep_columns:
        if column != "ofr_date":
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.sort_values("ofr_date").drop_duplicates("ofr_date", keep="last")
    if result.empty:
        raise ValueError("OFR FSI table is empty after cleaning.")
    return result.reset_index(drop=True)


def build_sample_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine Stage 2 split sample tables into one stable sample table."""

    rows: list[pd.DataFrame] = []
    for split_name, frame in splits.items():
        sample_frame = frame.copy()
        sample_frame["split"] = str(split_name)
        rows.append(sample_frame)
    samples = pd.concat(rows, ignore_index=True)
    samples["Date"] = pd.to_datetime(samples["Date"]).dt.normalize()
    if "label_end_date" in samples.columns:
        samples["label_end_date"] = pd.to_datetime(samples["label_end_date"]).dt.normalize()
    samples["_split_order"] = samples["split"].map(SPLIT_ORDER).fillna(99).astype(int)
    samples = samples.sort_values(["_split_order", "Date", "sample_index"])
    return samples.drop(columns=["_split_order"]).reset_index(drop=True)


def build_fsi_sample_context(
    *,
    samples: pd.DataFrame,
    fsi: pd.DataFrame,
    asof_lag_days: int,
    include_category_features: bool,
) -> pd.DataFrame:
    """Align OFR FSI to BTC sample dates and compute trailing features."""

    sample_min = pd.to_datetime(samples["Date"]).min().normalize()
    sample_max = pd.to_datetime(samples["Date"]).max().normalize()
    calendar = pd.DataFrame({"Date": pd.date_range(sample_min, sample_max, freq="D")})
    calendar["ofr_asof_date"] = calendar["Date"] - pd.to_timedelta(int(asof_lag_days), unit="D")

    fsi_sorted = fsi.sort_values("ofr_date").reset_index(drop=True)
    aligned = pd.merge_asof(
        calendar.sort_values("ofr_asof_date"),
        fsi_sorted,
        left_on="ofr_asof_date",
        right_on="ofr_date",
        direction="backward",
    ).sort_values("Date")
    aligned["ofr_source_date"] = aligned["ofr_date"]
    aligned["ofr_age_days"] = (aligned["Date"] - aligned["ofr_source_date"]).dt.days
    aligned["ofr_exact_asof_match"] = aligned["ofr_source_date"].eq(aligned["ofr_asof_date"])
    aligned["ofr_missing"] = aligned["ofr_fsi_value"].isna()

    value = pd.to_numeric(aligned["ofr_fsi_value"], errors="coerce")
    aligned["ofr_fsi_mean_20"] = value.rolling(20, min_periods=20).mean()
    aligned["ofr_fsi_mean_60"] = value.rolling(60, min_periods=60).mean()
    aligned["ofr_fsi_delta_20"] = value - value.shift(20)
    aligned["ofr_fsi_delta_60"] = value - value.shift(60)
    aligned["ofr_fsi_std_60"] = value.rolling(60, min_periods=60).std(ddof=0)

    context_columns = [
        "Date",
        "ofr_asof_date",
        "ofr_source_date",
        "ofr_age_days",
        "ofr_missing",
        "ofr_exact_asof_match",
        *BASE_FSI_FEATURES,
    ]
    if include_category_features:
        context_columns.extend(CATEGORY_FEATURES)
    fsi_context = aligned.loc[:, context_columns].copy()

    table = samples.merge(fsi_context, on="Date", how="left")
    table["image_end_date"] = pd.to_datetime(table["Date"]).dt.date.astype(str)
    table["ofr_asof_date"] = pd.to_datetime(table["ofr_asof_date"]).dt.date.astype("string")
    table["ofr_source_date"] = pd.to_datetime(table["ofr_source_date"]).dt.date.astype("string")
    if "label_end_date" in table.columns:
        table["label_end_date"] = pd.to_datetime(table["label_end_date"]).dt.date.astype(str)

    feature_order = BASE_FSI_FEATURES + (CATEGORY_FEATURES if include_category_features else [])
    for feature in feature_order:
        table[f"{feature}_missing"] = ~np.isfinite(pd.to_numeric(table[feature], errors="coerce"))

    metadata = [
        "split",
        "sample_index",
        "Date",
        "image_end_date",
        "start_index",
        "end_index",
        "label_end_index",
        "label_end_date",
        "image_window",
        "return_horizon",
        "entry_close",
        "exit_close",
        "future_return",
        "label",
        "ofr_asof_date",
        "ofr_source_date",
        "ofr_age_days",
        "ofr_missing",
        "ofr_exact_asof_match",
    ]
    ordered = (
        [column for column in metadata if column in table.columns]
        + feature_order
        + [f"{feature}_missing" for feature in feature_order]
        + [column for column in table.columns if column not in set(metadata + feature_order)]
    )
    return table.loc[:, list(dict.fromkeys(ordered))].reset_index(drop=True)


def fit_transform_fsi_context_features(
    frame: pd.DataFrame,
    *,
    feature_order: list[str],
    lower_q: float,
    upper_q: float,
    epsilon: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Impute, clip, and z-score FSI features with train-only statistics."""

    context = frame.copy()
    train_mask = context["split"].astype(str).eq("train")
    if not train_mask.any():
        raise ValueError("Cannot fit OFR FSI context scaler because split == train is empty.")

    missing_rate_by_split = missing_rates(context, feature_order)
    medians: dict[str, float] = {}
    q01: dict[str, float] = {}
    q99: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for feature in feature_order:
        raw = pd.to_numeric(context[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        train_raw = raw.loc[train_mask]
        median = float(train_raw.median())
        if np.isnan(median):
            raise ValueError(f"Cannot fit all-missing OFR FSI feature: {feature}")
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

        context[f"{feature}_imputed_clipped"] = clipped
        context[f"{feature}_normalized"] = (clipped - mean) / max(std, epsilon)
        medians[feature] = median
        q01[feature] = lower
        q99[feature] = upper
        means[feature] = mean
        stds[feature] = std

    scaler = {
        "feature_order": feature_order,
        "normalized_feature_columns": [f"{feature}_normalized" for feature in feature_order],
        "feature_groups": feature_groups(feature_order),
        "transforms": {feature: "identity" for feature in feature_order},
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
    """Group OFR FSI context features for reporting."""

    groups: dict[str, list[str]] = {
        "ofr_fsi_trailing": [],
        "ofr_fsi_categories": [],
    }
    for feature in feature_order:
        if feature in CATEGORY_FEATURES:
            groups["ofr_fsi_categories"].append(feature)
        else:
            groups["ofr_fsi_trailing"].append(feature)
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
    fsi: pd.DataFrame,
    fsi_source: str,
    btc_source: str,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    asof_lag_days: int,
) -> dict[str, Any]:
    """Build compact source/feature audit metadata."""

    feature_order = list(scaler["feature_order"])
    normalized_columns = scaler["normalized_feature_columns"]
    split_counts = {
        str(split): int(count)
        for split, count in context_table["split"].value_counts(sort=False).items()
    }
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
    finite_normalized = bool(
        np.isfinite(context_table[normalized_columns].to_numpy(dtype="float64")).all()
    )
    if not finite_normalized:
        warnings.append("normalized OFR FSI feature matrix contains non-finite values")

    return {
        "status": "ok",
        "stage": "4-N13-1",
        "context_source": "ofr_fsi_risk_off_proxy",
        "terminology": {
            "ofr_fsi": "official financial-stress / risk-off proxy, not direct RORO",
            "risk_off_direction": "higher OFR FSI means higher financial stress",
            "btc_direction": "not hard-coded; learned by context-FiLM",
        },
        "source": {
            "ofr_fsi_source": str(fsi_source),
            "btc_source": str(btc_source),
            "ofr_fsi_date_min": date_or_none(fsi["ofr_date"].min()),
            "ofr_fsi_date_max": date_or_none(fsi["ofr_date"].max()),
            "ofr_fsi_rows": int(len(fsi)),
        },
        "alignment": {
            "asof_lag_days": int(asof_lag_days),
            "policy": "use latest OFR FSI source date <= BTC image end date minus lag",
            "same_day_policy": "disabled by default through asof_lag_days=1",
            "max_age_days": safe_float(pd.to_numeric(context_table["ofr_age_days"], errors="coerce").max()),
            "mean_age_days": safe_float(pd.to_numeric(context_table["ofr_age_days"], errors="coerce").mean()),
        },
        "image_window": int(image_window),
        "image_spec": str(image_spec),
        "return_horizon": int(return_horizon),
        "context_dim": int(len(feature_order)),
        "feature_order": feature_order,
        "normalized_feature_columns": normalized_columns,
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
    groups = feature_groups(feature_order)
    reverse_group = {
        feature: group
        for group, features in groups.items()
        for feature in features
    }
    for split, split_frame in context_table.groupby("split", sort=True):
        for feature in feature_order:
            raw = pd.to_numeric(split_frame[feature], errors="coerce")
            normalized = pd.to_numeric(split_frame[f"{feature}_normalized"], errors="coerce")
            rows.append(
                {
                    "split": str(split),
                    "feature": feature,
                    "feature_group": reverse_group.get(feature, "other"),
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
        "stage": "4-N13-1",
        "context_source": "ofr_fsi_risk_off_proxy",
        "context_dim": audit["context_dim"],
        "feature_order": scaler["feature_order"],
        "normalized_feature_columns": scaler["normalized_feature_columns"],
        "feature_group_counts": audit["feature_group_counts"],
        "split_counts": audit["split_counts"],
        "fit_on": "train",
        "normalization": scaler["normalization"],
        "alignment": audit["alignment"],
        "terminology": audit["terminology"],
    }


def make_fsi_context_output_name(
    *,
    output_prefix: str,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    feature_set_name: str,
) -> str:
    """Return filesystem-safe Stage 4 FSI context artifact name."""

    suffix = sanitize_name_suffix(feature_set_name)
    base = f"{output_prefix}_i{int(image_window)}_{image_spec}_r{int(return_horizon)}"
    return f"{base}_{suffix}" if suffix else base


def write_parquet_if_available(frame: pd.DataFrame, path: Path) -> bool:
    """Write parquet when pandas has a parquet engine; otherwise keep CSV only."""

    try:
        frame.to_parquet(path, index=False)
    except Exception:
        return False
    return True


def safe_float(value: Any) -> float | None:
    """Return None for NaN-like values, otherwise float."""

    if pd.isna(value):
        return None
    return float(value)


def date_or_none(value: Any) -> str | None:
    """Return ISO date string or None."""

    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


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
