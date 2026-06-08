"""Build Stage 4 derivatives/leverage context features.

This is the 4-N16-1 context builder. It converts already-downloaded BitMEX
funding/activity and CFTC/CME Bitcoin futures COT data into a model-ready
prebuilt Stage 4 context artifact:

    context_features.csv
    context_scaler.json
    context_feature_audit.json
    context_feature_summary.csv

The context is aligned to each BTC chart sample by image end date. BitMEX daily
features use a conservative lag by default, and CFTC features use the
release-lagged daily file prepared in data_inventory.

Marker: N16 derivatives leverage context feature builder
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


SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
DEFAULT_OUTPUT_PREFIX = "stage4_derivatives_context"
DEFAULT_FEATURE_SET = "derivatives_all"

FUNDING_WINDOWS = (7, 20, 60)
ACTIVITY_WINDOWS = (7, 20, 60)
CFTC_WINDOWS = (20, 60)

FUNDING_BASE_FEATURES = (
    "funding_rate_mean",
    "funding_rate_sum",
    "funding_rate_abs_mean",
    "funding_rate_min",
    "funding_rate_max",
)
ACTIVITY_BASE_FEATURES = (
    "trades",
    "volume",
    "turnover",
    "homeNotional",
    "foreignNotional",
)
CFTC_CURRENT_FEATURES = (
    "cot_open_interest",
    "cot_lev_money_net_ratio",
    "cot_asset_mgr_net_ratio",
    "cot_dealer_net_ratio",
    "cot_other_rept_net_ratio",
    "cot_nonrept_net_ratio",
    "cot_age_days",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=None)
    parser.add_argument("--image-spec", default=None)
    parser.add_argument("--return-horizon", type=int, default=None)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    parser.add_argument("--feature-set-name", default="")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--daily-asof-lag-days", type=int, default=1)
    parser.add_argument("--clip-lower-quantile", type=float, default=0.01)
    parser.add_argument("--clip-upper-quantile", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0e-8)
    parser.add_argument("--funding-csv", type=Path, default=None)
    parser.add_argument("--activity-csv", type=Path, default=None)
    parser.add_argument("--cftc-csv", type=Path, default=None)
    parser.add_argument("--write-report-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
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

    project_root = paths.project_root
    source_paths = resolve_source_paths(
        project_root=project_root,
        funding_csv=args.funding_csv,
        activity_csv=args.activity_csv,
        cftc_csv=args.cftc_csv,
    )
    sources = load_derivatives_sources(source_paths)
    raw_context = build_derivatives_sample_context(
        samples=sample_table,
        funding=sources["funding"],
        activity=sources["activity"],
        cftc=sources["cftc"],
        daily_asof_lag_days=int(args.daily_asof_lag_days),
    )
    feature_order = select_feature_order(str(args.feature_set))
    missing = [feature for feature in feature_order if feature not in raw_context.columns]
    if missing:
        raise KeyError("Selected derivatives feature(s) missing: " + ", ".join(missing))

    context_table, scaler = fit_transform_derivatives_context_features(
        raw_context,
        feature_order=feature_order,
        lower_q=float(args.clip_lower_quantile),
        upper_q=float(args.clip_upper_quantile),
        epsilon=float(args.epsilon),
    )
    context_name = make_derivatives_context_output_name(
        output_prefix=str(args.output_prefix),
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        feature_set_name=str(args.feature_set_name or args.feature_set),
    )
    audit = build_audit(
        context_table,
        scaler=scaler,
        context_name=context_name,
        feature_set=str(args.feature_set),
        source_paths=source_paths,
        btc_source=str(btc_source),
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        daily_asof_lag_days=int(args.daily_asof_lag_days),
    )
    summary = build_feature_summary(context_table, feature_order)
    manifest = build_manifest(audit, scaler)

    output_dir = paths.context_root / context_name / f"seed_{int(args.run_seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_root.mkdir(parents=True, exist_ok=True)

    context_csv = output_dir / "context_features.csv"
    context_parquet = output_dir / "context_features.parquet"
    scaler_path = output_dir / "context_scaler.json"
    audit_path = output_dir / "context_feature_audit.json"
    summary_path = output_dir / "context_feature_summary.csv"
    manifest_path = output_dir / "derivatives_context_manifest.json"

    context_table.to_csv(context_csv, index=False)
    parquet_written = write_parquet_if_available(context_table, context_parquet)
    scaler_path.write_text(json.dumps(_jsonable(scaler), indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
    summary.to_csv(summary_path, index=False)
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")

    report_files: dict[str, str] = {}
    if args.write_report_copy:
        report_prefix = f"{context_name}_seed{int(args.run_seed)}"
        report_audit = paths.tables_root / f"{report_prefix}_derivatives_context_feature_audit.json"
        report_summary = paths.tables_root / f"{report_prefix}_derivatives_context_feature_summary.csv"
        report_manifest = paths.tables_root / f"{report_prefix}_derivatives_context_manifest.json"
        report_audit.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
        summary.to_csv(report_summary, index=False)
        report_manifest.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
        report_files = {
            "report_audit": str(report_audit),
            "report_summary": str(report_summary),
            "report_manifest": str(report_manifest),
        }

    print(
        json.dumps(
            _jsonable(
                {
                    "status": "ok",
                    "stage": "4-N16-1",
                    "context_name": context_name,
                    "feature_set": str(args.feature_set),
                    "context_dim": len(feature_order),
                    "image_window": image_window,
                    "image_spec": image_spec,
                    "return_horizon": return_horizon,
                    "run_seed": int(args.run_seed),
                    "split_counts": audit["split_counts"],
                    "missing_warnings": audit["warnings"],
                    "written": {
                        "context_features": str(context_csv),
                        "context_features_parquet": str(context_parquet) if parquet_written else None,
                        "context_scaler": str(scaler_path),
                        "context_feature_audit": str(audit_path),
                        "context_feature_summary": str(summary_path),
                        "derivatives_context_manifest": str(manifest_path),
                        **report_files,
                    },
                }
            ),
            indent=2,
        )
    )


def validate_args(args: argparse.Namespace) -> None:
    if int(args.daily_asof_lag_days) < 0:
        raise ValueError("--daily-asof-lag-days must be >= 0.")
    if not 0.0 <= float(args.clip_lower_quantile) <= 1.0:
        raise ValueError("--clip-lower-quantile must be in [0, 1].")
    if not 0.0 <= float(args.clip_upper_quantile) <= 1.0:
        raise ValueError("--clip-upper-quantile must be in [0, 1].")
    if float(args.clip_lower_quantile) > float(args.clip_upper_quantile):
        raise ValueError("--clip-lower-quantile must be <= --clip-upper-quantile.")
    if float(args.epsilon) <= 0.0:
        raise ValueError("--epsilon must be positive.")
    select_feature_order(str(args.feature_set))


def resolve_source_paths(
    *,
    project_root: Path,
    funding_csv: Path | None,
    activity_csv: Path | None,
    cftc_csv: Path | None,
) -> dict[str, Path]:
    base = project_root / "data_inventory" / "crypto_derivatives"
    result = {
        "funding": funding_csv
        or base / "bitmex_xbtusd" / "processed" / "bitmex_xbtusd_funding_daily_2018_2024.csv",
        "activity": activity_csv
        or base / "bitmex_xbtusd" / "processed" / "bitmex_xbtusd_derivatives_activity_daily_2018_2024.csv",
        "cftc": cftc_csv
        or base
        / "cftc_cme_bitcoin_cot"
        / "processed"
        / "cftc_cme_bitcoin_main_plus_micro_cot_daily_release_lag3_ffill_2018_2024.csv",
    }
    missing = [f"{key}: {path}" for key, path in result.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing derivatives source file(s): " + "; ".join(missing))
    return {key: Path(path).expanduser() for key, path in result.items()}


def load_derivatives_sources(source_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {
        "funding": load_funding_daily(source_paths["funding"]),
        "activity": load_activity_daily(source_paths["activity"]),
        "cftc": load_cftc_daily(source_paths["cftc"]),
    }


def load_funding_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Date", *FUNDING_BASE_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Funding table missing columns: {missing}")
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    for column in FUNDING_BASE_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def load_activity_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Date", *ACTIVITY_BASE_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Activity table missing columns: {missing}")
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    for column in ACTIVITY_BASE_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def load_cftc_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename = {
        "Open_Interest_All": "cot_open_interest",
        "lev_money_net_ratio_all": "cot_lev_money_net_ratio",
        "asset_mgr_net_ratio_all": "cot_asset_mgr_net_ratio",
        "dealer_net_ratio_all": "cot_dealer_net_ratio",
        "other_rept_net_ratio_all": "cot_other_rept_net_ratio",
        "nonrept_net_ratio_all": "cot_nonrept_net_ratio",
    }
    required = {"Date", "cot_source_report_date", "cot_available_date", "cot_age_days", *rename}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"CFTC table missing columns: {missing}")
    frame = frame.rename(columns=rename)
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    frame["cot_source_report_date"] = pd.to_datetime(frame["cot_source_report_date"]).dt.normalize()
    frame["cot_available_date"] = pd.to_datetime(frame["cot_available_date"]).dt.normalize()
    for column in CFTC_CURRENT_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    keep = ["Date", "cot_source_report_date", "cot_available_date", *CFTC_CURRENT_FEATURES]
    return frame.loc[:, keep].sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def build_sample_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
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


def build_derivatives_sample_context(
    *,
    samples: pd.DataFrame,
    funding: pd.DataFrame,
    activity: pd.DataFrame,
    cftc: pd.DataFrame,
    daily_asof_lag_days: int,
) -> pd.DataFrame:
    sample_min = pd.to_datetime(samples["Date"]).min().normalize()
    sample_max = pd.to_datetime(samples["Date"]).max().normalize()
    calendar = pd.DataFrame({"Date": pd.date_range(sample_min, sample_max, freq="D")})
    calendar["derivatives_asof_date"] = (
        calendar["Date"] - pd.to_timedelta(int(daily_asof_lag_days), unit="D")
    )

    funding_features = add_funding_features(funding)
    activity_features = add_activity_features(activity)
    cftc_features = add_cftc_features(cftc)

    aligned = pd.merge_asof(
        calendar.sort_values("derivatives_asof_date"),
        funding_features.sort_values("Date").rename(columns={"Date": "funding_source_date"}),
        left_on="derivatives_asof_date",
        right_on="funding_source_date",
        direction="backward",
    ).sort_values("Date")
    aligned = pd.merge_asof(
        aligned.sort_values("derivatives_asof_date"),
        activity_features.sort_values("Date").rename(columns={"Date": "activity_source_date"}),
        left_on="derivatives_asof_date",
        right_on="activity_source_date",
        direction="backward",
    ).sort_values("Date")
    aligned = aligned.merge(cftc_features, on="Date", how="left")

    aligned["funding_age_days"] = (aligned["Date"] - aligned["funding_source_date"]).dt.days
    aligned["activity_age_days"] = (aligned["Date"] - aligned["activity_source_date"]).dt.days
    aligned["funding_missing"] = aligned["funding_rate_mean_7"].isna()
    aligned["activity_missing"] = aligned["bitmex_volume_mean_7"].isna()
    aligned["cftc_missing"] = aligned["cot_open_interest"].isna()

    context_columns = [
        "Date",
        "derivatives_asof_date",
        "funding_source_date",
        "activity_source_date",
        "funding_age_days",
        "activity_age_days",
        "funding_missing",
        "activity_missing",
        "cftc_missing",
        "cot_source_report_date",
        "cot_available_date",
        *all_derivatives_features(),
    ]
    context_columns = [column for column in context_columns if column in aligned.columns]
    context = aligned.loc[:, context_columns].copy()

    table = samples.merge(context, on="Date", how="left")
    table["image_end_date"] = pd.to_datetime(table["Date"]).dt.date.astype(str)
    for column in ("derivatives_asof_date", "funding_source_date", "activity_source_date", "cot_source_report_date", "cot_available_date"):
        if column in table.columns:
            table[column] = pd.to_datetime(table[column]).dt.date.astype("string")
    if "label_end_date" in table.columns:
        table["label_end_date"] = pd.to_datetime(table["label_end_date"]).dt.date.astype(str)

    feature_order = all_derivatives_features()
    for feature in feature_order:
        if feature in table.columns:
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
        "derivatives_asof_date",
        "funding_source_date",
        "activity_source_date",
        "cot_source_report_date",
        "cot_available_date",
        "funding_age_days",
        "activity_age_days",
        "cot_age_days",
        "funding_missing",
        "activity_missing",
        "cftc_missing",
    ]
    ordered = (
        [column for column in metadata if column in table.columns]
        + [column for column in feature_order if column in table.columns]
        + [f"{feature}_missing" for feature in feature_order if f"{feature}_missing" in table.columns]
        + [column for column in table.columns if column not in set(metadata + feature_order)]
    )
    return table.loc[:, list(dict.fromkeys(ordered))].reset_index(drop=True)


def add_funding_features(funding: pd.DataFrame) -> pd.DataFrame:
    frame = funding.sort_values("Date").reset_index(drop=True).copy()
    for window in FUNDING_WINDOWS:
        frame[f"funding_rate_mean_{window}"] = (
            frame["funding_rate_mean"].rolling(window, min_periods=window).mean()
        )
        frame[f"funding_rate_sum_{window}"] = (
            frame["funding_rate_sum"].rolling(window, min_periods=window).sum()
        )
        frame[f"funding_rate_abs_mean_{window}"] = (
            frame["funding_rate_abs_mean"].rolling(window, min_periods=window).mean()
        )
        frame[f"funding_rate_min_{window}"] = (
            frame["funding_rate_min"].rolling(window, min_periods=window).min()
        )
        frame[f"funding_rate_max_{window}"] = (
            frame["funding_rate_max"].rolling(window, min_periods=window).max()
        )
    keep = ["Date", *funding_features()]
    return frame.loc[:, keep]


def add_activity_features(activity: pd.DataFrame) -> pd.DataFrame:
    frame = activity.sort_values("Date").reset_index(drop=True).copy()
    for column in ACTIVITY_BASE_FEATURES:
        safe = to_safe_feature_token(column)
        windows = ACTIVITY_WINDOWS if column in {"trades", "volume", "turnover"} else (20, 60)
        for window in windows:
            frame[f"bitmex_{safe}_mean_{window}"] = (
                frame[column].rolling(window, min_periods=window).mean()
            )
    keep = ["Date", *activity_features()]
    return frame.loc[:, keep]


def add_cftc_features(cftc: pd.DataFrame) -> pd.DataFrame:
    frame = cftc.sort_values("Date").reset_index(drop=True).copy()
    oi = pd.to_numeric(frame["cot_open_interest"], errors="coerce")
    for window in CFTC_WINDOWS:
        frame[f"cot_open_interest_change_{window}"] = oi - oi.shift(window)
        shifted = oi.shift(window).replace(0.0, np.nan)
        frame[f"cot_open_interest_pct_change_{window}"] = (oi / shifted) - 1.0
    rolling_mean = oi.rolling(60, min_periods=60).mean()
    rolling_std = oi.rolling(60, min_periods=60).std(ddof=0).replace(0.0, np.nan)
    frame["cot_open_interest_zscore_60"] = (oi - rolling_mean) / rolling_std

    for ratio_feature in (
        "cot_lev_money_net_ratio",
        "cot_asset_mgr_net_ratio",
        "cot_dealer_net_ratio",
    ):
        values = pd.to_numeric(frame[ratio_feature], errors="coerce")
        for window in CFTC_WINDOWS:
            frame[f"{ratio_feature}_change_{window}"] = values - values.shift(window)

    keep = [
        "Date",
        "cot_source_report_date",
        "cot_available_date",
        *cftc_features(),
    ]
    return frame.loc[:, keep]


def select_feature_order(feature_set: str) -> list[str]:
    key = sanitize_name_suffix(feature_set) or DEFAULT_FEATURE_SET
    mapping = {
        "funding_only": funding_features(),
        "funding_plus_cftc_oi": funding_features() + cftc_features(),
        "funding_plus_activity": funding_features() + activity_features(),
        "funding_plus_activity_plus_cftc_oi": funding_features() + activity_features() + cftc_features(),
        "derivatives_all": funding_features() + activity_features() + cftc_features(),
        "all": funding_features() + activity_features() + cftc_features(),
        "cftc_oi_only": cftc_features(),
        "activity_only": activity_features(),
    }
    if key not in mapping:
        available = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported --feature-set={feature_set!r}. Available: {available}")
    return list(dict.fromkeys(mapping[key]))


def funding_features() -> list[str]:
    result: list[str] = []
    for window in FUNDING_WINDOWS:
        for feature in FUNDING_BASE_FEATURES:
            result.append(f"{feature}_{window}")
    return result


def activity_features() -> list[str]:
    result: list[str] = []
    for column in ACTIVITY_BASE_FEATURES:
        safe = to_safe_feature_token(column)
        windows = ACTIVITY_WINDOWS if column in {"trades", "volume", "turnover"} else (20, 60)
        for window in windows:
            result.append(f"bitmex_{safe}_mean_{window}")
    return result


def cftc_features() -> list[str]:
    result = list(CFTC_CURRENT_FEATURES)
    for window in CFTC_WINDOWS:
        result.append(f"cot_open_interest_change_{window}")
        result.append(f"cot_open_interest_pct_change_{window}")
    result.append("cot_open_interest_zscore_60")
    for ratio_feature in (
        "cot_lev_money_net_ratio",
        "cot_asset_mgr_net_ratio",
        "cot_dealer_net_ratio",
    ):
        for window in CFTC_WINDOWS:
            result.append(f"{ratio_feature}_change_{window}")
    return result


def all_derivatives_features() -> list[str]:
    return list(dict.fromkeys(funding_features() + activity_features() + cftc_features()))


def to_safe_feature_token(value: str) -> str:
    return sanitize_name_suffix(value).replace("home_notional", "home_notional").replace(
        "foreign_notional",
        "foreign_notional",
    )


def fit_transform_derivatives_context_features(
    frame: pd.DataFrame,
    *,
    feature_order: list[str],
    lower_q: float,
    upper_q: float,
    epsilon: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    context = frame.copy()
    train_mask = context["split"].astype(str).eq("train")
    if not train_mask.any():
        raise ValueError("Cannot fit derivatives context scaler because split == train is empty.")

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
            raise ValueError(f"Cannot fit all-missing derivatives feature: {feature}")
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
    groups: dict[str, list[str]] = {
        "bitmex_funding": [],
        "bitmex_activity": [],
        "cftc_cme_oi_positioning": [],
    }
    for feature in feature_order:
        if feature.startswith("funding_"):
            groups["bitmex_funding"].append(feature)
        elif feature.startswith("bitmex_"):
            groups["bitmex_activity"].append(feature)
        elif feature.startswith("cot_"):
            groups["cftc_cme_oi_positioning"].append(feature)
        else:
            groups.setdefault("other", []).append(feature)
    return groups


def missing_rates(frame: pd.DataFrame, feature_order: list[str]) -> dict[str, dict[str, float]]:
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
    context_name: str,
    feature_set: str,
    source_paths: dict[str, Path],
    btc_source: str,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    daily_asof_lag_days: int,
) -> dict[str, Any]:
    feature_order = list(scaler["feature_order"])
    normalized_columns = list(scaler["normalized_feature_columns"])
    split_counts = {
        str(split): int(count)
        for split, count in context_table["split"].value_counts(sort=False).items()
    }
    warnings: list[str] = []
    for split, rates in scaler["missing_rate_by_split"].items():
        high = [feature for feature, rate in rates.items() if float(rate) > 0.05]
        if high:
            warnings.append(f"{split}: high raw missing rate >5%: " + ", ".join(high[:12]))
    finite_normalized = bool(
        np.isfinite(context_table[normalized_columns].to_numpy(dtype="float64")).all()
    )
    if not finite_normalized:
        warnings.append("normalized derivatives feature matrix contains non-finite values")

    age_columns = [
        column
        for column in ("funding_age_days", "activity_age_days", "cot_age_days")
        if column in context_table.columns
    ]
    age_summary = {
        column: {
            "max": safe_float(pd.to_numeric(context_table[column], errors="coerce").max()),
            "mean": safe_float(pd.to_numeric(context_table[column], errors="coerce").mean()),
        }
        for column in age_columns
    }

    return {
        "status": "ok",
        "stage": "4-N16-1",
        "context_name": context_name,
        "context_source": "crypto_derivatives_leverage",
        "feature_set": str(feature_set),
        "terminology": {
            "funding": "perpetual swap funding pressure / long-short crowding proxy",
            "activity": "BitMEX futures trading activity and notional participation",
            "cftc": "release-lagged CFTC/CME Bitcoin futures open interest and positioning",
            "btc_direction": "not hard-coded; learned by context-FiLM",
        },
        "source": {
            "btc_source": str(btc_source),
            **{f"{key}_source": str(path) for key, path in source_paths.items()},
        },
        "alignment": {
            "daily_asof_lag_days": int(daily_asof_lag_days),
            "daily_policy": "use latest BitMEX daily source date <= BTC image end date minus lag",
            "cftc_policy": "use release-lagged CFTC daily forward-filled file; no extra lag applied",
            "age_summary": age_summary,
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
    return {
        "status": "ok",
        "stage": "4-N16-1",
        "context_source": audit["context_source"],
        "feature_set": audit["feature_set"],
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


def make_derivatives_context_output_name(
    *,
    output_prefix: str,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    feature_set_name: str,
) -> str:
    suffix = sanitize_name_suffix(feature_set_name)
    base = f"{output_prefix}_i{int(image_window)}_{image_spec}_r{int(return_horizon)}"
    return f"{base}_{suffix}" if suffix else base


def write_parquet_if_available(frame: pd.DataFrame, path: Path) -> bool:
    try:
        frame.to_parquet(path, index=False)
    except Exception:
        return False
    return True


def safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _jsonable(value: Any) -> Any:
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
