"""Build Stage 4 public-data RORO proxy context features.

This is the 4-N13-3 macro-regime context step. It does not replicate the
Kansas City Fed proprietary/full Risk-On Risk-Off index. Instead, it builds a
transparent public-data proxy inspired by the same idea:

    public market stress components -> risk-off aligned z-scores
    train-only PCA first component -> roro_proxy

The sign is fixed so larger values mean stronger risk-off pressure. BTC
direction is not hard-coded; the Stage 4 context-FiLM model learns how the
proxy should modulate the frozen Stage 2 chart features.

Marker: KC Fed-inspired public-data RORO proxy builder
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


FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
DEFAULT_SERIES: dict[str, dict[str, str]] = {
    "vix": {
        "series_id": "VIXCLS",
        "source_url": FRED_BASE_URL.format(series_id="VIXCLS"),
        "description": "CBOE VIX daily close via FRED",
    },
    "sp500": {
        "series_id": "SP500",
        "source_url": FRED_BASE_URL.format(series_id="SP500"),
        "description": "S&P 500 index via FRED",
    },
    "nasdaq": {
        "series_id": "NASDAQCOM",
        "source_url": FRED_BASE_URL.format(series_id="NASDAQCOM"),
        "description": "NASDAQ Composite index via FRED",
    },
    "dollar": {
        "series_id": "DTWEXBGS",
        "source_url": FRED_BASE_URL.format(series_id="DTWEXBGS"),
        "description": "Nominal broad U.S. dollar index via FRED",
    },
    "ten_year": {
        "series_id": "DGS10",
        "source_url": FRED_BASE_URL.format(series_id="DGS10"),
        "description": "10-year Treasury constant maturity rate via FRED",
    },
    "hy_oas": {
        "series_id": "BAMLH0A0HYM2",
        "source_url": FRED_BASE_URL.format(series_id="BAMLH0A0HYM2"),
        "description": "ICE BofA U.S. high-yield option-adjusted spread via FRED",
    },
}
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
RISK_OFF_COMPONENTS = [
    "riskoff_vix_change_20",
    "riskoff_hy_oas_change_20",
    "riskoff_neg_sp500_return_20",
    "riskoff_neg_nasdaq_return_20",
    "riskoff_dollar_return_20",
    "riskoff_neg_10y_yield_change_20",
]
ANCHOR_COMPONENTS = [
    "riskoff_vix_change_20",
    "riskoff_hy_oas_change_20",
    "riskoff_neg_sp500_return_20",
    "riskoff_neg_nasdaq_return_20",
]
RORO_PROXY_FEATURES = [
    "roro_proxy_value",
    "roro_proxy_mean_20",
    "roro_proxy_mean_60",
    "roro_proxy_delta_20",
    "roro_proxy_delta_60",
    "roro_proxy_std_60",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=None)
    parser.add_argument("--image-spec", default=None)
    parser.add_argument("--return-horizon", type=int, default=None)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument(
        "--roro-data-dir",
        default="data_inventory/roro_public/raw",
        help="Folder for cached public source CSV files.",
    )
    parser.add_argument(
        "--refresh-downloads",
        action="store_true",
        help="Re-download official public CSV files even if cached files exist.",
    )
    parser.add_argument(
        "--asof-lag-days",
        type=int,
        default=1,
        help=(
            "Conservative lag between BTC image end date t and the macro value "
            "used for context. Default 1 means use macro data as of t-1."
        ),
    )
    parser.add_argument(
        "--include-component-features",
        action="store_true",
        help="Include normalized risk-off component features beside the RORO proxy.",
    )
    parser.add_argument(
        "--output-prefix",
        default="stage4_roro_context",
        help="Prefix for context artifact and report names.",
    )
    parser.add_argument(
        "--feature-set-name",
        default="public_roro_pca_lag1_w20_60",
        help="Filesystem-safe suffix describing this RORO context feature set.",
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
    """Build normalized RORO proxy context artifacts."""

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

    source_dir = resolve_roro_data_dir(Path(args.roro_data_dir), config_path=Path(args.config))
    macro_sources = load_public_macro_sources(
        source_dir=source_dir,
        refresh_downloads=bool(args.refresh_downloads),
    )
    macro_components = build_risk_off_components(macro_sources)
    raw_context, pca_metadata = build_roro_sample_context(
        samples=sample_table,
        macro_components=macro_components,
        asof_lag_days=int(args.asof_lag_days),
        epsilon=float(args.epsilon),
    )
    component_columns = list(pca_metadata["component_columns"])
    feature_order = RORO_PROXY_FEATURES + (
        component_columns if bool(args.include_component_features) else []
    )
    context_table, scaler = fit_transform_roro_context_features(
        raw_context,
        feature_order=feature_order,
        lower_q=float(args.clip_lower_quantile),
        upper_q=float(args.clip_upper_quantile),
        epsilon=float(args.epsilon),
    )
    audit = build_audit(
        context_table,
        scaler=scaler,
        pca_metadata=pca_metadata,
        source_manifest=build_source_manifest(macro_sources, source_dir),
        btc_source=str(btc_source),
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        asof_lag_days=int(args.asof_lag_days),
    )
    summary = build_feature_summary(context_table, feature_order)

    context_name = make_roro_context_output_name(
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
    manifest_path = output_dir / "roro_context_manifest.json"

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
        report_audit = paths.tables_root / f"{report_prefix}_roro_context_feature_audit.json"
        report_summary = paths.tables_root / f"{report_prefix}_roro_context_feature_summary.csv"
        report_manifest = paths.tables_root / f"{report_prefix}_roro_context_manifest.json"
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
                "stage": "4-N13-3",
                "context_name": context_name,
                "context_source": "public_data_roro_proxy",
                "image_window": image_window,
                "image_spec": image_spec,
                "return_horizon": return_horizon,
                "run_seed": int(args.run_seed),
                "asof_lag_days": int(args.asof_lag_days),
                "context_dim": int(len(feature_order)),
                "split_counts": audit["split_counts"],
                "pca_explained_variance_ratio": pca_metadata["explained_variance_ratio"],
                "missing_warnings": audit["warnings"],
                "written": {
                    "context_features": str(context_csv),
                    "context_features_parquet": str(context_parquet) if parquet_written else None,
                    "context_scaler": str(scaler_path),
                    "context_feature_audit": str(audit_path),
                    "context_feature_summary": str(summary_path),
                    "roro_context_manifest": str(manifest_path),
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


def resolve_roro_data_dir(path: Path, *, config_path: Path) -> Path:
    """Resolve a relative RORO data directory against the Stage 4 project root."""

    if path.is_absolute():
        return path
    config_parent = config_path.expanduser().resolve().parent
    project_candidate = config_parent.parent / path
    return project_candidate.resolve()


def load_public_macro_sources(
    *,
    source_dir: Path,
    refresh_downloads: bool,
) -> dict[str, pd.DataFrame]:
    """Load cached official public CSV files, downloading missing files."""

    source_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, pd.DataFrame] = {}
    required_sources = {"vix", "sp500", "ten_year"}
    for name, metadata in DEFAULT_SERIES.items():
        series_id = metadata["series_id"]
        cache_path = source_dir / f"{series_id}.csv"
        if refresh_downloads or not cache_path.exists():
            if not refresh_downloads and not cache_path.exists():
                continue
            frame = pd.read_csv(metadata["source_url"])
            frame.to_csv(cache_path, index=False)
        if not cache_path.exists():
            continue
        frame = pd.read_csv(cache_path)
        sources[name] = clean_fred_series(
            frame,
            series_id=series_id,
            output_column=name,
            cache_path=cache_path,
        )
    missing_required = sorted(required_sources.difference(sources))
    if missing_required:
        missing_text = ", ".join(missing_required)
        raise FileNotFoundError(
            "Missing required RORO public source cache(s): "
            f"{missing_text}. Place CSV files in {source_dir} or run with --refresh-downloads."
        )
    write_source_readme(source_dir)
    return sources


def clean_fred_series(
    frame: pd.DataFrame,
    *,
    series_id: str,
    output_column: str,
    cache_path: Path,
) -> pd.DataFrame:
    """Clean one FRED fredgraph CSV."""

    date_column = "observation_date"
    if date_column not in frame.columns:
        if "DATE" in frame.columns:
            date_column = "DATE"
        elif "Date" in frame.columns:
            date_column = "Date"
        else:
            raise KeyError(f"{cache_path} has no observation_date/DATE/Date column.")
    value_column = series_id if series_id in frame.columns else None
    if value_column is None:
        candidates = [column for column in frame.columns if column != date_column]
        if len(candidates) != 1:
            raise KeyError(f"{cache_path} cannot infer value column for {series_id}.")
        value_column = candidates[0]

    result = frame.loc[:, [date_column, value_column]].copy()
    result.columns = ["macro_date", output_column]
    result["macro_date"] = pd.to_datetime(result["macro_date"]).dt.normalize()
    result[output_column] = pd.to_numeric(
        result[output_column].replace(".", np.nan),
        errors="coerce",
    )
    result = result.sort_values("macro_date").drop_duplicates("macro_date", keep="last")
    if result.empty:
        raise ValueError(f"{cache_path} is empty after cleaning.")
    return result.reset_index(drop=True)


def write_source_readme(source_dir: Path) -> None:
    """Write a compact source note beside cached public macro CSV files."""

    readme = source_dir.parent / "README.md"
    lines = [
        "# Public RORO Proxy Source Cache",
        "",
        "These files are cached for Stage 4 N13-3 so Kaggle runs do not depend on",
        "live public endpoints. The training context is a KC Fed-inspired public-data",
        "RORO proxy, not a full replication of the Kansas City Fed RORO index.",
        "",
        "The official KC Fed daily/weekly files are kept in `kc_fed_official/` for",
        "terminology and source audit. They currently start in June 2023, so the",
        "training proxy uses longer-history public source caches from `raw/`.",
        "",
        "| Local file | Series | Cached | Source URL | Interpretation |",
        "| --- | --- | --- | --- | --- |",
    ]
    interpretation = {
        "VIXCLS": "VIX up is risk-off pressure.",
        "SP500": "Negative S&P 500 return is risk-off pressure.",
        "NASDAQCOM": "Negative NASDAQ return is risk-off pressure.",
        "DTWEXBGS": "Dollar strength is treated as risk-off pressure.",
        "DGS10": "Falling 10-year yield is treated as risk-off pressure with caution.",
        "BAMLH0A0HYM2": "High-yield spread widening is risk-off pressure.",
    }
    source_urls = {
        "VIXCLS": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
        "SP500": DEFAULT_SERIES["sp500"]["source_url"],
        "NASDAQCOM": DEFAULT_SERIES["nasdaq"]["source_url"],
        "DTWEXBGS": DEFAULT_SERIES["dollar"]["source_url"],
        "DGS10": (
            "https://home.treasury.gov/resource-center/data-chart-center/"
            "interest-rates/pages/xml?data=daily_treasury_yield_curve"
        ),
        "BAMLH0A0HYM2": DEFAULT_SERIES["hy_oas"]["source_url"],
    }
    for metadata in DEFAULT_SERIES.values():
        series_id = metadata["series_id"]
        cache_path = source_dir / f"{series_id}.csv"
        cached = "yes" if cache_path.exists() else "optional/missing"
        lines.append(
            f"| `raw/{series_id}.csv` | `{series_id}` | "
            f"{cached} | {source_urls[series_id]} | {interpretation[series_id]} |"
        )
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_source_manifest(
    sources: dict[str, pd.DataFrame],
    source_dir: Path,
) -> dict[str, Any]:
    """Build a JSON-safe manifest of public source data."""

    rows: dict[str, Any] = {}
    for name, frame in sources.items():
        metadata = DEFAULT_SERIES[name]
        series_id = metadata["series_id"]
        rows[name] = {
            "series_id": series_id,
            "description": metadata["description"],
            "source_url": metadata["source_url"],
            "cache_path": str(source_dir / f"{series_id}.csv"),
            "date_min": date_or_none(frame["macro_date"].min()),
            "date_max": date_or_none(frame["macro_date"].max()),
            "rows": int(len(frame)),
            "non_missing": int(pd.to_numeric(frame[name], errors="coerce").notna().sum()),
        }
    return rows


def build_risk_off_components(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create daily risk-off aligned market components from public sources."""

    merged: pd.DataFrame | None = None
    for name, frame in sources.items():
        if merged is None:
            merged = frame.copy()
        else:
            merged = merged.merge(frame, on="macro_date", how="outer")
    if merged is None or merged.empty:
        raise ValueError("No macro source data loaded.")

    merged = merged.sort_values("macro_date").reset_index(drop=True)
    calendar = pd.DataFrame(
        {
            "macro_date": pd.date_range(
                pd.to_datetime(merged["macro_date"]).min(),
                pd.to_datetime(merged["macro_date"]).max(),
                freq="D",
            )
        }
    )
    # Public market sources have different holiday calendars. Use only past
    # observations by forward-filling each source before computing changes.
    daily = calendar.merge(merged, on="macro_date", how="left").sort_values("macro_date")
    for column in ["vix", "sp500", "nasdaq", "dollar", "ten_year", "hy_oas"]:
        if column in daily.columns:
            daily[column] = pd.to_numeric(daily[column], errors="coerce").ffill()

    if "vix" in daily.columns:
        daily["riskoff_vix_change_20"] = daily["vix"] - daily["vix"].shift(20)
    if "hy_oas" in daily.columns:
        daily["riskoff_hy_oas_change_20"] = daily["hy_oas"] - daily["hy_oas"].shift(20)
    if "sp500" in daily.columns:
        daily["riskoff_neg_sp500_return_20"] = -safe_log_return(daily["sp500"], 20)
    if "nasdaq" in daily.columns:
        daily["riskoff_neg_nasdaq_return_20"] = -safe_log_return(daily["nasdaq"], 20)
    if "dollar" in daily.columns:
        daily["riskoff_dollar_return_20"] = safe_log_return(daily["dollar"], 20)
    if "ten_year" in daily.columns:
        daily["riskoff_neg_10y_yield_change_20"] = -(daily["ten_year"] - daily["ten_year"].shift(20))

    keep_columns = [
        column
        for column in [
            "macro_date",
            "vix",
            "sp500",
            "nasdaq",
            "dollar",
            "ten_year",
            "hy_oas",
            *RISK_OFF_COMPONENTS,
        ]
        if column in daily.columns
    ]
    return daily.loc[:, keep_columns].reset_index(drop=True)


def safe_log_return(series: pd.Series, window: int) -> pd.Series:
    """Compute log return and return NaN for non-positive inputs."""

    values = pd.to_numeric(series, errors="coerce")
    previous = values.shift(int(window))
    valid = values.gt(0) & previous.gt(0)
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    result.loc[valid] = np.log(values.loc[valid] / previous.loc[valid])
    return result


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


def build_roro_sample_context(
    *,
    samples: pd.DataFrame,
    macro_components: pd.DataFrame,
    asof_lag_days: int,
    epsilon: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align public macro components and source-level RORO proxy to BTC samples."""

    sample_min = pd.to_datetime(samples["Date"]).min().normalize()
    sample_max = pd.to_datetime(samples["Date"]).max().normalize()
    calendar = pd.DataFrame({"Date": pd.date_range(sample_min, sample_max, freq="D")})
    calendar["macro_asof_date"] = calendar["Date"] - pd.to_timedelta(int(asof_lag_days), unit="D")

    aligned_components = pd.merge_asof(
        calendar.sort_values("macro_asof_date"),
        macro_components.sort_values("macro_date"),
        left_on="macro_asof_date",
        right_on="macro_date",
        direction="backward",
    ).sort_values("Date")

    available_components = [column for column in RISK_OFF_COMPONENTS if column in macro_components.columns]
    sample_aligned = samples.loc[:, ["sample_index", "split", "Date"]].merge(
        aligned_components.loc[:, ["Date", "macro_date", *available_components]],
        on="Date",
        how="left",
    )
    train_mask = sample_aligned["split"].astype(str).eq("train")
    pca_fit = fit_train_only_pca(
        sample_aligned,
        train_mask=train_mask,
        component_columns=available_components,
        anchor_columns=[column for column in ANCHOR_COMPONENTS if column in available_components],
        epsilon=epsilon,
    )
    component_columns = list(pca_fit["metadata"]["component_columns"])

    macro_with_proxy = apply_pca_proxy_to_macro_components(macro_components, pca_fit)
    macro_with_proxy = add_source_level_roro_features(macro_with_proxy)

    aligned_proxy = pd.merge_asof(
        calendar.sort_values("macro_asof_date"),
        macro_with_proxy.sort_values("macro_date"),
        left_on="macro_asof_date",
        right_on="macro_date",
        direction="backward",
    ).sort_values("Date")
    aligned_proxy["roro_source_date"] = aligned_proxy["macro_date"]
    aligned_proxy["roro_age_days"] = (aligned_proxy["Date"] - aligned_proxy["roro_source_date"]).dt.days
    aligned_proxy["roro_exact_asof_match"] = aligned_proxy["roro_source_date"].eq(
        aligned_proxy["macro_asof_date"]
    )
    aligned_proxy["roro_missing"] = aligned_proxy["roro_proxy_value"].isna()

    context_columns = [
        "Date",
        "macro_asof_date",
        "roro_source_date",
        "roro_age_days",
        "roro_missing",
        "roro_exact_asof_match",
        *RORO_PROXY_FEATURES,
        *component_columns,
    ]
    roro_context = aligned_proxy.loc[:, context_columns].copy()

    table = samples.merge(roro_context, on="Date", how="left")
    table["image_end_date"] = pd.to_datetime(table["Date"]).dt.date.astype(str)
    table["macro_asof_date"] = pd.to_datetime(table["macro_asof_date"]).dt.date.astype("string")
    table["roro_source_date"] = pd.to_datetime(table["roro_source_date"]).dt.date.astype("string")
    if "label_end_date" in table.columns:
        table["label_end_date"] = pd.to_datetime(table["label_end_date"]).dt.date.astype(str)

    feature_order = RORO_PROXY_FEATURES + component_columns
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
        "macro_asof_date",
        "roro_source_date",
        "roro_age_days",
        "roro_missing",
        "roro_exact_asof_match",
    ]
    ordered = (
        [column for column in metadata if column in table.columns]
        + feature_order
        + [f"{feature}_missing" for feature in feature_order]
        + [column for column in table.columns if column not in set(metadata + feature_order)]
    )
    return table.loc[:, list(dict.fromkeys(ordered))].reset_index(drop=True), pca_fit["metadata"]


def fit_train_only_pca(
    frame: pd.DataFrame,
    *,
    train_mask: pd.Series,
    component_columns: list[str],
    anchor_columns: list[str],
    epsilon: float,
) -> dict[str, Any]:
    """Fit train-only scaler and first principal component on risk-off components."""

    if not train_mask.any():
        raise ValueError("Cannot fit RORO PCA because split == train is empty.")
    raw = frame.loc[:, component_columns].apply(pd.to_numeric, errors="coerce")
    train_raw = raw.loc[train_mask].replace([np.inf, -np.inf], np.nan)
    usable_columns = [
        column
        for column in component_columns
        if train_raw[column].notna().any()
    ]
    if len(usable_columns) < 2:
        raise ValueError(
            "RORO PCA needs at least two public components with train-period "
            "coverage. Usable components: " + ", ".join(usable_columns)
        )
    component_columns = usable_columns
    anchor_columns = [column for column in anchor_columns if column in component_columns]
    if not anchor_columns:
        anchor_columns = list(component_columns)
    raw = raw.loc[:, component_columns]
    train_raw = raw.loc[train_mask].replace([np.inf, -np.inf], np.nan)

    medians = train_raw.median()
    q01 = train_raw.fillna(medians).quantile(0.01)
    q99 = train_raw.fillna(medians).quantile(0.99)
    imputed = raw.fillna(medians)
    clipped = imputed.clip(lower=q01, upper=q99, axis=1)
    means = clipped.loc[train_mask].mean()
    stds = clipped.loc[train_mask].std(ddof=0).replace(0.0, np.nan).fillna(1.0)
    z = (clipped - means) / stds.clip(lower=epsilon)
    train_z = z.loc[train_mask].to_numpy(dtype="float64")
    if not np.isfinite(train_z).all():
        raise ValueError("Train RORO PCA matrix contains non-finite values.")

    _, singular_values, vt = np.linalg.svd(train_z, full_matrices=False)
    weights = vt[0, :].astype("float64")
    explained_variance = singular_values**2 / max(train_z.shape[0] - 1, 1)
    total_variance = float(np.sum(explained_variance))
    explained_ratio = (
        float(explained_variance[0] / total_variance)
        if total_variance > 0
        else 0.0
    )

    proxy_train_raw = train_z @ weights
    anchor_index = [component_columns.index(column) for column in anchor_columns]
    anchor = train_z[:, anchor_index].mean(axis=1)
    correlation = safe_corr(proxy_train_raw, anchor)
    sign_multiplier = 1.0
    if correlation is not None and correlation < 0:
        weights = -weights
        proxy_train_raw = -proxy_train_raw
        sign_multiplier = -1.0
        correlation = -correlation
    proxy_mean = float(np.mean(proxy_train_raw))
    proxy_std = float(np.std(proxy_train_raw, ddof=0))
    if not np.isfinite(proxy_std) or proxy_std < epsilon:
        proxy_std = 1.0

    metadata = {
        "method": "train_only_svd_first_component",
        "component_columns": component_columns,
        "anchor_columns": anchor_columns,
        "direction": "higher roro_proxy means stronger risk-off pressure",
        "component_direction_rules": {
            "riskoff_vix_change_20": "VIX up over 20 observations -> risk-off",
            "riskoff_hy_oas_change_20": "high-yield OAS widening over 20 observations -> risk-off",
            "riskoff_neg_sp500_return_20": "S&P 500 falling over 20 observations -> risk-off",
            "riskoff_neg_nasdaq_return_20": "NASDAQ falling over 20 observations -> risk-off",
            "riskoff_dollar_return_20": "broad dollar strengthening over 20 observations -> risk-off",
            "riskoff_neg_10y_yield_change_20": "10Y yield falling over 20 observations -> risk-off with caution",
        },
        "medians": medians.to_dict(),
        "q01": q01.to_dict(),
        "q99": q99.to_dict(),
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "weights": {column: float(weight) for column, weight in zip(component_columns, weights, strict=True)},
        "sign_multiplier": float(sign_multiplier),
        "anchor_correlation_after_sign_fix": correlation,
        "explained_variance_ratio": explained_ratio,
        "proxy_train_mean_raw": proxy_mean,
        "proxy_train_std_raw": proxy_std,
    }
    return {
        "metadata": metadata,
        "medians": medians,
        "q01": q01,
        "q99": q99,
        "means": means,
        "stds": stds,
        "weights": weights,
        "proxy_mean": proxy_mean,
        "proxy_std": proxy_std,
    }


def apply_pca_proxy_to_macro_components(
    macro_components: pd.DataFrame,
    pca_fit: dict[str, Any],
) -> pd.DataFrame:
    """Apply train-fitted PCA proxy to all source macro dates."""

    result = macro_components.copy()
    columns = list(pca_fit["metadata"]["component_columns"])
    raw = result.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    imputed = raw.fillna(pca_fit["medians"])
    clipped = imputed.clip(lower=pca_fit["q01"], upper=pca_fit["q99"], axis=1)
    z = (clipped - pca_fit["means"]) / pca_fit["stds"]
    proxy_raw = z.to_numpy(dtype="float64") @ np.asarray(pca_fit["weights"], dtype="float64")
    result["roro_proxy_value"] = (proxy_raw - float(pca_fit["proxy_mean"])) / float(pca_fit["proxy_std"])
    return result


def add_source_level_roro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing RORO proxy features on full source history before merge."""

    result = macro.sort_values("macro_date").reset_index(drop=True).copy()
    proxy = pd.to_numeric(result["roro_proxy_value"], errors="coerce")
    result["roro_proxy_mean_20"] = proxy.rolling(20, min_periods=20).mean()
    result["roro_proxy_mean_60"] = proxy.rolling(60, min_periods=60).mean()
    result["roro_proxy_delta_20"] = proxy - proxy.shift(20)
    result["roro_proxy_delta_60"] = proxy - proxy.shift(60)
    result["roro_proxy_std_60"] = proxy.rolling(60, min_periods=60).std(ddof=0)
    return result


def fit_transform_roro_context_features(
    frame: pd.DataFrame,
    *,
    feature_order: list[str],
    lower_q: float,
    upper_q: float,
    epsilon: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Impute, clip, and z-score RORO context features with train-only stats."""

    context = frame.copy()
    train_mask = context["split"].astype(str).eq("train")
    if not train_mask.any():
        raise ValueError("Cannot fit RORO context scaler because split == train is empty.")

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
            raise ValueError(f"Cannot fit all-missing RORO feature: {feature}")
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
    """Group RORO context features for reporting."""

    groups: dict[str, list[str]] = {
        "roro_proxy": [],
        "risk_off_components": [],
    }
    for feature in feature_order:
        if feature in RISK_OFF_COMPONENTS:
            groups["risk_off_components"].append(feature)
        else:
            groups["roro_proxy"].append(feature)
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
    pca_metadata: dict[str, Any],
    source_manifest: dict[str, Any],
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
        warnings.append("normalized RORO feature matrix contains non-finite values")

    return {
        "status": "ok",
        "stage": "4-N13-3",
        "context_source": "public_data_roro_proxy",
        "terminology": {
            "roro_proxy": (
                "KC Fed-inspired public-data risk-on/risk-off proxy; "
                "not a full KC Fed replication"
            ),
            "risk_off_direction": "higher roro_proxy means stronger risk-off pressure",
            "btc_direction": "not hard-coded; learned by context-FiLM",
        },
        "method": {
            "raw_component_direction": "components are aligned so positive means risk-off",
            "proxy_construction": "train-only z-score plus SVD/PCA first component",
            "sign_policy": "flip first component sign to correlate positively with risk-off anchors",
            "no_label_leakage": "PCA/scaler uses macro components and train split dates only, not BTC labels",
        },
        "source": {
            "btc_source": str(btc_source),
            "macro_sources": source_manifest,
        },
        "alignment": {
            "asof_lag_days": int(asof_lag_days),
            "policy": "use latest public macro source date <= BTC image end date minus lag",
            "same_day_policy": "disabled by default through asof_lag_days=1",
            "max_age_days": safe_float(pd.to_numeric(context_table["roro_age_days"], errors="coerce").max()),
            "mean_age_days": safe_float(pd.to_numeric(context_table["roro_age_days"], errors="coerce").mean()),
        },
        "pca": pca_metadata,
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
        "stage": "4-N13-3",
        "context_source": "public_data_roro_proxy",
        "context_dim": audit["context_dim"],
        "feature_order": scaler["feature_order"],
        "normalized_feature_columns": scaler["normalized_feature_columns"],
        "feature_group_counts": audit["feature_group_counts"],
        "split_counts": audit["split_counts"],
        "fit_on": "train",
        "normalization": scaler["normalization"],
        "alignment": audit["alignment"],
        "terminology": audit["terminology"],
        "method": audit["method"],
        "pca": audit["pca"],
    }


def make_roro_context_output_name(
    *,
    output_prefix: str,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    feature_set_name: str,
) -> str:
    """Return filesystem-safe Stage 4 RORO context artifact name."""

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


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return correlation or None when one side is degenerate."""

    if left.size == 0 or right.size == 0:
        return None
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


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
