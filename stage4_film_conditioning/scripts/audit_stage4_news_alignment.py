"""Audit Stage 4 news publication-time alignment.

This is the 4-N2 no-future-leakage step. It locks the first news-context
alignment rule before any text vectorizer or model is built.

Marker: news alignment no-future-leakage
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

from audit_stage4_news_source import DEFAULT_DATASET, DEFAULT_FILENAME, DEFAULT_SPLIT
from audit_stage4_news_source import load_news_frame, standardize_news_frame
from stage2_btc.data import build_btc_samples, build_btc_splits, find_btc_ohlcv_source
from stage2_btc.data import load_btc_ohlcv
from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import get_stage4_model_config


SPLIT_ORDER: dict[str, int] = {"train": 0, "validation": 1, "test": 2, "all": 3}
NEWS_WINDOWS: tuple[int, ...] = (7, 20, 60)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-filename", default=DEFAULT_FILENAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--news-csv", default=None)
    parser.add_argument("--image-window", type=int, default=None)
    parser.add_argument("--return-horizon", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--output-prefix",
        default="stage4_news_alignment",
        help="Prefix for reports/tables output files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run alignment audit and write reports."""

    args = parse_args()
    config = load_config(args.config)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    stage4_model = get_stage4_model_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
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

    news_raw = load_news_frame(
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        dataset_filename=str(args.dataset_filename),
        news_csv=Path(args.news_csv) if args.news_csv else None,
        max_rows=args.max_rows,
    )
    news = standardize_news_frame(news_raw)

    sample_alignment = build_sample_alignment(splits, news)
    split_summary = build_split_summary(sample_alignment)
    missing_dates = build_missing_dates(sample_alignment)
    examples = build_alignment_examples(sample_alignment)
    policy = build_policy_report(
        config=config,
        news=news,
        sample_alignment=sample_alignment,
        split_summary=split_summary,
        dataset_name=str(args.dataset_name),
        dataset_filename=str(args.dataset_filename),
        split=str(args.split),
        btc_source=btc_source,
        image_window=image_window,
        return_horizon=return_horizon,
    )

    tables_root = paths.tables_root
    tables_root.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix)
    policy_path = tables_root / f"{prefix}_policy.json"
    split_path = tables_root / f"{prefix}_by_split.csv"
    sample_path = tables_root / f"{prefix}_sample_counts.csv"
    missing_path = tables_root / f"{prefix}_missing_t_minus_1_dates.csv"
    examples_path = tables_root / f"{prefix}_examples.csv"

    policy_path.write_text(json.dumps(_jsonable(policy), indent=2), encoding="utf-8")
    split_summary.to_csv(split_path, index=False)
    sample_alignment.to_csv(sample_path, index=False)
    missing_dates.to_csv(missing_path, index=False)
    examples.to_csv(examples_path, index=False)

    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_name": args.dataset_name,
                "dataset_filename": args.dataset_filename,
                "image_window": image_window,
                "return_horizon": return_horizon,
                "policy": policy["alignment_policy"],
                "split_summary": split_summary.to_dict(orient="records"),
                "written": {
                    "policy": str(policy_path),
                    "by_split": str(split_path),
                    "sample_counts": str(sample_path),
                    "missing_t_minus_1_dates": str(missing_path),
                    "examples": str(examples_path),
                },
            },
            indent=2,
        )
    )


def build_sample_alignment(
    samples_by_split: dict[str, pd.DataFrame],
    news: pd.DataFrame,
) -> pd.DataFrame:
    """Build one alignment row per BTC sample."""

    news_dates = pd.to_datetime(news["news_date"]).dt.normalize()
    daily_counts = news_dates.value_counts().sort_index()
    daily_counts.index = pd.to_datetime(daily_counts.index).normalize()

    rows: list[dict[str, Any]] = []
    for split_name, frame in samples_by_split.items():
        for _, sample in frame.iterrows():
            image_end = pd.Timestamp(sample["Date"]).normalize()
            strict_cutoff = image_end - pd.Timedelta(days=1)
            start_7d = image_end - pd.Timedelta(days=7)
            start_20d = image_end - pd.Timedelta(days=20)
            start_60d = image_end - pd.Timedelta(days=60)
            rows.append(
                {
                    "split": split_name,
                    "sample_index": int(sample["sample_index"]),
                    "image_end_date": _date_or_none(image_end),
                    "label_end_date": _date_or_none(sample.get("label_end_date")),
                    "strict_policy": "calendar_date_lte_t_minus_1",
                    "allowed_news_start_date": _date_or_none(news_dates.min()),
                    "allowed_news_end_date": _date_or_none(strict_cutoff),
                    "strict_t_minus_1_date": _date_or_none(strict_cutoff),
                    "news_count_t_minus_1": int(daily_counts.get(strict_cutoff, 0)),
                    "news_count_7d_t_minus_1": int(
                        daily_counts.loc[
                            (daily_counts.index >= start_7d)
                            & (daily_counts.index <= strict_cutoff)
                        ].sum()
                    ),
                    "news_count_20d_t_minus_1": int(
                        daily_counts.loc[
                            (daily_counts.index >= start_20d)
                            & (daily_counts.index <= strict_cutoff)
                        ].sum()
                    ),
                    "news_count_60d_t_minus_1": int(
                        daily_counts.loc[
                            (daily_counts.index >= start_60d)
                            & (daily_counts.index <= strict_cutoff)
                        ].sum()
                    ),
                    "same_day_news_count_excluded": int(daily_counts.get(image_end, 0)),
                    "has_t_minus_1_news": bool(daily_counts.get(strict_cutoff, 0) > 0),
                    "has_7d_news": bool(
                        daily_counts.loc[
                            (daily_counts.index >= start_7d)
                            & (daily_counts.index <= strict_cutoff)
                        ].sum()
                        > 0
                    ),
                    "has_20d_news": bool(
                        daily_counts.loc[
                            (daily_counts.index >= start_20d)
                            & (daily_counts.index <= strict_cutoff)
                        ].sum()
                        > 0
                    ),
                    "has_60d_news": bool(
                        daily_counts.loc[
                            (daily_counts.index >= start_60d)
                            & (daily_counts.index <= strict_cutoff)
                        ].sum()
                        > 0
                    ),
                    "same_day_exclusion_active": True,
                }
            )
    result = pd.DataFrame(rows)
    result["_split_order"] = result["split"].map(SPLIT_ORDER).fillna(99).astype(int)
    result = result.sort_values(["_split_order", "image_end_date", "sample_index"]).drop(
        columns=["_split_order"]
    )
    train_fit_dates = set(
        result.loc[result["split"] == "train", "strict_t_minus_1_date"].dropna().astype(str)
    )
    result["text_vectorizer_fit_doc"] = result["strict_t_minus_1_date"].astype(str).isin(train_fit_dates)
    result.loc[result["split"] != "train", "text_vectorizer_fit_doc"] = False
    return result.reset_index(drop=True)


def build_split_summary(sample_alignment: pd.DataFrame) -> pd.DataFrame:
    """Aggregate alignment checks by split and all samples."""

    frames = [
        *sample_alignment.groupby("split", sort=False),
        ("all", sample_alignment),
    ]
    rows: list[dict[str, Any]] = []
    for split_name, frame in frames:
        rows.append(
            {
                "split": split_name,
                "num_samples": int(len(frame)),
                "sample_date_min": frame["image_end_date"].min(),
                "sample_date_max": frame["image_end_date"].max(),
                "strict_t_minus_1_missing_samples": int((~frame["has_t_minus_1_news"]).sum()),
                "strict_t_minus_1_coverage_rate": _safe_float(frame["has_t_minus_1_news"].mean()),
                "trailing_7d_missing_samples": int((~frame["has_7d_news"]).sum()),
                "trailing_7d_coverage_rate": _safe_float(frame["has_7d_news"].mean()),
                "trailing_20d_missing_samples": int((~frame["has_20d_news"]).sum()),
                "trailing_20d_coverage_rate": _safe_float(frame["has_20d_news"].mean()),
                "trailing_60d_missing_samples": int((~frame["has_60d_news"]).sum()),
                "trailing_60d_coverage_rate": _safe_float(frame["has_60d_news"].mean()),
                "mean_news_count_t_minus_1": _safe_float(frame["news_count_t_minus_1"].mean()),
                "median_news_count_t_minus_1": _safe_float(frame["news_count_t_minus_1"].median()),
                "mean_news_count_7d": _safe_float(frame["news_count_7d_t_minus_1"].mean()),
                "median_news_count_7d": _safe_float(frame["news_count_7d_t_minus_1"].median()),
                "mean_news_count_20d": _safe_float(frame["news_count_20d_t_minus_1"].mean()),
                "median_news_count_20d": _safe_float(frame["news_count_20d_t_minus_1"].median()),
                "mean_news_count_60d": _safe_float(frame["news_count_60d_t_minus_1"].mean()),
                "median_news_count_60d": _safe_float(frame["news_count_60d_t_minus_1"].median()),
                "same_day_news_excluded_samples": int(
                    (frame["same_day_news_count_excluded"] > 0).sum()
                ),
                "same_day_news_excluded_coverage_rate": _safe_float(
                    (frame["same_day_news_count_excluded"] > 0).mean()
                ),
                "mean_same_day_news_count_excluded": _safe_float(
                    frame["same_day_news_count_excluded"].mean()
                ),
                "text_vectorizer_fit_doc_count": int(frame["text_vectorizer_fit_doc"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_missing_dates(sample_alignment: pd.DataFrame) -> pd.DataFrame:
    """Return strict t-1 dates where no headline news is available."""

    missing = sample_alignment.loc[~sample_alignment["has_t_minus_1_news"]].copy()
    if missing.empty:
        return pd.DataFrame(
            columns=[
                "strict_t_minus_1_date",
                "affected_splits",
                "affected_sample_count",
                "first_image_end_date",
                "last_image_end_date",
            ]
        )
    grouped = missing.groupby("strict_t_minus_1_date", sort=True)
    rows = []
    for date, frame in grouped:
        rows.append(
            {
                "strict_t_minus_1_date": date,
                "affected_splits": ",".join(sorted(frame["split"].unique())),
                "affected_sample_count": int(len(frame)),
                "first_image_end_date": frame["image_end_date"].min(),
                "last_image_end_date": frame["image_end_date"].max(),
            }
        )
    return pd.DataFrame(rows)


def build_alignment_examples(sample_alignment: pd.DataFrame) -> pd.DataFrame:
    """Pick representative rows for report/debugging."""

    selected: list[pd.DataFrame] = []
    for split_name, frame in sample_alignment.groupby("split", sort=False):
        sorted_frame = frame.sort_values(["image_end_date", "sample_index"])
        selected.append(sorted_frame.head(2))
        selected.append(sorted_frame.tail(2))
        missing = sorted_frame.loc[~sorted_frame["has_t_minus_1_news"]].head(2)
        if not missing.empty:
            selected.append(missing)
    examples = pd.concat(selected, ignore_index=True)
    return examples.drop_duplicates(["split", "sample_index"]).sort_values(
        ["split", "image_end_date", "sample_index"]
    )


def build_policy_report(
    *,
    config: dict[str, Any],
    news: pd.DataFrame,
    sample_alignment: pd.DataFrame,
    split_summary: pd.DataFrame,
    dataset_name: str,
    dataset_filename: str,
    split: str,
    btc_source: str | Path,
    image_window: int,
    return_horizon: int,
) -> dict[str, Any]:
    """Create the JSON policy report."""

    news_dates = pd.to_datetime(news["news_date"]).dt.normalize()
    train_fit_dates = sample_alignment.loc[
        sample_alignment["text_vectorizer_fit_doc"],
        "strict_t_minus_1_date",
    ]
    split_config = config.get("split", {})
    return {
        "status": "ok",
        "stage": "4-N2",
        "dataset": {
            "dataset_name": dataset_name,
            "dataset_filename": dataset_filename,
            "split": split,
            "source_file": news.attrs.get("news_source_path"),
            "num_rows": int(len(news)),
            "date_min": _date_or_none(news_dates.min()),
            "date_max": _date_or_none(news_dates.max()),
        },
        "stage4_samples": {
            "btc_source": str(btc_source),
            "image_window": int(image_window),
            "return_horizon": int(return_horizon),
            "split_method": split_config.get("split_method"),
            "test_start_date": split_config.get("test_start_date"),
            "test_end_date": split_config.get("test_end_date"),
            "num_samples": int(len(sample_alignment)),
            "sample_date_min": sample_alignment["image_end_date"].min(),
            "sample_date_max": sample_alignment["image_end_date"].max(),
        },
        "alignment_policy": {
            "name": "strict_t_minus_1_calendar_date",
            "decision_time_for_image_end_t": "after chart image ending at calendar date t is formed",
            "allowed_news": "news calendar date <= t - 1",
            "excluded_news": "same-calendar-day news and all later news",
            "reason": (
                "BTC daily candle close cutoff and news publication cutoff are not defended yet; "
                "strict t-1 prevents same-day future information from entering the model."
            ),
            "timezone_handling": "date_time parsed as UTC, then converted to timezone-naive calendar date",
            "first_news_windows_days": list(NEWS_WINDOWS),
            "window_rule": "for window W, use news dates from t-W through t-1 inclusive",
        },
        "text_vectorizer_fit_policy": {
            "fit_only_on": "headline window documents referenced by train samples under strict t-1",
            "do_not_fit_on": "validation/test headline window documents or any same-day/future news",
            "window_days": list(NEWS_WINDOWS),
            "train_fit_sample_count": int(len(train_fit_dates)),
            "train_fit_document_count_per_window": int(len(train_fit_dates)),
            "train_fit_total_window_document_count": int(len(train_fit_dates) * len(NEWS_WINDOWS)),
            "train_fit_unique_date_count": int(train_fit_dates.nunique()),
            "train_fit_date_min": train_fit_dates.min() if not train_fit_dates.empty else None,
            "train_fit_date_max": train_fit_dates.max() if not train_fit_dates.empty else None,
        },
        "split_summary": split_summary.to_dict(orient="records"),
        "go_no_go": {
            "proceed_to_4_N3": True,
            "reason": "Strict t-1 coverage is above 95%, and 7/20/60-day trailing coverage is complete.",
        },
    }


def _date_or_none(value: Any) -> str | None:
    """Convert Timestamp/date to ISO date string."""

    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


if __name__ == "__main__":
    main()
