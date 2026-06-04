"""Audit the Stage 4 candidate BTC news source.

This is the first news-context step. It checks dataset coverage and leakage
alignment only; it does not build vectors, train, or evaluate a model.

Marker: news source audit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from stage2_btc.data import build_btc_samples, build_btc_splits, find_btc_ohlcv_source
from stage2_btc.data import load_btc_ohlcv
from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import get_stage4_model_config


DEFAULT_DATASET = "edaschau/bitcoin_news"
DEFAULT_SPLIT = "train"
DEFAULT_FILENAME = "BTC_match_title.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--dataset-filename",
        default=DEFAULT_FILENAME,
        help="File inside the Hugging Face dataset repo. BTC_match_title.csv supports headline-only context.",
    )
    parser.add_argument(
        "--news-csv",
        default=None,
        help="Optional local CSV override. Useful when Kaggle internet is disabled.",
    )
    parser.add_argument("--image-window", type=int, default=None)
    parser.add_argument("--return-horizon", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--output-prefix",
        default="stage4_news",
        help="Prefix for reports/tables output files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the source audit and write table/json reports."""

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

    news = load_news_frame(
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        dataset_filename=str(args.dataset_filename),
        news_csv=Path(args.news_csv) if args.news_csv else None,
        max_rows=args.max_rows,
    )
    news = standardize_news_frame(news)
    audit, daily_coverage, source_distribution, duplicate_audit, sample_coverage = (
        build_news_audit(
            news=news,
            samples_by_split=splits,
            dataset_name=str(args.dataset_name),
            split=str(args.split),
            image_window=image_window,
            return_horizon=return_horizon,
            btc_source=btc_source,
            max_rows=args.max_rows,
        )
    )

    prefix = str(args.output_prefix)
    tables_root = paths.tables_root
    tables_root.mkdir(parents=True, exist_ok=True)
    audit_path = tables_root / f"{prefix}_source_audit.json"
    daily_path = tables_root / f"{prefix}_daily_coverage.csv"
    source_path = tables_root / f"{prefix}_source_distribution.csv"
    duplicate_path = tables_root / f"{prefix}_duplicate_audit.csv"
    sample_path = tables_root / f"{prefix}_sample_coverage_by_split.csv"

    audit_path.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
    daily_coverage.to_csv(daily_path, index=False)
    source_distribution.to_csv(source_path, index=False)
    duplicate_audit.to_csv(duplicate_path, index=False)
    sample_coverage.to_csv(sample_path, index=False)

    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_name": args.dataset_name,
                "split": args.split,
                "rows": audit["dataset"]["num_rows"],
                "date_min": audit["dataset"]["date_min"],
                "date_max": audit["dataset"]["date_max"],
                "sample_date_min": audit["stage4_samples"]["date_min"],
                "sample_date_max": audit["stage4_samples"]["date_max"],
                "sample_coverage": audit["sample_news_coverage"],
                "written": {
                    "source_audit": str(audit_path),
                    "daily_coverage": str(daily_path),
                    "source_distribution": str(source_path),
                    "duplicate_audit": str(duplicate_path),
                    "sample_coverage_by_split": str(sample_path),
                },
            },
            indent=2,
        )
    )


def load_news_frame(
    dataset_name: str,
    split: str,
    dataset_filename: str,
    news_csv: Path | None,
    max_rows: int | None,
) -> pd.DataFrame:
    """Load the candidate news CSV into a pandas frame."""

    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("--max-rows must be positive when supplied.")

    if news_csv is not None:
        source_path = news_csv
    else:
        source_path = Path(
            hf_hub_download(
                repo_id=dataset_name,
                filename=dataset_filename,
                repo_type="dataset",
            )
        )

    if not source_path.exists():
        raise FileNotFoundError(f"News CSV not found: {source_path}")

    read_kwargs: dict[str, Any] = {"low_memory": False}
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows
    try:
        frame = pd.read_csv(source_path, **read_kwargs)
        loader = "pandas_c_engine"
    except pd.errors.ParserError:
        # Some auxiliary files in the public dataset contain malformed quotes.
        # For source audit purposes, skipping malformed rows is safer than
        # failing before coverage can be assessed.
        frame = pd.read_csv(
            source_path,
            engine="python",
            on_bad_lines="skip",
            nrows=max_rows,
        )
        loader = "pandas_python_engine_on_bad_lines_skip"
    frame.attrs["news_source_path"] = str(source_path)
    frame.attrs["news_loader"] = loader
    frame.attrs["dataset_split_argument"] = split
    return frame


def standardize_news_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize news columns needed for audit."""

    if raw.empty:
        raise ValueError("News dataset is empty.")

    frame = raw.copy()
    if "date_time" in frame.columns:
        parsed_time = pd.to_datetime(frame["date_time"], errors="coerce", utc=True).dt.tz_convert(None)
    elif "time_unix" in frame.columns:
        parsed_time = pd.to_datetime(
            frame["time_unix"],
            errors="coerce",
            unit="s",
            utc=True,
        ).dt.tz_convert(None)
    else:
        raise KeyError("News dataset must include either date_time or time_unix.")

    if parsed_time.isna().all():
        raise ValueError("Could not parse any news timestamps.")

    frame["news_time"] = parsed_time
    frame["news_date"] = parsed_time.dt.normalize()
    frame["title_text"] = _column_as_string(frame, "title")
    frame["article_text_string"] = _column_as_string(frame, "article_text")
    frame["url_text"] = _column_as_string(frame, "url")
    frame["source_text"] = _column_as_string(frame, "source")
    frame["normalized_title"] = frame["title_text"].map(normalize_text_for_duplicate_check)
    frame["article_length_chars"] = frame["article_text_string"].str.len()
    frame["title_length_chars"] = frame["title_text"].str.len()
    frame["has_title"] = frame["title_text"].str.strip().ne("")
    frame["has_article_text"] = frame["article_text_string"].str.strip().ne("")
    return frame.dropna(subset=["news_date"]).reset_index(drop=True)


def build_news_audit(
    *,
    news: pd.DataFrame,
    samples_by_split: dict[str, pd.DataFrame],
    dataset_name: str,
    split: str,
    image_window: int,
    return_horizon: int,
    btc_source: str | Path,
    max_rows: int | None,
) -> tuple[pd.DataFrame | dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create source-level and sample-alignment audit outputs."""

    all_samples = pd.concat(
        [frame.assign(split=name) for name, frame in samples_by_split.items()],
        ignore_index=True,
    )
    sample_dates = pd.to_datetime(all_samples["Date"]).dt.normalize()
    sample_min = sample_dates.min()
    sample_max = sample_dates.max()
    news_dates = pd.to_datetime(news["news_date"]).dt.normalize()

    full_news_range = pd.date_range(news_dates.min(), news_dates.max(), freq="D")
    news_daily = (
        news.groupby("news_date", dropna=False)
        .agg(
            news_count=("news_date", "size"),
            title_count=("has_title", "sum"),
            article_count=("has_article_text", "sum"),
            unique_sources=("source_text", lambda values: int(pd.Series(values).nunique(dropna=True))),
        )
        .reset_index()
        .rename(columns={"news_date": "date"})
    )
    daily_coverage = pd.DataFrame({"date": full_news_range})
    daily_coverage = daily_coverage.merge(news_daily, on="date", how="left")
    for column in ("news_count", "title_count", "article_count", "unique_sources"):
        daily_coverage[column] = daily_coverage[column].fillna(0).astype(int)
    daily_coverage["in_stage4_sample_range"] = daily_coverage["date"].between(sample_min, sample_max)

    source_distribution = (
        news["source_text"]
        .replace("", pd.NA)
        .value_counts(dropna=False)
        .rename_axis("source")
        .reset_index(name="row_count")
    )
    source_distribution["source"] = source_distribution["source"].astype("string").fillna("<missing>")

    duplicate_audit = build_duplicate_audit(news)
    sample_coverage = build_sample_coverage(samples_by_split, daily_coverage)

    stage4_range_daily = daily_coverage.loc[daily_coverage["in_stage4_sample_range"]].copy()
    sample_coverage_rows = sample_coverage.loc[sample_coverage["split"] == "all"]

    audit: dict[str, Any] = {
        "dataset": {
            "dataset_name": dataset_name,
            "split": split,
            "source_file": news.attrs.get("news_source_path"),
            "loader": news.attrs.get("news_loader"),
            "max_rows": max_rows,
            "num_rows": int(len(news)),
            "columns": [str(column) for column in news.columns],
            "date_min": _date_or_none(news_dates.min()),
            "date_max": _date_or_none(news_dates.max()),
            "has_2018_coverage": bool(news_dates.min() <= pd.Timestamp("2018-01-01")),
            "overlaps_stage4_sample_range": bool(news_dates.min() <= sample_max and news_dates.max() >= sample_min),
        },
        "stage4_samples": {
            "btc_source": str(btc_source),
            "image_window": int(image_window),
            "return_horizon": int(return_horizon),
            "num_samples": int(len(all_samples)),
            "date_min": _date_or_none(sample_min),
            "date_max": _date_or_none(sample_max),
            "split_counts": {
                split_name: int(len(frame))
                for split_name, frame in samples_by_split.items()
            },
        },
        "source_content": {
            "title_coverage_rate": _safe_float(news["has_title"].mean()),
            "article_text_coverage_rate": _safe_float(news["has_article_text"].mean()),
            "title_length_mean": _safe_float(news.loc[news["has_title"], "title_length_chars"].mean()),
            "article_length_mean": _safe_float(
                news.loc[news["has_article_text"], "article_length_chars"].mean()
            ),
            "num_sources": int(news["source_text"].replace("", pd.NA).nunique(dropna=True)),
            "top_sources": source_distribution.head(20).to_dict(orient="records"),
        },
        "daily_coverage": {
            "num_days_inside_news_range": int(len(daily_coverage)),
            "missing_days_inside_news_range": int((daily_coverage["news_count"] == 0).sum()),
            "stage4_range_days": int(len(stage4_range_daily)),
            "stage4_range_missing_days": int((stage4_range_daily["news_count"] == 0).sum()),
            "stage4_range_mean_news_per_day": _safe_float(stage4_range_daily["news_count"].mean()),
            "stage4_range_median_news_per_day": _safe_float(stage4_range_daily["news_count"].median()),
        },
        "sample_news_coverage": sample_coverage_rows.iloc[0].to_dict()
        if not sample_coverage_rows.empty
        else {},
        "leakage_policy": {
            "first_policy": "strict_t_minus_1",
            "allowed_news_for_image_end_t": "calendar date <= t - 1",
            "same_day_news": "excluded until BTC close cutoff and news timestamp cutoff are defended",
        },
    }
    return audit, daily_coverage, source_distribution, duplicate_audit, sample_coverage


def build_duplicate_audit(news: pd.DataFrame) -> pd.DataFrame:
    """Build URL/title duplicate summary."""

    total = len(news)
    url_nonempty = news["url_text"].str.strip().ne("")
    title_nonempty = news["normalized_title"].str.strip().ne("")
    rows = [
        duplicate_row("url", news.loc[url_nonempty, "url_text"], total),
        duplicate_row("normalized_title", news.loc[title_nonempty, "normalized_title"], total),
    ]
    return pd.DataFrame(rows)


def duplicate_row(name: str, values: pd.Series, total_rows: int) -> dict[str, Any]:
    """Summarize duplicate rate for one identifier column."""

    nonempty = int(len(values))
    unique = int(values.nunique(dropna=True))
    duplicate_rows = int(nonempty - unique)
    return {
        "field": name,
        "total_rows": int(total_rows),
        "nonempty_rows": nonempty,
        "unique_values": unique,
        "duplicate_rows": duplicate_rows,
        "duplicate_rate_total": duplicate_rows / total_rows if total_rows else None,
        "duplicate_rate_nonempty": duplicate_rows / nonempty if nonempty else None,
    }


def build_sample_coverage(
    samples_by_split: dict[str, pd.DataFrame],
    daily_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate strict t-1 and trailing 7-day news coverage by split."""

    counts = daily_coverage.set_index("date")["news_count"].sort_index()
    rows: list[dict[str, Any]] = []
    for split_name, frame in [
        *samples_by_split.items(),
        ("all", pd.concat(samples_by_split.values(), ignore_index=True)),
    ]:
        dates = pd.to_datetime(frame["Date"]).dt.normalize()
        prev_dates = dates - pd.Timedelta(days=1)
        prev_counts = prev_dates.map(counts).fillna(0).astype(int)
        trailing_7_counts = [
            int(counts.loc[(counts.index >= date - pd.Timedelta(days=7)) & (counts.index <= date - pd.Timedelta(days=1))].sum())
            for date in dates
        ]
        trailing_7 = pd.Series(trailing_7_counts, index=frame.index)
        rows.append(
            {
                "split": split_name,
                "num_samples": int(len(frame)),
                "sample_date_min": _date_or_none(dates.min()),
                "sample_date_max": _date_or_none(dates.max()),
                "strict_t_minus_1_missing_samples": int((prev_counts == 0).sum()),
                "strict_t_minus_1_coverage_rate": _safe_float((prev_counts > 0).mean()),
                "strict_t_minus_1_mean_news_count": _safe_float(prev_counts.mean()),
                "strict_t_minus_1_median_news_count": _safe_float(prev_counts.median()),
                "trailing_7d_missing_samples": int((trailing_7 == 0).sum()),
                "trailing_7d_coverage_rate": _safe_float((trailing_7 > 0).mean()),
                "trailing_7d_mean_news_count": _safe_float(trailing_7.mean()),
                "trailing_7d_median_news_count": _safe_float(trailing_7.median()),
            }
        )
    return pd.DataFrame(rows)


def _column_as_string(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a string series even when the column is absent."""

    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype="string")
    return frame[column].fillna("").astype("string")


def normalize_text_for_duplicate_check(value: Any) -> str:
    """Normalize text for duplicate-title checks."""

    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


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
