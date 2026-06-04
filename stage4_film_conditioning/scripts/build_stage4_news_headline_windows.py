"""Build Stage 4 headline-only news window aggregation tables.

This is the 4-N3 news-context step. It does not vectorize text and does not
train a model. It creates leakage-safe headline-window documents that 4-N4 can
use for train-only TF-IDF/SVD.

Marker: headline-only daily aggregation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from audit_stage4_news_alignment import NEWS_WINDOWS, SPLIT_ORDER
from audit_stage4_news_source import DEFAULT_DATASET, DEFAULT_FILENAME, DEFAULT_SPLIT
from audit_stage4_news_source import load_news_frame, standardize_news_frame
from stage2_btc.data import build_btc_samples, build_btc_splits, find_btc_ohlcv_source
from stage2_btc.data import load_btc_ohlcv
from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import get_stage4_model_config


SEPARATOR = " [SEP] "


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
    parser.add_argument("--window-days", type=int, nargs="+", default=list(NEWS_WINDOWS))
    parser.add_argument(
        "--max-headlines-per-window",
        type=int,
        default=0,
        help="Optional cap for newest headlines per sample window. 0 keeps all headlines.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=240,
        help="Maximum characters for report example previews.",
    )
    parser.add_argument(
        "--output-prefix",
        default="stage4_news_headline_windows",
        help="Prefix for reports/tables and outputs/stage4/news files.",
    )
    return parser.parse_args()


def main() -> None:
    """Build headline-window tables."""

    args = parse_args()
    config = load_config(args.config)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    stage4_model = get_stage4_model_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
    return_horizon = int(args.return_horizon or stage4_model["primary_return_horizon"])
    window_days = tuple(sorted({int(day) for day in args.window_days}))
    if not window_days:
        raise ValueError("--window-days must contain at least one positive integer.")
    if any(day <= 0 for day in window_days):
        raise ValueError("--window-days must contain only positive integers.")

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
    deduped_news, dedupe_audit = dedupe_headline_news(news)

    daily_full, daily_report = build_daily_headline_tables(deduped_news)
    sample_windows = build_sample_window_table(
        samples_by_split=splits,
        news=deduped_news,
        window_days=window_days,
        max_headlines_per_window=int(args.max_headlines_per_window),
    )
    summary = build_window_summary(sample_windows, window_days)
    examples = build_window_examples(sample_windows, window_days, preview_chars=int(args.preview_chars))

    output_name = f"{args.output_prefix}_i{image_window}_r{return_horizon}"
    news_output_root = paths.output_root / "news" / output_name
    news_output_root.mkdir(parents=True, exist_ok=True)
    tables_root = paths.tables_root
    tables_root.mkdir(parents=True, exist_ok=True)

    daily_full_path = write_large_table(daily_full, news_output_root / "daily_headlines")
    sample_windows_path = write_large_table(sample_windows, news_output_root / "sample_headline_windows")

    manifest = build_manifest(
        args=args,
        config=config,
        btc_source=btc_source,
        news=news,
        deduped_news=deduped_news,
        dedupe_audit=dedupe_audit,
        daily_full_path=daily_full_path,
        sample_windows_path=sample_windows_path,
        image_window=image_window,
        return_horizon=return_horizon,
        window_days=window_days,
        sample_windows=sample_windows,
    )

    manifest_path = tables_root / f"{args.output_prefix}_manifest.json"
    summary_path = tables_root / f"{args.output_prefix}_summary.csv"
    examples_path = tables_root / f"{args.output_prefix}_examples.csv"
    daily_report_path = tables_root / f"{args.output_prefix}_daily_coverage.csv"
    dedupe_path = tables_root / f"{args.output_prefix}_dedupe_audit.csv"

    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    summary.to_csv(summary_path, index=False)
    examples.to_csv(examples_path, index=False)
    daily_report.to_csv(daily_report_path, index=False)
    pd.DataFrame([dedupe_audit]).to_csv(dedupe_path, index=False)

    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_name": args.dataset_name,
                "dataset_filename": args.dataset_filename,
                "image_window": image_window,
                "return_horizon": return_horizon,
                "window_days": list(window_days),
                "raw_news_rows": int(len(news)),
                "deduped_news_rows": int(len(deduped_news)),
                "sample_rows": int(len(sample_windows)),
                "summary": summary.to_dict(orient="records"),
                "written": {
                    "full_daily_headlines": str(daily_full_path),
                    "full_sample_headline_windows": str(sample_windows_path),
                    "manifest": str(manifest_path),
                    "summary": str(summary_path),
                    "examples": str(examples_path),
                    "daily_coverage": str(daily_report_path),
                    "dedupe_audit": str(dedupe_path),
                },
            },
            indent=2,
        )
    )


def dedupe_headline_news(news: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove empty headlines and exact URL/title duplicates."""

    frame = news.copy()
    frame["headline_text"] = frame["title_text"].map(clean_headline_text)
    frame = frame.loc[frame["headline_text"].str.strip().ne("")].copy()
    frame = frame.sort_values(["news_time", "url_text", "normalized_title"]).reset_index(drop=True)

    rows_after_empty_title_filter = len(frame)
    url_nonempty = frame["url_text"].str.strip().ne("")
    duplicate_url = url_nonempty & frame["url_text"].duplicated(keep="first")
    frame = frame.loc[~duplicate_url].copy().reset_index(drop=True)

    title_nonempty = frame["normalized_title"].str.strip().ne("")
    duplicate_title = title_nonempty & frame["normalized_title"].duplicated(keep="first")
    frame = frame.loc[~duplicate_title].copy().reset_index(drop=True)

    frame["news_date"] = pd.to_datetime(frame["news_date"]).dt.normalize()
    frame["news_time"] = pd.to_datetime(frame["news_time"])
    audit = {
        "raw_rows": int(len(news)),
        "rows_after_empty_title_filter": int(rows_after_empty_title_filter),
        "dropped_empty_title_rows": int(len(news) - rows_after_empty_title_filter),
        "dropped_duplicate_url_rows": int(duplicate_url.sum()),
        "dropped_duplicate_normalized_title_rows": int(duplicate_title.sum()),
        "deduped_rows": int(len(frame)),
        "dedupe_policy": "drop empty titles, then duplicate non-empty URL, then duplicate normalized title",
    }
    return frame, audit


def build_daily_headline_tables(news: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build daily full headline text and compact daily coverage report."""

    rows: list[dict[str, Any]] = []
    for date, frame in news.groupby("news_date", sort=True):
        ordered = frame.sort_values(["news_time", "source_text", "headline_text"])
        headline_text = join_headlines(ordered["headline_text"])
        sources = sorted(set(str(value) for value in ordered["source_text"] if str(value).strip()))
        rows.append(
            {
                "date": _date_or_none(date),
                "news_count": int(len(ordered)),
                "unique_source_count": int(len(sources)),
                "sources": " | ".join(sources),
                "headline_text": headline_text,
                "headline_text_chars": int(len(headline_text)),
                "headline_text_hash": stable_hash(headline_text),
            }
        )
    daily_full = pd.DataFrame(rows)
    daily_report = daily_full.drop(columns=["headline_text"]).copy()
    return daily_full, daily_report


def build_sample_window_table(
    *,
    samples_by_split: dict[str, pd.DataFrame],
    news: pd.DataFrame,
    window_days: tuple[int, ...],
    max_headlines_per_window: int,
) -> pd.DataFrame:
    """Build 7/20/60-day headline documents for each BTC sample."""

    all_samples = pd.concat(
        [frame.assign(split=name) for name, frame in samples_by_split.items()],
        ignore_index=True,
    )
    all_samples["_split_order"] = all_samples["split"].map(SPLIT_ORDER).fillna(99).astype(int)
    all_samples = all_samples.sort_values(["_split_order", "Date", "sample_index"]).drop(
        columns=["_split_order"]
    )

    news = news.sort_values(["news_date", "news_time", "source_text", "headline_text"]).reset_index(drop=True)
    news_dates = pd.to_datetime(news["news_date"]).dt.normalize()

    rows: list[dict[str, Any]] = []
    for _, sample in all_samples.iterrows():
        image_end = pd.Timestamp(sample["Date"]).normalize()
        strict_end = image_end - pd.Timedelta(days=1)
        row: dict[str, Any] = {
            "split": str(sample["split"]),
            "sample_index": int(sample["sample_index"]),
            "image_end_date": _date_or_none(image_end),
            "label_end_date": _date_or_none(sample.get("label_end_date")),
            "strict_policy": "calendar_date_lte_t_minus_1",
            "same_day_excluded": True,
        }
        for window in window_days:
            start = image_end - pd.Timedelta(days=window)
            mask = (news_dates >= start) & (news_dates <= strict_end)
            window_frame = news.loc[mask]
            if max_headlines_per_window > 0 and len(window_frame) > max_headlines_per_window:
                window_frame = window_frame.tail(max_headlines_per_window)
            headline_text = join_headlines(window_frame["headline_text"])
            sources = sorted(
                set(str(value) for value in window_frame["source_text"] if str(value).strip())
            )
            prefix = f"{window}d"
            row[f"headline_start_date_{prefix}"] = _date_or_none(start)
            row[f"headline_end_date_{prefix}"] = _date_or_none(strict_end)
            row[f"news_count_{prefix}"] = int(len(window_frame))
            row[f"unique_source_count_{prefix}"] = int(len(sources))
            row[f"headline_text_{prefix}"] = headline_text
            row[f"headline_text_chars_{prefix}"] = int(len(headline_text))
            row[f"headline_text_hash_{prefix}"] = stable_hash(headline_text)
        rows.append(row)
    return pd.DataFrame(rows)


def build_window_summary(sample_windows: pd.DataFrame, window_days: tuple[int, ...]) -> pd.DataFrame:
    """Summarize sample-window headline counts by split."""

    frames = [
        *sample_windows.groupby("split", sort=False),
        ("all", sample_windows),
    ]
    rows: list[dict[str, Any]] = []
    for split_name, frame in frames:
        row: dict[str, Any] = {
            "split": split_name,
            "num_samples": int(len(frame)),
            "sample_date_min": frame["image_end_date"].min(),
            "sample_date_max": frame["image_end_date"].max(),
        }
        for window in window_days:
            prefix = f"{window}d"
            count = frame[f"news_count_{prefix}"]
            chars = frame[f"headline_text_chars_{prefix}"]
            sources = frame[f"unique_source_count_{prefix}"]
            row[f"coverage_rate_{prefix}"] = _safe_float((count > 0).mean())
            row[f"missing_samples_{prefix}"] = int((count == 0).sum())
            row[f"mean_news_count_{prefix}"] = _safe_float(count.mean())
            row[f"median_news_count_{prefix}"] = _safe_float(count.median())
            row[f"mean_unique_sources_{prefix}"] = _safe_float(sources.mean())
            row[f"median_unique_sources_{prefix}"] = _safe_float(sources.median())
            row[f"mean_headline_chars_{prefix}"] = _safe_float(chars.mean())
            row[f"median_headline_chars_{prefix}"] = _safe_float(chars.median())
        rows.append(row)
    return pd.DataFrame(rows)


def build_window_examples(
    sample_windows: pd.DataFrame,
    window_days: tuple[int, ...],
    preview_chars: int,
) -> pd.DataFrame:
    """Build compact examples with short headline previews."""

    selected: list[pd.DataFrame] = []
    for split_name, frame in sample_windows.groupby("split", sort=False):
        sorted_frame = frame.sort_values(["image_end_date", "sample_index"])
        selected.append(sorted_frame.head(2))
        selected.append(sorted_frame.tail(2))
    examples = pd.concat(selected, ignore_index=True)
    examples = examples.drop_duplicates(["split", "sample_index"]).sort_values(
        ["split", "image_end_date", "sample_index"]
    )
    keep_columns = ["split", "sample_index", "image_end_date", "label_end_date"]
    result = examples[keep_columns].copy()
    for window in window_days:
        prefix = f"{window}d"
        result[f"news_count_{prefix}"] = examples[f"news_count_{prefix}"].astype(int)
        result[f"unique_source_count_{prefix}"] = examples[f"unique_source_count_{prefix}"].astype(int)
        result[f"headline_preview_{prefix}"] = examples[f"headline_text_{prefix}"].map(
            lambda value: truncate_text(value, preview_chars)
        )
        result[f"headline_text_hash_{prefix}"] = examples[f"headline_text_hash_{prefix}"]
    return result


def build_manifest(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    btc_source: str | Path,
    news: pd.DataFrame,
    deduped_news: pd.DataFrame,
    dedupe_audit: dict[str, Any],
    daily_full_path: Path,
    sample_windows_path: Path,
    image_window: int,
    return_horizon: int,
    window_days: tuple[int, ...],
    sample_windows: pd.DataFrame,
) -> dict[str, Any]:
    """Build run manifest for headline-window aggregation."""

    split_config = config.get("split", {})
    return {
        "status": "ok",
        "stage": "4-N3",
        "dataset": {
            "dataset_name": str(args.dataset_name),
            "dataset_filename": str(args.dataset_filename),
            "split": str(args.split),
            "source_file": news.attrs.get("news_source_path"),
            "raw_rows": int(len(news)),
            "deduped_rows": int(len(deduped_news)),
            "date_min": _date_or_none(pd.to_datetime(deduped_news["news_date"]).min()),
            "date_max": _date_or_none(pd.to_datetime(deduped_news["news_date"]).max()),
        },
        "stage4_samples": {
            "btc_source": str(btc_source),
            "image_window": int(image_window),
            "return_horizon": int(return_horizon),
            "split_method": split_config.get("split_method"),
            "num_samples": int(len(sample_windows)),
            "sample_date_min": sample_windows["image_end_date"].min(),
            "sample_date_max": sample_windows["image_end_date"].max(),
        },
        "alignment_policy": {
            "name": "strict_t_minus_1_calendar_date",
            "window_days": list(window_days),
            "window_rule": "for window W, use news dates from t-W through t-1 inclusive",
            "same_day_news": "excluded",
        },
        "dedupe_audit": dedupe_audit,
        "headline_cap": {
            "max_headlines_per_window": int(args.max_headlines_per_window),
            "policy": "0 means no cap; otherwise keep newest headlines within each window",
        },
        "outputs": {
            "daily_headlines": str(daily_full_path),
            "sample_headline_windows": str(sample_windows_path),
        },
    }


def write_large_table(frame: pd.DataFrame, output_base: Path) -> Path:
    """Write a large table as parquet when available, otherwise gzip CSV."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = output_base.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet_path, index=False, compression="zstd")
        return parquet_path
    except Exception:
        gzip_path = output_base.with_suffix(".csv.gz")
        frame.to_csv(gzip_path, index=False, compression="gzip")
        return gzip_path


def clean_headline_text(value: Any) -> str:
    """Normalize headline whitespace for aggregation."""

    text = str(value or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def join_headlines(values: pd.Series) -> str:
    """Join headline strings with a stable separator."""

    headlines = [clean_headline_text(value) for value in values if clean_headline_text(value)]
    return SEPARATOR.join(headlines)


def stable_hash(text: str) -> str:
    """Return a stable short hash for long headline text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def truncate_text(value: Any, limit: int) -> str:
    """Return a compact preview for reports."""

    text = clean_headline_text(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


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
