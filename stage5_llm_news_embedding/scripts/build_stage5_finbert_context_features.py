#!/usr/bin/env python3
"""Aggregate FinBERT headline sentiment into sample-level context features.

This is Stage 5 5-9C. It converts the headline-level FinBERT output from 5-9A
into strict t-1 7/20/60-day trailing-window features that can be passed to the
Stage 4 prebuilt-context FiLM runner.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_TABLES = ROOT / "reports" / "tables"
DATA_INVENTORY = ROOT / "data_inventory"
DEFAULT_RUN_ID = "finbert_prosusai_headline_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--finbert-items",
        default="",
        help="Path to the 5-9A headline-level FinBERT sentiment CSV.",
    )
    parser.add_argument(
        "--sample-table",
        default="",
        help=(
            "Optional Stage4/Stage5 sample table. If omitted, the script uses "
            "the Stage4 deduplicated headline-window parquet when available."
        ),
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[7, 20, 60])
    parser.add_argument("--high-confidence-threshold", type=float, default=0.80)
    parser.add_argument("--output-name", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--output-prefix",
        default="stage5_9c_finbert_sentiment_context",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_finbert_items(args: argparse.Namespace) -> Path:
    candidates: list[Path] = []
    if args.finbert_items:
        candidates.append(Path(args.finbert_items))

    for manifest_path in [
        DATA_INVENTORY / f"{args.run_id}_manifest.json",
        REPORT_TABLES / f"{args.run_id}_manifest.json",
    ]:
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            value = manifest.get("outputs", {}).get("finbert_sentiment_items")
            if value:
                candidates.append(Path(value))

    candidates.extend(
        [
            ROOT
            / "outputs/stage5/finbert_sentiment"
            / args.run_id
            / "stage5_finbert_sentiment_items.csv",
            ROOT.parent
            / "5_9a_results/outputs/stage5/finbert_sentiment"
            / args.run_id
            / "stage5_finbert_sentiment_items.csv",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find FinBERT sentiment item CSV. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def resolve_sample_table(user_path: str) -> Path:
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"Sample table missing: {path}")
        return path

    candidates = [
        ROOT.parent
        / "stage4_film_conditioning/outputs/stage4/news/"
        "stage4_news_headline_windows_i60_r20/sample_headline_windows.parquet",
        REPORT_TABLES / "stage5_news_alignment_sample_counts.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not find sample table for FinBERT aggregation.")


def read_sample_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: " + ", ".join(missing))


def safe_float(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def std_or_zero(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return safe_float(values.std(ddof=1))


def build_scaler(features: pd.DataFrame, windows: list[int]) -> dict[str, dict[str, float]]:
    train = features.loc[features["split"].eq("train")]
    if train.empty:
        raise ValueError("Cannot fit FinBERT context scaler: train split is empty.")

    scaler: dict[str, dict[str, float]] = {}
    for window in windows:
        col = f"finbert_sentiment_mean_{window}d"
        values = pd.to_numeric(train[col], errors="coerce").dropna()
        if values.empty:
            mean = 0.0
            std = 1.0
        else:
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
        scaler[col] = {"mean": mean, "std": std}
    return scaler


def aggregate_context(
    *,
    samples: pd.DataFrame,
    items: pd.DataFrame,
    windows: list[int],
    high_confidence_threshold: float,
) -> pd.DataFrame:
    samples = samples.copy().reset_index(drop=True)
    items = items.copy().reset_index(drop=True)

    require_columns(
        samples,
        ["split", "sample_index", "image_end_date", "label_end_date", "strict_policy"],
        "sample table",
    )
    require_columns(
        items,
        [
            "news_item_id",
            "news_date",
            "source_text",
            "finbert_positive_prob",
            "finbert_negative_prob",
            "finbert_neutral_prob",
            "finbert_label",
            "finbert_confidence",
            "finbert_sentiment_score",
        ],
        "FinBERT item CSV",
    )

    samples["image_end_date_ts"] = pd.to_datetime(samples["image_end_date"]).dt.normalize()
    items["news_date_ts"] = pd.to_datetime(items["news_date"]).dt.normalize()
    items["source_text"] = items["source_text"].fillna("").astype(str)

    order = np.argsort(items["news_date_ts"].to_numpy(dtype="datetime64[ns]"))
    sorted_dates = items["news_date_ts"].iloc[order].reset_index(drop=True).to_numpy(dtype="datetime64[ns]")
    sorted_items = items.iloc[order].reset_index(drop=True)

    metadata_cols = ["split", "sample_index", "image_end_date", "label_end_date", "strict_policy"]
    out = samples[metadata_cols].copy()

    if "same_day_excluded" in samples.columns:
        out["same_day_news_count_excluded"] = samples["same_day_excluded"]
    elif "same_day_news_count_excluded" in samples.columns:
        out["same_day_news_count_excluded"] = samples["same_day_news_count_excluded"]
    else:
        out["same_day_news_count_excluded"] = pd.NA

    for window in windows:
        rows: list[dict[str, Any]] = []
        for _, sample in samples.iterrows():
            image_date = pd.Timestamp(sample["image_end_date_ts"])
            start_date = image_date - pd.Timedelta(days=int(window))
            end_date = image_date - pd.Timedelta(days=1)

            left = int(np.searchsorted(sorted_dates, np.datetime64(start_date), side="left"))
            right = int(np.searchsorted(sorted_dates, np.datetime64(end_date), side="right"))
            frame = sorted_items.iloc[left:right]
            count = int(len(frame))

            row: dict[str, Any] = {
                f"finbert_window_start_date_{window}d": start_date.date().isoformat(),
                f"finbert_window_end_date_{window}d": end_date.date().isoformat(),
                f"finbert_news_count_{window}d": count,
                f"finbert_missing_{window}d": count == 0,
            }
            if count == 0:
                row.update(
                    {
                        f"finbert_unique_source_count_{window}d": 0,
                        f"finbert_positive_count_{window}d": 0,
                        f"finbert_negative_count_{window}d": 0,
                        f"finbert_neutral_count_{window}d": 0,
                        f"finbert_positive_ratio_{window}d": 0.0,
                        f"finbert_negative_ratio_{window}d": 0.0,
                        f"finbert_neutral_ratio_{window}d": 0.0,
                        f"finbert_confidence_mean_{window}d": 0.0,
                        f"finbert_confidence_max_{window}d": 0.0,
                        f"finbert_low_confidence_count_{window}d": 0,
                        f"finbert_high_confidence_positive_count_{window}d": 0,
                        f"finbert_high_confidence_negative_count_{window}d": 0,
                        f"finbert_sentiment_mean_{window}d": 0.0,
                        f"finbert_sentiment_sum_{window}d": 0.0,
                        f"finbert_sentiment_std_{window}d": 0.0,
                        f"finbert_sentiment_abs_mean_{window}d": 0.0,
                        f"finbert_confidence_weighted_sentiment_mean_{window}d": 0.0,
                        f"finbert_positive_prob_mean_{window}d": 0.0,
                        f"finbert_negative_prob_mean_{window}d": 0.0,
                        f"finbert_neutral_prob_mean_{window}d": 0.0,
                    }
                )
                rows.append(row)
                continue

            confidence = pd.to_numeric(frame["finbert_confidence"], errors="coerce").fillna(0.0)
            sentiment = pd.to_numeric(frame["finbert_sentiment_score"], errors="coerce").fillna(0.0)
            pos_prob = pd.to_numeric(frame["finbert_positive_prob"], errors="coerce").fillna(0.0)
            neg_prob = pd.to_numeric(frame["finbert_negative_prob"], errors="coerce").fillna(0.0)
            neu_prob = pd.to_numeric(frame["finbert_neutral_prob"], errors="coerce").fillna(0.0)
            labels = frame["finbert_label"].fillna("").astype(str)
            high_conf = confidence.ge(high_confidence_threshold)
            weighted_denominator = float(confidence.sum())
            if weighted_denominator > 0:
                weighted_mean = float((sentiment * confidence).sum() / weighted_denominator)
            else:
                weighted_mean = float(sentiment.mean())

            row.update(
                {
                    f"finbert_unique_source_count_{window}d": int(
                        len(set(value for value in frame["source_text"] if str(value).strip()))
                    ),
                    f"finbert_positive_count_{window}d": int(labels.eq("positive").sum()),
                    f"finbert_negative_count_{window}d": int(labels.eq("negative").sum()),
                    f"finbert_neutral_count_{window}d": int(labels.eq("neutral").sum()),
                    f"finbert_positive_ratio_{window}d": float(labels.eq("positive").mean()),
                    f"finbert_negative_ratio_{window}d": float(labels.eq("negative").mean()),
                    f"finbert_neutral_ratio_{window}d": float(labels.eq("neutral").mean()),
                    f"finbert_confidence_mean_{window}d": float(confidence.mean()),
                    f"finbert_confidence_max_{window}d": float(confidence.max()),
                    f"finbert_low_confidence_count_{window}d": int(confidence.lt(0.50).sum()),
                    f"finbert_high_confidence_positive_count_{window}d": int(
                        (high_conf & labels.eq("positive")).sum()
                    ),
                    f"finbert_high_confidence_negative_count_{window}d": int(
                        (high_conf & labels.eq("negative")).sum()
                    ),
                    f"finbert_sentiment_mean_{window}d": float(sentiment.mean()),
                    f"finbert_sentiment_sum_{window}d": float(sentiment.sum()),
                    f"finbert_sentiment_std_{window}d": std_or_zero(sentiment),
                    f"finbert_sentiment_abs_mean_{window}d": float(sentiment.abs().mean()),
                    f"finbert_confidence_weighted_sentiment_mean_{window}d": weighted_mean,
                    f"finbert_positive_prob_mean_{window}d": float(pos_prob.mean()),
                    f"finbert_negative_prob_mean_{window}d": float(neg_prob.mean()),
                    f"finbert_neutral_prob_mean_{window}d": float(neu_prob.mean()),
                }
            )
            rows.append(row)

        window_frame = pd.DataFrame(rows)
        out = pd.concat([out, window_frame], axis=1)

        expected_col = None
        for candidate in [f"news_count_{window}d", f"news_count_{window}d_t_minus_1"]:
            if candidate in samples.columns:
                expected_col = candidate
                break
        if expected_col is not None:
            out[f"finbert_count_matches_sample_table_{window}d"] = (
                out[f"finbert_news_count_{window}d"].astype(int) == samples[expected_col].astype(int)
            )

    scaler = build_scaler(out, windows)
    for window in windows:
        sentiment_col = f"finbert_sentiment_mean_{window}d"
        weighted_col = f"finbert_confidence_weighted_sentiment_mean_{window}d"
        scale = scaler[sentiment_col]
        z = (pd.to_numeric(out[sentiment_col], errors="coerce").fillna(0.0) - scale["mean"]) / scale["std"]
        out[f"finbert_sentiment_mean_z_{window}d"] = z
        out[f"finbert_news_fg_proxy_{window}d"] = 50.0 + 50.0 * np.tanh(z)

        # A weighted proxy is exported separately, but the train scaler is still
        # based on plain sentiment_mean so the transformation has one reference.
        weighted_z = (
            pd.to_numeric(out[weighted_col], errors="coerce").fillna(0.0) - scale["mean"]
        ) / scale["std"]
        out[f"finbert_weighted_sentiment_z_{window}d"] = weighted_z
        out[f"finbert_weighted_news_fg_proxy_{window}d"] = 50.0 + 50.0 * np.tanh(weighted_z)

    if 7 in windows and 20 in windows:
        out["finbert_sentiment_delta_20d_7d"] = (
            out["finbert_sentiment_mean_20d"] - out["finbert_sentiment_mean_7d"]
        )
        out["finbert_news_fg_proxy_delta_20d_7d"] = (
            out["finbert_news_fg_proxy_20d"] - out["finbert_news_fg_proxy_7d"]
        )
    if 20 in windows and 60 in windows:
        out["finbert_sentiment_delta_60d_20d"] = (
            out["finbert_sentiment_mean_60d"] - out["finbert_sentiment_mean_20d"]
        )
        out["finbert_news_fg_proxy_delta_60d_20d"] = (
            out["finbert_news_fg_proxy_60d"] - out["finbert_news_fg_proxy_20d"]
        )

    out.attrs["scaler"] = scaler
    return out


def build_split_summary(features: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test", "all"]:
        frame = features if split == "all" else features.loc[features["split"].eq(split)]
        if frame.empty:
            continue
        row: dict[str, Any] = {"split": split, "num_samples": int(len(frame))}
        for window in windows:
            row[f"missing_rate_{window}d"] = float(frame[f"finbert_missing_{window}d"].mean())
            row[f"mean_news_count_{window}d"] = float(frame[f"finbert_news_count_{window}d"].mean())
            row[f"mean_sentiment_{window}d"] = float(frame[f"finbert_sentiment_mean_{window}d"].mean())
            row[f"std_sentiment_{window}d"] = float(frame[f"finbert_sentiment_mean_{window}d"].std(ddof=1))
            row[f"mean_weighted_sentiment_{window}d"] = float(
                frame[f"finbert_confidence_weighted_sentiment_mean_{window}d"].mean()
            )
            row[f"mean_positive_ratio_{window}d"] = float(frame[f"finbert_positive_ratio_{window}d"].mean())
            row[f"mean_negative_ratio_{window}d"] = float(frame[f"finbert_negative_ratio_{window}d"].mean())
            row[f"mean_neutral_ratio_{window}d"] = float(frame[f"finbert_neutral_ratio_{window}d"].mean())
            row[f"mean_news_fg_proxy_{window}d"] = float(frame[f"finbert_news_fg_proxy_{window}d"].mean())
            match_col = f"finbert_count_matches_sample_table_{window}d"
            if match_col in frame.columns:
                row[f"count_match_rate_{window}d"] = float(frame[match_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric_cols = [
        col
        for col in features.columns
        if col.startswith("finbert_") and pd.api.types.is_numeric_dtype(features[col])
    ]
    for split in ["train", "validation", "test", "all"]:
        frame = features if split == "all" else features.loc[features["split"].eq(split)]
        if frame.empty:
            continue
        for col in numeric_cols:
            values = pd.to_numeric(frame[col], errors="coerce")
            rows.append(
                {
                    "split": split,
                    "feature": col,
                    "num_rows": int(len(values)),
                    "missing_rate": float(values.isna().mean()),
                    "mean": safe_float(values.mean()),
                    "std": safe_float(values.std(ddof=1)),
                    "min": safe_float(values.min()),
                    "max": safe_float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def build_report(
    *,
    manifest: dict[str, Any],
    split_summary: pd.DataFrame,
    features: pd.DataFrame,
    windows: list[int],
) -> str:
    all_row = split_summary.loc[split_summary["split"].eq("all")].iloc[0]
    train_row = split_summary.loc[split_summary["split"].eq("train")].iloc[0]
    test_row = split_summary.loc[split_summary["split"].eq("test")].iloc[0]

    coverage_lines = []
    sentiment_lines = []
    for window in windows:
        coverage_lines.append(
            f"- {window}d: missing `{all_row[f'missing_rate_{window}d']:.4f}`, "
            f"mean count `{all_row[f'mean_news_count_{window}d']:.2f}`, "
            f"count match `{all_row.get(f'count_match_rate_{window}d', float('nan')):.4f}`"
        )
        sentiment_lines.append(
            f"- {window}d: train sentiment `{train_row[f'mean_sentiment_{window}d']:.4f}`, "
            f"test sentiment `{test_row[f'mean_sentiment_{window}d']:.4f}`, "
            f"test news-FG proxy `{test_row[f'mean_news_fg_proxy_{window}d']:.2f}`"
        )

    feature_cols = [col for col in features.columns if col.startswith("finbert_")]
    return f"""# 5-9C FinBERT Sentiment Context Features

Status: `{manifest["status"]}`.

## Purpose

This step converts headline-level FinBERT sentiment into sample-level context
features aligned to the Stage4 image samples. For a sample ending at date `t`,
news is included only when:

```text
t-window <= news_date <= t-1
```

Same-day news is excluded, matching the Stage5 leakage policy.

## Inputs

- FinBERT items: `{manifest["inputs"]["finbert_items"]}`
- Sample table: `{manifest["inputs"]["sample_table"]}`
- Windows: `{windows}`
- Headlines: `{manifest["num_finbert_items"]}`
- Samples: `{manifest["num_samples"]}`

## Coverage

{chr(10).join(coverage_lines)}

## Sentiment Regime Pattern

{chr(10).join(sentiment_lines)}

## Exported Context

- Feature columns: `{len(feature_cols)}`
- Output CSV: `{manifest["outputs"]["context_features"]}`
- Scaler: `{manifest["outputs"]["scaler"]}`
- Split summary: `{manifest["outputs"]["split_summary"]}`
- Feature summary: `{manifest["outputs"]["feature_summary"]}`

## Interpretation

The feature is usable for `5-9D` if coverage is complete, count alignment is
stable, and the train/test sentiment distribution is not collapsed. This step
does not prove predictive value; it only prepares a leakage-safe sentiment
regime context.
"""


def main() -> None:
    args = parse_args()
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    DATA_INVENTORY.mkdir(parents=True, exist_ok=True)
    windows = sorted({int(window) for window in args.windows})
    if not windows or any(window <= 0 for window in windows):
        raise ValueError("--windows must contain positive integers.")

    finbert_items_path = resolve_finbert_items(args)
    sample_table_path = resolve_sample_table(args.sample_table)

    items = pd.read_csv(finbert_items_path, parse_dates=["news_date"])
    samples = read_sample_table(sample_table_path)
    features = aggregate_context(
        samples=samples,
        items=items,
        windows=windows,
        high_confidence_threshold=float(args.high_confidence_threshold),
    )
    scaler = features.attrs.get("scaler", {})
    split_summary = build_split_summary(features, windows)
    feature_summary = build_feature_summary(features)

    output_root = ROOT / "outputs/stage5/finbert_context" / args.output_name
    output_root.mkdir(parents=True, exist_ok=True)
    context_features_path = output_root / "stage5_finbert_context_features.csv"
    scaler_path = output_root / "stage5_finbert_context_scaler.json"
    context_features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(context_features_path, index=False)
    write_json(scaler_path, {"fit_on": "train", "scaler": scaler})

    prefix = args.output_prefix
    split_summary_path = REPORT_TABLES / f"{prefix}_split_summary.csv"
    feature_summary_path = REPORT_TABLES / f"{prefix}_feature_summary.csv"
    report_path = REPORT_TABLES / f"{prefix}_report.md"
    manifest_path = DATA_INVENTORY / f"{prefix}_manifest.json"

    split_summary.to_csv(split_summary_path, index=False)
    feature_summary.to_csv(feature_summary_path, index=False)

    manifest = {
        "status": "ok",
        "stage": "5-9C",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": args.run_id,
        "windows": windows,
        "strict_lag": "same-day news excluded; newest allowed news date is image_end_date - 1 day",
        "high_confidence_threshold": float(args.high_confidence_threshold),
        "num_finbert_items": int(len(items)),
        "num_samples": int(len(features)),
        "num_context_features": int(len([col for col in features.columns if col.startswith("finbert_")])),
        "inputs": {
            "finbert_items": str(finbert_items_path),
            "sample_table": str(sample_table_path),
        },
        "outputs": {
            "context_features": str(context_features_path),
            "scaler": str(scaler_path),
            "split_summary": str(split_summary_path),
            "feature_summary": str(feature_summary_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)
    write_json(REPORT_TABLES / f"{prefix}_manifest.json", manifest)
    report_path.write_text(
        build_report(
            manifest=manifest,
            split_summary=split_summary,
            features=features,
            windows=windows,
        ),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
