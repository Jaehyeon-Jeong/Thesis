#!/usr/bin/env python3
"""Audit Stage 5 FinBERT headline sentiment outputs.

This is Stage 5 5-9B. It creates sampled audit tables and distribution
diagnostics from the headline-level FinBERT artifact created in 5-9A.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_TABLES = ROOT / "reports" / "tables"
DATA_INVENTORY = ROOT / "data_inventory"
DEFAULT_RUN_ID = "finbert_prosusai_headline_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--input-items", default="", help="Path to 5-9A sentiment items CSV.")
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-prefix", default="stage5_9b_finbert_sentiment_audit")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_input_items(args: argparse.Namespace) -> Path:
    candidates: list[Path] = []
    if args.input_items:
        candidates.append(Path(args.input_items))

    manifest_candidates = [
        DATA_INVENTORY / f"{args.run_id}_manifest.json",
        REPORT_TABLES / f"{args.run_id}_manifest.json",
    ]
    for manifest_path in manifest_candidates:
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            value = manifest.get("outputs", {}).get("finbert_sentiment_items")
            if value:
                candidates.append(Path(value))

    candidates.append(
        ROOT / "outputs/stage5/finbert_sentiment" / args.run_id / "stage5_finbert_sentiment_items.csv"
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find FinBERT sentiment item CSV. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def require_columns(frame: pd.DataFrame) -> None:
    required = [
        "news_item_id",
        "news_date",
        "headline_text",
        "finbert_positive_prob",
        "finbert_negative_prob",
        "finbert_neutral_prob",
        "finbert_label",
        "finbert_confidence",
        "finbert_sentiment_score",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("FinBERT item CSV is missing required columns: " + ", ".join(missing))


def safe_sample(frame: pd.DataFrame, n: int, random_seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=random_seed).copy()


def build_audit_samples(frame: pd.DataFrame, sample_size: int, random_seed: int) -> pd.DataFrame:
    buckets: list[pd.DataFrame] = []

    def add_bucket(name: str, bucket_frame: pd.DataFrame) -> None:
        cols = [
            "news_date",
            "source_text",
            "headline_text",
            "finbert_label",
            "finbert_positive_prob",
            "finbert_negative_prob",
            "finbert_neutral_prob",
            "finbert_confidence",
            "finbert_sentiment_score",
            "news_item_id",
        ]
        available = [column for column in cols if column in bucket_frame.columns]
        out = bucket_frame[available].copy()
        out.insert(0, "audit_bucket", name)
        buckets.append(out)

    add_bucket("random", safe_sample(frame, sample_size, random_seed))
    add_bucket(
        "high_positive",
        frame.sort_values(["finbert_sentiment_score", "finbert_confidence"], ascending=False).head(sample_size),
    )
    add_bucket(
        "high_negative",
        frame.sort_values(["finbert_sentiment_score", "finbert_confidence"], ascending=[True, False]).head(sample_size),
    )
    add_bucket(
        "high_neutral",
        frame.sort_values(["finbert_neutral_prob", "finbert_confidence"], ascending=False).head(sample_size),
    )
    add_bucket(
        "low_confidence",
        frame.sort_values(["finbert_confidence", "finbert_neutral_prob"], ascending=[True, False]).head(sample_size),
    )
    ambiguous = frame.loc[
        frame["finbert_confidence"].lt(0.55) | frame["finbert_sentiment_score"].abs().lt(0.05)
    ].copy()
    add_bucket("ambiguous_score_or_confidence", safe_sample(ambiguous, sample_size, random_seed))

    return pd.concat(buckets, ignore_index=True)


def quantile_table(series: pd.Series, name: str) -> pd.DataFrame:
    qs = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    values = series.quantile(qs)
    return pd.DataFrame({"metric": name, "quantile": qs, "value": values.values})


def build_report(
    *,
    manifest: dict[str, Any],
    label_distribution: pd.DataFrame,
    year_summary: pd.DataFrame,
    quality_summary: dict[str, Any],
) -> str:
    label_lines = "\n".join(
        f"- {row['finbert_label']}: `{int(row['rows'])}` (`{float(row['share']):.4f}`)"
        for _, row in label_distribution.iterrows()
    )
    year_lines = "\n".join(
        f"- {int(row['year'])}: sentiment_mean `{row['sentiment_mean']:.4f}`, "
        f"positive `{row['positive_share']:.4f}`, negative `{row['negative_share']:.4f}`"
        for _, row in year_summary.iterrows()
    )
    warnings = quality_summary.get("warnings", [])
    warning_block = "\n".join(f"- {item}" for item in warnings) if warnings else "- none"

    return f"""# 5-9B FinBERT Sentiment Output Audit

Status: `{manifest["status"]}`.

## Purpose

This audit checks whether the 5-9A FinBERT headline labels are usable for
7/20/60-day news sentiment aggregation. It does not evaluate market prediction
performance.

## Label Distribution

{label_lines}

## Confidence And Quality

- Mean confidence: `{quality_summary["mean_confidence"]:.6f}`
- Median confidence: `{quality_summary["median_confidence"]:.6f}`
- Low-confidence rows `< 0.50`: `{quality_summary["low_confidence_rows"]}`
- Low-confidence share `< 0.50`: `{quality_summary["low_confidence_share"]:.4f}`
- Mean sentiment score: `{quality_summary["mean_sentiment_score"]:.6f}`
- Sentiment std: `{quality_summary["std_sentiment_score"]:.6f}`

Quality warnings:

{warning_block}

## Year-Level Pattern

{year_lines}

## Interpretation

The output is suitable for the next aggregation step if label distribution is
not collapsed, confidence is generally high, and low-confidence samples are
plausibly ambiguous. The generated audit sample table should be reviewed before
treating the feature as final.

## Outputs

- Audit samples: `{manifest["outputs"]["audit_samples"]}`
- Label distribution: `{manifest["outputs"]["label_distribution"]}`
- Year summary: `{manifest["outputs"]["year_summary"]}`
- Daily summary: `{manifest["outputs"]["daily_summary"]}`
- Quantiles: `{manifest["outputs"]["quantiles"]}`
- Manifest: `{manifest["outputs"]["manifest"]}`
"""


def main() -> None:
    args = parse_args()
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    DATA_INVENTORY.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input_items(args)
    frame = pd.read_csv(input_path, parse_dates=["news_date"])
    require_columns(frame)
    frame["year"] = frame["news_date"].dt.year

    prefix = args.output_prefix
    audit_samples_path = REPORT_TABLES / f"{prefix}_audit_samples.csv"
    label_distribution_path = REPORT_TABLES / f"{prefix}_label_distribution.csv"
    year_summary_path = REPORT_TABLES / f"{prefix}_year_summary.csv"
    daily_summary_path = REPORT_TABLES / f"{prefix}_daily_summary.csv"
    quantiles_path = REPORT_TABLES / f"{prefix}_quantiles.csv"
    manifest_path = DATA_INVENTORY / f"{prefix}_manifest.json"
    report_path = REPORT_TABLES / f"{prefix}_report.md"

    label_distribution = (
        frame["finbert_label"]
        .value_counts()
        .rename_axis("finbert_label")
        .reset_index(name="rows")
        .sort_values("finbert_label")
    )
    label_distribution["share"] = label_distribution["rows"] / len(frame)

    year_summary = (
        frame.groupby("year", dropna=False)
        .agg(
            rows=("news_item_id", "count"),
            positive_share=("finbert_label", lambda s: float((s == "positive").mean())),
            negative_share=("finbert_label", lambda s: float((s == "negative").mean())),
            neutral_share=("finbert_label", lambda s: float((s == "neutral").mean())),
            sentiment_mean=("finbert_sentiment_score", "mean"),
            sentiment_std=("finbert_sentiment_score", "std"),
            confidence_mean=("finbert_confidence", "mean"),
        )
        .reset_index()
    )

    daily_summary = (
        frame.groupby("news_date", dropna=False)
        .agg(
            news_count=("news_item_id", "count"),
            positive_share=("finbert_label", lambda s: float((s == "positive").mean())),
            negative_share=("finbert_label", lambda s: float((s == "negative").mean())),
            neutral_share=("finbert_label", lambda s: float((s == "neutral").mean())),
            sentiment_mean=("finbert_sentiment_score", "mean"),
            sentiment_sum=("finbert_sentiment_score", "sum"),
            confidence_mean=("finbert_confidence", "mean"),
        )
        .reset_index()
    )

    quantiles = pd.concat(
        [
            quantile_table(frame["finbert_confidence"], "finbert_confidence"),
            quantile_table(frame["finbert_sentiment_score"], "finbert_sentiment_score"),
            quantile_table(daily_summary["sentiment_mean"], "daily_sentiment_mean"),
            quantile_table(daily_summary["news_count"], "daily_news_count"),
        ],
        ignore_index=True,
    )

    audit_samples = build_audit_samples(frame, args.sample_size, args.random_seed)

    label_share = dict(zip(label_distribution["finbert_label"], label_distribution["share"]))
    warnings: list[str] = []
    if max(label_share.values()) > 0.75:
        warnings.append("label distribution is highly concentrated in one class")
    if label_share.get("positive", 0.0) < 0.10:
        warnings.append("positive class share is below 10%")
    if label_share.get("negative", 0.0) < 0.10:
        warnings.append("negative class share is below 10%")
    low_confidence_rows = int(frame["finbert_confidence"].lt(0.50).sum())
    low_confidence_share = float(low_confidence_rows / len(frame))
    if low_confidence_share > 0.10:
        warnings.append("low-confidence share is above 10%")

    quality_summary = {
        "num_rows": int(len(frame)),
        "mean_confidence": float(frame["finbert_confidence"].mean()),
        "median_confidence": float(frame["finbert_confidence"].median()),
        "low_confidence_rows": low_confidence_rows,
        "low_confidence_share": low_confidence_share,
        "mean_sentiment_score": float(frame["finbert_sentiment_score"].mean()),
        "std_sentiment_score": float(frame["finbert_sentiment_score"].std(ddof=1)),
        "warnings": warnings,
    }

    audit_samples.to_csv(audit_samples_path, index=False)
    label_distribution.to_csv(label_distribution_path, index=False)
    year_summary.to_csv(year_summary_path, index=False)
    daily_summary.to_csv(daily_summary_path, index=False)
    quantiles.to_csv(quantiles_path, index=False)

    manifest = {
        "status": "ok",
        "stage": "5-9B",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_items": str(input_path),
        "run_id": args.run_id,
        "sample_size": int(args.sample_size),
        "random_seed": int(args.random_seed),
        "quality_summary": quality_summary,
        "outputs": {
            "audit_samples": str(audit_samples_path),
            "label_distribution": str(label_distribution_path),
            "year_summary": str(year_summary_path),
            "daily_summary": str(daily_summary_path),
            "quantiles": str(quantiles_path),
            "manifest": str(manifest_path),
            "report": str(report_path),
        },
    }
    write_json(manifest_path, manifest)
    write_json(REPORT_TABLES / f"{prefix}_manifest.json", manifest)
    report_path.write_text(
        build_report(
            manifest=manifest,
            label_distribution=label_distribution,
            year_summary=year_summary,
            quality_summary=quality_summary,
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
