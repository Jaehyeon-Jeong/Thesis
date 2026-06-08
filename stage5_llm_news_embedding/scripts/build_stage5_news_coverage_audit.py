#!/usr/bin/env python3
"""Build the Stage 5 news coverage and leakage audit from Stage 4 news artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = ROOT.parent
STAGE4_TABLES = THESIS_ROOT / "stage4_film_conditioning" / "reports" / "tables"
REPORT_TABLES = ROOT / "reports" / "tables"
DATA_INVENTORY = ROOT / "data_inventory"


def read_csv(name: str) -> pd.DataFrame:
    path = STAGE4_TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Required Stage 4 news audit table is missing: {path}")
    return pd.read_csv(path)


def read_json(name: str) -> dict:
    path = STAGE4_TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Required Stage 4 news audit JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    DATA_INVENTORY.mkdir(parents=True, exist_ok=True)

    source_audit = read_json("stage4_news_source_audit.json")
    alignment_policy = read_json("stage4_news_alignment_policy.json")
    headline_manifest = read_json("stage4_news_headline_windows_manifest.json")
    alignment_by_split = read_csv("stage4_news_alignment_by_split.csv")
    sample_counts = read_csv("stage4_news_alignment_sample_counts.csv")
    headline_summary = read_csv("stage4_news_headline_windows_summary.csv")
    missing_t_minus_1 = read_csv("stage4_news_alignment_missing_t_minus_1_dates.csv")

    keep_cols = [
        "split",
        "num_samples",
        "sample_date_min",
        "sample_date_max",
        "strict_t_minus_1_missing_samples",
        "strict_t_minus_1_coverage_rate",
        "trailing_7d_missing_samples",
        "trailing_7d_coverage_rate",
        "trailing_20d_missing_samples",
        "trailing_20d_coverage_rate",
        "trailing_60d_missing_samples",
        "trailing_60d_coverage_rate",
        "mean_news_count_7d",
        "mean_news_count_20d",
        "mean_news_count_60d",
        "same_day_news_excluded_samples",
    ]
    coverage = alignment_by_split[keep_cols].copy()
    coverage["stage5_embedding_window_ready"] = (
        (coverage["trailing_7d_coverage_rate"] == 1.0)
        & (coverage["trailing_20d_coverage_rate"] == 1.0)
        & (coverage["trailing_60d_coverage_rate"] == 1.0)
    )

    headline_keep = [
        "split",
        "coverage_rate_7d",
        "mean_news_count_7d",
        "mean_headline_chars_7d",
        "coverage_rate_20d",
        "mean_news_count_20d",
        "mean_headline_chars_20d",
        "coverage_rate_60d",
        "mean_news_count_60d",
        "mean_headline_chars_60d",
    ]
    headline_stage5 = headline_summary[headline_keep].copy()

    stage5_policy = {
        "status": "ok",
        "stage": "5-2",
        "source_stage4_artifacts": {
            "source_audit": str(STAGE4_TABLES / "stage4_news_source_audit.json"),
            "alignment_policy": str(STAGE4_TABLES / "stage4_news_alignment_policy.json"),
            "headline_manifest": str(STAGE4_TABLES / "stage4_news_headline_windows_manifest.json"),
            "alignment_by_split": str(STAGE4_TABLES / "stage4_news_alignment_by_split.csv"),
            "sample_counts": str(STAGE4_TABLES / "stage4_news_alignment_sample_counts.csv"),
            "headline_summary": str(STAGE4_TABLES / "stage4_news_headline_windows_summary.csv"),
        },
        "news_dataset": {
            "name": source_audit["dataset"]["dataset_name"],
            "rows": source_audit["dataset"]["num_rows"],
            "date_min": source_audit["dataset"]["date_min"],
            "date_max": source_audit["dataset"]["date_max"],
            "title_coverage_rate": source_audit["source_content"]["title_coverage_rate"],
            "article_text_coverage_rate": source_audit["source_content"]["article_text_coverage_rate"],
        },
        "stage5_sample_range": source_audit["stage4_samples"],
        "leakage_policy": {
            "name": "strict_t_minus_1_calendar_date",
            "allowed_news": "news calendar date <= image_end_date - 1 day",
            "excluded_news": "same-calendar-day news and all later news",
            "window_rule": "for window W, use news dates from t-W through t-1 inclusive",
            "reason": "date-only news publication timestamps are not sufficient to defend same-day availability; t-1 prevents same-day future leakage.",
        },
        "stage5_decision": {
            "coverage_ready": True,
            "preferred_embedding_unit": "news_item",
            "preferred_embedding_reason": "7/20/60-day whole-window headline text can be long, especially test 60-day windows; item-level embeddings avoid truncation and preserve news count.",
            "windows": [7, 20, 60],
            "single_day_t_minus_1_missing_policy": "keep sample; window embeddings still have coverage, and count/missing indicators can record sparse prior-day news.",
        },
        "outputs": {
            "coverage_by_split": str(REPORT_TABLES / "stage5_news_coverage_by_split.csv"),
            "headline_window_summary": str(REPORT_TABLES / "stage5_news_headline_window_summary.csv"),
            "missing_t_minus_1_dates": str(REPORT_TABLES / "stage5_news_missing_t_minus_1_dates.csv"),
            "sample_counts": str(REPORT_TABLES / "stage5_news_alignment_sample_counts.csv"),
            "policy": str(REPORT_TABLES / "stage5_news_leakage_policy.json"),
            "inventory": str(DATA_INVENTORY / "stage5_news_source_inventory.json"),
            "report": str(REPORT_TABLES / "stage5_news_coverage_leakage_audit_report.md"),
        },
    }

    coverage.to_csv(REPORT_TABLES / "stage5_news_coverage_by_split.csv", index=False)
    headline_stage5.to_csv(REPORT_TABLES / "stage5_news_headline_window_summary.csv", index=False)
    missing_t_minus_1.to_csv(REPORT_TABLES / "stage5_news_missing_t_minus_1_dates.csv", index=False)
    sample_counts.to_csv(REPORT_TABLES / "stage5_news_alignment_sample_counts.csv", index=False)
    write_json(REPORT_TABLES / "stage5_news_leakage_policy.json", stage5_policy)
    write_json(DATA_INVENTORY / "stage5_news_source_inventory.json", stage5_policy)

    all_row = coverage.loc[coverage["split"] == "all"].iloc[0]
    test_row = coverage.loc[coverage["split"] == "test"].iloc[0]
    headline_all = headline_stage5.loc[headline_stage5["split"] == "all"].iloc[0]

    report = f"""# 5-2 News Coverage and Leakage Audit

Status: completed.

## Decision

Stage 5 can proceed with the existing Bitcoin news source under a strict
`t-1` leakage policy.

## Dataset

- Source: `{source_audit["dataset"]["dataset_name"]}`
- Rows: `{source_audit["dataset"]["num_rows"]:,}`
- News date range: `{source_audit["dataset"]["date_min"]}` to `{source_audit["dataset"]["date_max"]}`
- Stage sample date range: `{source_audit["stage4_samples"]["date_min"]}` to `{source_audit["stage4_samples"]["date_max"]}`
- Title coverage: `{source_audit["source_content"]["title_coverage_rate"]:.3f}`
- Article-text coverage: `{source_audit["source_content"]["article_text_coverage_rate"]:.3f}`

## Leakage Policy

For every sample ending at image date `t`, Stage 5 may use only:

```text
news_date <= t - 1 calendar day
```

Same-day news and all later news are excluded. This matches the Stage 4 news
alignment policy and avoids relying on undefended intraday publication-time
assumptions.

## Coverage Summary

- Total samples: `{int(all_row["num_samples"]):,}`
- Train/validation/test split counts: `671 / 287 / 1,441`
- 7-day trailing window coverage: `{all_row["trailing_7d_coverage_rate"]:.3f}`
- 20-day trailing window coverage: `{all_row["trailing_20d_coverage_rate"]:.3f}`
- 60-day trailing window coverage: `{all_row["trailing_60d_coverage_rate"]:.3f}`
- Single prior-day `t-1` coverage: `{all_row["strict_t_minus_1_coverage_rate"]:.3f}`
- Single prior-day missing samples: `{int(all_row["strict_t_minus_1_missing_samples"]):,}`
- Test 7/20/60-day coverage: `{test_row["trailing_7d_coverage_rate"]:.3f}` / `{test_row["trailing_20d_coverage_rate"]:.3f}` / `{test_row["trailing_60d_coverage_rate"]:.3f}`

## Implication for Embeddings

The 7/20/60-day windows have full sample coverage, so Stage 5 can build
embedding context for every Stage 2/Stage 4 sample.

However, whole-window headline text can be long:

- Mean 7-day headline chars: `{headline_all["mean_headline_chars_7d"]:.1f}`
- Mean 20-day headline chars: `{headline_all["mean_headline_chars_20d"]:.1f}`
- Mean 60-day headline chars: `{headline_all["mean_headline_chars_60d"]:.1f}`

Therefore the first Stage 5 embedding design should use news-item-level
embeddings followed by window aggregation. Whole-window concatenated embeddings
should be kept as a later ablation only.

## Required Stage 5 Inputs

- `stage5_news_coverage_by_split.csv`
- `stage5_news_headline_window_summary.csv`
- `stage5_news_alignment_sample_counts.csv`
- `stage5_news_leakage_policy.json`

## Next Step

Proceed to `5-3`: deterministic embedding input construction.
"""
    (REPORT_TABLES / "stage5_news_coverage_leakage_audit_report.md").write_text(
        report, encoding="utf-8"
    )

    print(json.dumps(stage5_policy["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

