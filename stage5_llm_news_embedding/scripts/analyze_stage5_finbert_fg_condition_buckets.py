#!/usr/bin/env python3
"""Analyze Stage 5 FinBERT+F&G correction/regression condition buckets.

This is Stage 5 5-11. It joins the saved Stage2-vs-Stage5 transition table with
the prebuilt FinBERT+F&G context features and asks where corrections and
regressions happen.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = ROOT.parent / "5_9e_results"
DEFAULT_CONTEXT_NAME = "stage5_finbert_fg_context_i60_ohlc_ma_vb_r20_stage5_finbert_fg_sentiment_v1"
DEFAULT_PREFIX = "stage5_5_11_finbert_fg_condition_analysis"


SELECTED_FEATURES = [
    "fg_value",
    "fg_mean_60",
    "fg_delta_60",
    "fg_std_60",
    "finbert_news_count_7d",
    "finbert_news_count_20d",
    "finbert_news_count_60d",
    "finbert_positive_ratio_20d",
    "finbert_negative_ratio_20d",
    "finbert_sentiment_mean_7d",
    "finbert_sentiment_mean_20d",
    "finbert_sentiment_mean_60d",
    "finbert_confidence_weighted_sentiment_mean_20d",
    "finbert_confidence_weighted_sentiment_mean_60d",
    "finbert_news_fg_proxy_20d",
    "finbert_news_fg_proxy_60d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--context-name", default=DEFAULT_CONTEXT_NAME)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "tables")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--min-bucket-size", type=int, default=50)
    return parser.parse_args()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV missing: {path}")
    return pd.read_csv(path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def transition_table(result_root: Path) -> Path:
    return (
        result_root
        / "reports/tables/stage5_9e_stage2_vs_finbert_fg_correction_all_transitions.csv"
    )


def context_table(result_root: Path, context_name: str, seed: int) -> Path:
    return (
        result_root
        / "outputs/stage4/context"
        / context_name
        / f"seed_{seed}"
        / "context_features.csv"
    )


def load_contexts(result_root: Path, context_name: str, seeds: Iterable[int]) -> pd.DataFrame:
    frames = []
    for seed in sorted(set(int(s) for s in seeds)):
        path = context_table(result_root, context_name, seed)
        frame = read_csv_required(path)
        frame = frame[frame["split"].astype(str).eq("test")].copy()
        frame["run_seed"] = seed
        keep = ["run_seed", "sample_index", "image_end_date", "label_end_date"]
        for column in SELECTED_FEATURES:
            if column in frame.columns:
                keep.append(column)
        frames.append(frame[keep])
    return pd.concat(frames, ignore_index=True)


def add_transition_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["true_prob_stage2"] = np.where(
        out["label"].astype(int).eq(1), out["stage2_prob_up"], 1.0 - out["stage2_prob_up"]
    )
    out["true_prob_film"] = np.where(
        out["label"].astype(int).eq(1), out["film_prob_up"], 1.0 - out["film_prob_up"]
    )
    out["true_prob_delta"] = out["true_prob_film"] - out["true_prob_stage2"]
    out["stage2_uncertainty_abs"] = (out["stage2_prob_up"] - 0.5).abs()
    out["stage2_uncertain_45_55"] = out["stage2_prob_up"].between(0.45, 0.55)
    out["stage2_uncertain_40_60"] = out["stage2_prob_up"].between(0.40, 0.60)
    out["stage2_high_up_conf"] = out["stage2_prob_up"].ge(0.70)
    out["stage2_high_down_conf"] = out["stage2_prob_up"].le(0.30)
    out["stage2_high_conf"] = out["stage2_high_up_conf"] | out["stage2_high_down_conf"]
    out["changed_prediction"] = out["stage2_pred_class"].ne(out["film_pred_class"])
    return out


def fg_regime(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value <= 25:
        return "extreme_fear"
    if value <= 45:
        return "fear"
    if value < 55:
        return "neutral"
    if value < 75:
        return "greed"
    return "extreme_greed"


def add_bucket_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fg_value" in out.columns:
        out["bucket_fg_regime"] = out["fg_value"].map(fg_regime)

    for flag in [
        "stage2_uncertain_45_55",
        "stage2_uncertain_40_60",
        "stage2_high_conf",
        "stage2_high_up_conf",
        "stage2_high_down_conf",
    ]:
        out[f"bucket_{flag}"] = np.where(out[flag], "yes", "no")

    for feature in SELECTED_FEATURES:
        if feature not in out.columns:
            continue
        values = pd.to_numeric(out[feature], errors="coerce")
        if values.notna().sum() < 10 or values.nunique(dropna=True) < 3:
            continue
        q20, q80 = values.quantile([0.20, 0.80])
        bucket = pd.Series("middle_60pct", index=out.index, dtype="object")
        bucket[values.le(q20)] = "low_20pct"
        bucket[values.ge(q80)] = "high_20pct"
        bucket[values.isna()] = "missing"
        out[f"bucket_{feature}_q20_80"] = bucket
    return out


def summarize_group(frame: pd.DataFrame) -> dict:
    total = int(len(frame))
    correction = int(frame["transition"].eq("correction").sum())
    regression = int(frame["transition"].eq("regression").sum())
    changed = int(frame["changed_prediction"].sum())
    stage2_acc = float(frame["stage2_correct"].mean()) if total else np.nan
    film_acc = float(frame["film_correct"].mean()) if total else np.nan
    return {
        "n": total,
        "stage2_acc": stage2_acc,
        "film_acc": film_acc,
        "delta_acc": film_acc - stage2_acc,
        "corrections": correction,
        "regressions": regression,
        "net_corrections": correction - regression,
        "correction_rate": correction / total if total else np.nan,
        "regression_rate": regression / total if total else np.nan,
        "net_correction_rate": (correction - regression) / total if total else np.nan,
        "changed_predictions": changed,
        "changed_prediction_rate": changed / total if total else np.nan,
        "label_positive_rate": float(frame["label"].mean()) if total else np.nan,
        "stage2_pred_up_rate": float(frame["stage2_pred_class"].mean()) if total else np.nan,
        "film_pred_up_rate": float(frame["film_pred_class"].mean()) if total else np.nan,
        "mean_stage2_prob_up": float(frame["stage2_prob_up"].mean()) if total else np.nan,
        "mean_film_prob_up": float(frame["film_prob_up"].mean()) if total else np.nan,
        "mean_prob_up_delta": float(frame["prob_up_delta"].mean()) if total else np.nan,
        "mean_true_prob_delta": float(frame["true_prob_delta"].mean()) if total else np.nan,
    }


def bucket_summary(df: pd.DataFrame, min_bucket_size: int) -> pd.DataFrame:
    rows = []
    rows.append({"bucket_family": "all", "bucket": "all", **summarize_group(df)})
    bucket_cols = [c for c in df.columns if c.startswith("bucket_")]
    for column in bucket_cols:
        family = column.removeprefix("bucket_")
        for bucket, frame in df.groupby(column, dropna=False):
            if len(frame) < min_bucket_size:
                continue
            rows.append(
                {
                    "bucket_family": family,
                    "bucket": str(bucket),
                    **summarize_group(frame),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["bucket_family", "bucket"], kind="stable"
    )


def transition_context_summary(df: pd.DataFrame) -> pd.DataFrame:
    features = [f for f in SELECTED_FEATURES if f in df.columns]
    rows = []
    for transition, frame in df.groupby("transition"):
        row = {"transition": transition, "n": int(len(frame))}
        for feature in features:
            values = pd.to_numeric(frame[feature], errors="coerce")
            row[f"{feature}_mean"] = values.mean()
            row[f"{feature}_median"] = values.median()
        rows.append(row)
    return pd.DataFrame(rows)


def selected_samples(df: pd.DataFrame, per_seed: int = 6) -> pd.DataFrame:
    frames = []
    for seed, seed_frame in df.groupby("run_seed"):
        corr = seed_frame[seed_frame["transition"].eq("correction")].copy()
        reg = seed_frame[seed_frame["transition"].eq("regression")].copy()
        if not corr.empty:
            frames.append(
                corr.sort_values("true_prob_delta", ascending=False).head(per_seed)
            )
        if not reg.empty:
            frames.append(
                reg.sort_values("true_prob_delta", ascending=True).head(per_seed)
            )
    if not frames:
        return pd.DataFrame()

    keep = [
        "run_seed",
        "sample_index",
        "Date",
        "label",
        "transition",
        "stage2_prob_up",
        "film_prob_up",
        "prob_up_delta",
        "true_prob_stage2",
        "true_prob_film",
        "true_prob_delta",
        "fg_value",
        "fg_mean_60",
        "fg_delta_60",
        "fg_std_60",
        "finbert_news_count_20d",
        "finbert_sentiment_mean_20d",
        "finbert_confidence_weighted_sentiment_mean_20d",
        "finbert_news_fg_proxy_20d",
        "finbert_positive_ratio_20d",
        "finbert_negative_ratio_20d",
    ]
    out = pd.concat(frames, ignore_index=True)
    return out[[column for column in keep if column in out.columns]]


def markdown_table(df: pd.DataFrame, columns: list[str], n: int = 12) -> str:
    view = df[columns].head(n).copy()
    return view.to_markdown(index=False, floatfmt=".6f")


def write_report(
    path: Path,
    merged: pd.DataFrame,
    buckets: pd.DataFrame,
    transition_context: pd.DataFrame,
    selected: pd.DataFrame,
    outputs: dict[str, str],
) -> None:
    totals = summarize_group(merged)
    positive_buckets = buckets[
        (buckets["bucket_family"].ne("all"))
        & (buckets["n"].ge(100))
        & (buckets["net_corrections"].gt(0))
    ].sort_values(["net_correction_rate", "net_corrections"], ascending=False)
    negative_buckets = buckets[
        (buckets["bucket_family"].ne("all"))
        & (buckets["n"].ge(100))
        & (buckets["net_corrections"].lt(0))
    ].sort_values(["net_correction_rate", "net_corrections"], ascending=True)

    lines = [
        "# 5-11 FinBERT+F&G Conditional Correction Analysis",
        "",
        "## Purpose",
        "",
        "This analysis checks where the Stage5 FinBERT+F&G bounded FiLM model",
        "corrects or regresses relative to the frozen Stage2 `I60/R20/ohlc_ma_vb`",
        "visual baseline. It is designed for the thesis interpretability chapter,",
        "not as a new training run.",
        "",
        "## Overall Transition Summary",
        "",
        f"- Matched decisions: `{totals['n']}`",
        f"- Stage2 accuracy: `{totals['stage2_acc']:.6f}`",
        f"- Stage5 accuracy: `{totals['film_acc']:.6f}`",
        f"- Delta accuracy: `{totals['delta_acc']:.6f}`",
        f"- Corrections: `{totals['corrections']}`",
        f"- Regressions: `{totals['regressions']}`",
        f"- Net corrections: `{totals['net_corrections']}`",
        f"- Changed prediction rate: `{totals['changed_prediction_rate']:.6f}`",
        f"- Mean probability-up delta: `{totals['mean_prob_up_delta']:.6f}`",
        "",
        "Interpretation: the model changes only a small subset of decisions.",
        "The net correction is positive, but the effect is modest and should be",
        "presented as conditional correction evidence rather than a large accuracy",
        "gain.",
        "",
        "## Transition-Level Context Means",
        "",
        markdown_table(
            transition_context,
            [
                "transition",
                "n",
                "fg_value_mean",
                "fg_mean_60_mean",
                "finbert_news_count_20d_mean",
                "finbert_sentiment_mean_20d_mean",
                "finbert_news_fg_proxy_20d_mean",
            ],
            n=10,
        ),
        "",
        "## Positive Buckets",
        "",
        markdown_table(
            positive_buckets,
            [
                "bucket_family",
                "bucket",
                "n",
                "delta_acc",
                "corrections",
                "regressions",
                "net_corrections",
                "net_correction_rate",
                "mean_prob_up_delta",
            ],
            n=12,
        )
        if not positive_buckets.empty
        else "No positive bucket with n >= 100.",
        "",
        "## Negative Buckets",
        "",
        markdown_table(
            negative_buckets,
            [
                "bucket_family",
                "bucket",
                "n",
                "delta_acc",
                "corrections",
                "regressions",
                "net_corrections",
                "net_correction_rate",
                "mean_prob_up_delta",
            ],
            n=12,
        )
        if not negative_buckets.empty
        else "No negative bucket with n >= 100.",
        "",
        "## Selected Samples For 5-12",
        "",
        f"Selected sample rows for targeted Grad-CAM/modulation export: `{len(selected)}`.",
        "Use the selected-samples CSV as the input list for Stage5 `5-12`.",
        "",
        "## Written Outputs",
        "",
    ]
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Thesis Use",
            "",
            "Use this result to support a cautious claim: FinBERT+F&G context",
            "adds a small, conservative correction layer to the visual baseline.",
            "It should not be described as a large performance improvement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transitions = read_csv_required(transition_table(args.result_root))
    contexts = load_contexts(args.result_root, args.context_name, transitions["run_seed"].unique())
    merged = transitions.merge(
        contexts,
        on=["run_seed", "sample_index"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(transitions):
        raise RuntimeError("Join changed row count unexpectedly.")
    missing_context = int(merged["image_end_date"].isna().sum())
    if missing_context:
        raise RuntimeError(f"Missing context rows after join: {missing_context}")

    merged = add_bucket_columns(add_transition_fields(merged))
    buckets = bucket_summary(merged, args.min_bucket_size)
    transition_context = transition_context_summary(merged)
    selected = selected_samples(merged)

    outputs = {
        "merged_decisions": str(args.output_dir / f"{args.output_prefix}_merged_decisions.csv"),
        "bucket_summary": str(args.output_dir / f"{args.output_prefix}_bucket_summary.csv"),
        "transition_context_summary": str(
            args.output_dir / f"{args.output_prefix}_transition_context_summary.csv"
        ),
        "selected_samples_for_5_12": str(
            args.output_dir / f"{args.output_prefix}_selected_samples_for_5_12.csv"
        ),
        "report": str(args.output_dir / f"{args.output_prefix}_report.md"),
        "manifest": str(args.output_dir / f"{args.output_prefix}_manifest.json"),
    }

    merged.to_csv(outputs["merged_decisions"], index=False)
    buckets.to_csv(outputs["bucket_summary"], index=False)
    transition_context.to_csv(outputs["transition_context_summary"], index=False)
    selected.to_csv(outputs["selected_samples_for_5_12"], index=False)
    write_report(
        Path(outputs["report"]),
        merged=merged,
        buckets=buckets,
        transition_context=transition_context,
        selected=selected,
        outputs=outputs,
    )
    write_json(
        Path(outputs["manifest"]),
        {
            "status": "ok",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "result_root": str(args.result_root),
            "context_name": args.context_name,
            "num_rows": int(len(merged)),
            "num_selected_samples_for_5_12": int(len(selected)),
            "outputs": outputs,
        },
    )

    print(json.dumps({"status": "ok", "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
