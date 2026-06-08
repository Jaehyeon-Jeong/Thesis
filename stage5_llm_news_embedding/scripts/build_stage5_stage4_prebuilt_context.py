#!/usr/bin/env python3
"""Convert Stage 5 compact features into a Stage 4 prebuilt context."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = ROOT.parent
DEFAULT_STAGE4_ROOT = THESIS_ROOT / "stage4_film_conditioning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-table",
        default=str(
            ROOT
            / "outputs/stage5/embedding_svd/openai_text_embedding_3_small/features/mean/dim_16/stage5_news_embedding_svd_context_features.csv"
        ),
        help="5-7 compact feature table for one aggregation/dimension setting.",
    )
    parser.add_argument("--stage4-root", default=str(DEFAULT_STAGE4_ROOT))
    parser.add_argument("--image-window", type=int, default=60)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--context-window", type=int, default=60)
    parser.add_argument("--aggregation", default="mean")
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument(
        "--feature-source",
        choices=["embedding_svd", "finbert"],
        default="embedding_svd",
        help="How to select context feature columns from --feature-table.",
    )
    parser.add_argument(
        "--feature-prefix",
        default="finbert_",
        help="Feature prefix used when --feature-source=finbert.",
    )
    parser.add_argument("--run-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--context-name",
        default="",
        help="Explicit Stage4 context artifact name. If empty, a Stage5 name is built.",
    )
    parser.add_argument(
        "--feature-set-name",
        default="embedding_mean_dim16",
        help="Short feature-set label for reports.",
    )
    parser.add_argument(
        "--write-report-copy",
        action="store_true",
        help="Also write summary/audit/manifest under Stage4 reports/tables.",
    )
    return parser.parse_args()


def safe_suffix(value: str) -> str:
    cleaned: list[str] = []
    previous_sep = False
    for char in str(value).strip().lower():
        if char.isalnum():
            cleaned.append(char)
            previous_sep = False
        elif not previous_sep:
            cleaned.append("_")
            previous_sep = True
    return "".join(cleaned).strip("_")


def context_name_from_args(args: argparse.Namespace) -> str:
    if str(args.context_name).strip():
        return str(args.context_name).strip()
    suffix = safe_suffix(str(args.feature_set_name))
    return (
        f"stage5_embedding_context_i{int(args.image_window)}_{args.image_spec}_"
        f"r{int(args.return_horizon)}_{suffix}"
    )


def read_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stage5 feature table missing: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def select_feature_order(
    frame: pd.DataFrame,
    aggregation: str,
    dimension: int,
    feature_source: str = "embedding_svd",
    feature_prefix: str = "finbert_",
) -> list[str]:
    if str(feature_source) == "finbert":
        excluded_tokens = [
            "_window_start_date_",
            "_window_end_date_",
            "_count_matches_",
            "_missing_",
        ]
        feature_order = []
        for column in frame.columns:
            if not column.startswith(str(feature_prefix)):
                continue
            if any(token in column for token in excluded_tokens):
                continue
            if pd.api.types.is_numeric_dtype(frame[column]) or pd.to_numeric(
                frame[column], errors="coerce"
            ).notna().any():
                feature_order.append(column)
        if not feature_order:
            raise ValueError(f"No numeric FinBERT feature columns found with prefix={feature_prefix!r}.")
        return feature_order

    embed_prefix = f"stage5_news_embed_{aggregation}_"
    embed_token = f"_svd{int(dimension)}_"
    embedding_columns = [
        column
        for column in frame.columns
        if column.startswith(embed_prefix) and embed_token in column
    ]
    embedding_columns = sorted(
        embedding_columns,
        key=lambda name: (
            int(name.split("_")[4].rstrip("d")),
            int(name.rsplit("_", 1)[1]),
        ),
    )
    count_columns = [
        f"stage5_news_count_{window}d"
        for window in [7, 20, 60]
        if f"stage5_news_count_{window}d" in frame.columns
    ]
    feature_order = [*count_columns, *embedding_columns]
    if not embedding_columns:
        raise ValueError(
            f"No embedding SVD columns found for aggregation={aggregation!r}, dim={dimension}."
        )
    return feature_order


def fit_normalize_context(
    frame: pd.DataFrame,
    feature_order: list[str],
    epsilon: float = 1.0e-8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = ["split", "sample_index", "image_end_date", "label_end_date", *feature_order]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError("Stage5 feature table missing column(s): " + ", ".join(missing))

    context = frame[["split", "sample_index", "image_end_date", "label_end_date"]].copy()
    if "strict_policy" in frame.columns:
        context["strict_policy"] = frame["strict_policy"]
    if "same_day_news_count_excluded" in frame.columns:
        context["same_day_news_count_excluded"] = frame["same_day_news_count_excluded"]

    train_mask = frame["split"].astype(str).eq("train")
    if not train_mask.any():
        raise ValueError("Cannot fit Stage5 context scaler: no train split rows.")

    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    normalized_columns: list[str] = []
    feature_columns: dict[str, Any] = {}

    for feature in feature_order:
        raw = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        train_raw = raw.loc[train_mask]
        median = float(train_raw.median())
        if not np.isfinite(median):
            raise ValueError(f"Cannot normalize all-missing feature: {feature}")
        imputed = raw.fillna(median)
        train_imputed = imputed.loc[train_mask]
        mean = float(train_imputed.mean())
        std = float(train_imputed.std(ddof=0))
        if not np.isfinite(std) or std < epsilon:
            std = 1.0

        normalized = (imputed - mean) / max(std, epsilon)
        normalized_name = f"{feature}_normalized"
        feature_columns[feature] = raw
        feature_columns[f"{feature}_missing"] = raw.isna()
        feature_columns[normalized_name] = normalized
        normalized_columns.append(normalized_name)

        medians[feature] = median
        means[feature] = mean
        stds[feature] = std

    context = pd.concat([context, pd.DataFrame(feature_columns, index=context.index)], axis=1)
    scaler = {
        "feature_order": feature_order,
        "normalized_feature_columns": normalized_columns,
        "feature_groups": feature_groups(feature_order),
        "medians": medians,
        "means": means,
        "stds": stds,
        "epsilon": float(epsilon),
        "fit_on": "train",
        "normalization": "train_median_impute_train_zscore",
        "source": "stage5_context_features",
    }
    return context, scaler


def feature_groups(feature_order: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "news_counts": [],
        "embedding_7d": [],
        "embedding_20d": [],
        "embedding_60d": [],
        "finbert_counts": [],
        "finbert_ratios": [],
        "finbert_sentiment": [],
        "finbert_proxy": [],
        "finbert_other": [],
    }
    for feature in feature_order:
        if feature.startswith("stage5_news_count_"):
            groups["news_counts"].append(feature)
        elif feature.startswith("finbert_news_count_") or feature.startswith(
            "finbert_unique_source_count_"
        ):
            groups["finbert_counts"].append(feature)
        elif "_ratio_" in feature or "_prob_mean_" in feature:
            groups["finbert_ratios"].append(feature)
        elif "sentiment" in feature:
            groups["finbert_sentiment"].append(feature)
        elif "fg_proxy" in feature:
            groups["finbert_proxy"].append(feature)
        elif "_7d_" in feature:
            groups["embedding_7d"].append(feature)
        elif "_20d_" in feature:
            groups["embedding_20d"].append(feature)
        elif "_60d_" in feature:
            groups["embedding_60d"].append(feature)
        elif feature.startswith("finbert_"):
            groups["finbert_other"].append(feature)
    return groups


def build_summary(context: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, frame in context.groupby("split", sort=True):
        for feature in feature_order:
            normalized = f"{feature}_normalized"
            raw_values = pd.to_numeric(frame[feature], errors="coerce")
            norm_values = pd.to_numeric(frame[normalized], errors="coerce")
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "num_rows": int(len(frame)),
                    "raw_missing_rate": float(raw_values.isna().mean()),
                    "raw_mean": float(raw_values.mean()),
                    "raw_std": float(raw_values.std(ddof=0)),
                    "normalized_mean": float(norm_values.mean()),
                    "normalized_std": float(norm_values.std(ddof=0)),
                    "normalized_min": float(norm_values.min()),
                    "normalized_max": float(norm_values.max()),
                }
            )
    return pd.DataFrame(rows)


def build_audit(
    *,
    context_name: str,
    feature_table: Path,
    context: pd.DataFrame,
    feature_order: list[str],
    scaler: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_counts = {
        str(split): int(count)
        for split, count in context["split"].astype(str).value_counts().sort_index().items()
    }
    missing_rate_by_split: dict[str, dict[str, float]] = {}
    for split, frame in context.groupby("split", sort=True):
        missing_rate_by_split[str(split)] = {
            feature: float(pd.to_numeric(frame[feature], errors="coerce").isna().mean())
            for feature in feature_order
        }
    return {
        "status": "ok",
        "stage": "stage5-prebuilt-context-build",
        "context_name": context_name,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_feature_table": str(feature_table),
        "image_window": int(args.image_window),
        "image_spec": str(args.image_spec),
        "return_horizon": int(args.return_horizon),
        "context_window": int(args.context_window),
        "feature_set_name": str(args.feature_set_name),
        "feature_source": str(args.feature_source),
        "aggregation": str(args.aggregation),
        "dimension": int(args.dimension),
        "feature_order": feature_order,
        "normalized_feature_columns": scaler["normalized_feature_columns"],
        "context_dim": int(len(feature_order)),
        "split_counts": split_counts,
        "missing_rate_by_split": missing_rate_by_split,
        "warnings": [],
    }


def write_context_for_seed(
    *,
    output_root: Path,
    context_name: str,
    run_seed: int,
    context: pd.DataFrame,
    scaler: dict[str, Any],
    audit: dict[str, Any],
    summary: pd.DataFrame,
) -> dict[str, str]:
    output_dir = output_root / "context" / context_name / f"seed_{int(run_seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "context_features": output_dir / "context_features.csv",
        "context_features_parquet": output_dir / "context_features.parquet",
        "context_scaler": output_dir / "context_scaler.json",
        "context_feature_audit": output_dir / "context_feature_audit.json",
        "context_feature_summary": output_dir / "context_feature_summary.csv",
        "stage5_context_manifest": output_dir / "stage5_context_manifest.json",
    }
    context.to_csv(paths["context_features"], index=False)
    context.to_parquet(paths["context_features_parquet"], index=False)
    paths["context_scaler"].write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    paths["context_feature_audit"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    summary.to_csv(paths["context_feature_summary"], index=False)
    manifest = {
        "status": "ok",
        "stage": "stage5-prebuilt-context-build",
        "run_seed": int(run_seed),
        "context_name": context_name,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    paths["stage5_context_manifest"].write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    args = parse_args()
    feature_table = Path(args.feature_table)
    stage4_root = Path(args.stage4_root)
    output_root = stage4_root / "outputs" / "stage4"
    tables_root = stage4_root / "reports" / "tables"
    output_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    frame = read_feature_table(feature_table)
    feature_order = select_feature_order(
        frame,
        aggregation=str(args.aggregation),
        dimension=int(args.dimension),
        feature_source=str(args.feature_source),
        feature_prefix=str(args.feature_prefix),
    )
    context, scaler = fit_normalize_context(frame, feature_order)
    summary = build_summary(context, feature_order)
    context_name = context_name_from_args(args)
    audit = build_audit(
        context_name=context_name,
        feature_table=feature_table,
        context=context,
        feature_order=feature_order,
        scaler=scaler,
        args=args,
    )

    written_by_seed = {
        str(seed): write_context_for_seed(
            output_root=output_root,
            context_name=context_name,
            run_seed=int(seed),
            context=context,
            scaler=scaler,
            audit=audit,
            summary=summary,
        )
        for seed in args.run_seeds
    }

    report_files: dict[str, str] = {}
    if args.write_report_copy:
        prefix = f"{context_name}_seed{'_'.join(str(seed) for seed in args.run_seeds)}"
        audit_path = tables_root / f"{prefix}_context_feature_audit.json"
        summary_path = tables_root / f"{prefix}_context_feature_summary.csv"
        manifest_path = tables_root / f"{prefix}_context_manifest.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        summary.to_csv(summary_path, index=False)
        manifest = {
            "status": "ok",
            "stage": "stage5-prebuilt-context-build",
            "context_name": context_name,
            "feature_table": str(feature_table),
            "run_seeds": [int(seed) for seed in args.run_seeds],
            "written_by_seed": written_by_seed,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        report_files = {
            "report_audit": str(audit_path),
            "report_summary": str(summary_path),
            "report_manifest": str(manifest_path),
        }

    print(
        json.dumps(
            {
                "status": "ok",
                "stage": "stage5-prebuilt-context-build",
                "feature_source": str(args.feature_source),
                "context_name": context_name,
                "feature_table": str(feature_table),
                "context_dim": len(feature_order),
                "run_seeds": [int(seed) for seed in args.run_seeds],
                "split_counts": audit["split_counts"],
                "written_by_seed": written_by_seed,
                **report_files,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
