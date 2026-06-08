#!/usr/bin/env python3
"""Aggregate headline embeddings into sample-level trailing-window contexts."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-root",
        default=str(ROOT / "outputs/stage5/embeddings/openai_text_embedding_3_small"),
    )
    parser.add_argument(
        "--sample-counts",
        default="",
        help=(
            "Optional sample table. If omitted, prefer Stage 4 deduped headline-window "
            "table and fall back to the Stage 5 raw coverage table."
        ),
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[7, 20, 60])
    parser.add_argument(
        "--output-name",
        default="openai_text_embedding_3_small",
        help="Name used under outputs/stage5/embedding_context.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_sample_table(user_path: str) -> Path:
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"Sample table missing: {path}")
        return path

    candidates = [
        ROOT.parent
        / "stage4_film_conditioning/outputs/stage4/news/stage4_news_headline_windows_i60_r20/sample_headline_windows.parquet",
        REPORT_TABLES / "stage5_news_alignment_sample_counts.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find a Stage 5/Stage 4 sample table for aggregation.")


def read_sample_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_embedding_artifacts(embedding_root: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    item_manifest_path = embedding_root / "embedding_item_manifest.csv"
    vector_path = embedding_root / "embedding_vectors.npy"
    failure_path = embedding_root / "embedding_failures.csv"

    if not item_manifest_path.exists():
        raise FileNotFoundError(f"Embedding item manifest missing: {item_manifest_path}")
    if not vector_path.exists():
        raise FileNotFoundError(f"Embedding vector matrix missing: {vector_path}")

    item_manifest = pd.read_csv(item_manifest_path)
    vectors = np.load(vector_path, mmap_mode="r")
    if len(item_manifest) != vectors.shape[0]:
        raise ValueError(
            f"Embedding row mismatch: item_manifest={len(item_manifest)} vectors={vectors.shape[0]}"
        )
    if "embedding_vector_index" not in item_manifest.columns:
        raise ValueError("embedding_item_manifest.csv must include embedding_vector_index.")
    if item_manifest["embedding_vector_index"].min() != 0:
        raise ValueError("embedding_vector_index must start at 0.")
    if item_manifest["embedding_vector_index"].max() != len(item_manifest) - 1:
        raise ValueError("embedding_vector_index must end at len(items)-1.")

    cache_manifest_path = ROOT / "data_inventory/openai_text_embedding_3_small_cache_manifest.json"
    cache_manifest = read_json(cache_manifest_path) if cache_manifest_path.exists() else {}
    if failure_path.exists():
        failures = pd.read_csv(failure_path)
        if not failures.empty:
            raise ValueError(f"Embedding failure log is not empty: {failure_path}")

    return item_manifest, vectors, cache_manifest


def aggregate_windows(
    *,
    samples: pd.DataFrame,
    items: pd.DataFrame,
    vectors: np.ndarray,
    windows: list[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    samples = samples.copy().reset_index(drop=True)
    samples["image_end_date_ts"] = pd.to_datetime(samples["image_end_date"]).dt.normalize()

    item_dates = pd.to_datetime(items["news_date"]).dt.normalize()
    order = np.argsort(item_dates.values)
    sorted_dates = item_dates.iloc[order].reset_index(drop=True)
    sorted_vector_indices = items["embedding_vector_index"].to_numpy(dtype=int)[order]
    sorted_sources = items["source_text"].fillna("").astype(str).to_numpy()[order]

    num_samples = len(samples)
    num_windows = len(windows)
    dim = int(vectors.shape[1])
    mean_vectors = np.zeros((num_samples, num_windows, dim), dtype=np.float32)
    decay_vectors = np.zeros((num_samples, num_windows, dim), dtype=np.float32)

    metadata = samples[
        [
            "split",
            "sample_index",
            "image_end_date",
            "label_end_date",
            "strict_policy",
        ]
    ].copy()
    if "same_day_news_count_excluded" in samples.columns:
        metadata["same_day_news_count_excluded"] = samples["same_day_news_count_excluded"]
    elif "same_day_excluded" in samples.columns:
        metadata["same_day_news_count_excluded"] = samples["same_day_excluded"]
    else:
        metadata["same_day_news_count_excluded"] = pd.NA

    sorted_date_values = sorted_dates.to_numpy(dtype="datetime64[ns]")
    for window_pos, window in enumerate(windows):
        counts: list[int] = []
        source_counts: list[int] = []
        missing: list[bool] = []
        start_dates: list[str] = []
        end_dates: list[str] = []

        for sample_pos, row in samples.iterrows():
            image_date = pd.Timestamp(row["image_end_date_ts"])
            start_date = image_date - pd.Timedelta(days=int(window))
            end_date = image_date - pd.Timedelta(days=1)

            left = np.searchsorted(sorted_date_values, np.datetime64(start_date), side="left")
            right = np.searchsorted(sorted_date_values, np.datetime64(end_date), side="right")
            vector_indices = sorted_vector_indices[left:right]
            count = int(len(vector_indices))
            counts.append(count)
            missing.append(count == 0)
            start_dates.append(start_date.date().isoformat())
            end_dates.append(end_date.date().isoformat())

            if count == 0:
                source_counts.append(0)
                continue

            window_vectors = np.asarray(vectors[vector_indices], dtype=np.float32)
            mean_vectors[sample_pos, window_pos, :] = window_vectors.mean(axis=0)

            window_dates = pd.to_datetime(sorted_dates.iloc[left:right]).reset_index(drop=True)
            age_days = (image_date - window_dates).dt.days.to_numpy(dtype=np.float32)
            half_life = max(float(window) / 2.0, 1.0)
            weights = np.exp(-np.log(2.0) * age_days / half_life).astype(np.float32)
            weights_sum = float(weights.sum())
            if weights_sum > 0:
                weights = weights / weights_sum
                decay_vectors[sample_pos, window_pos, :] = (window_vectors * weights[:, None]).sum(axis=0)

            source_values = [value for value in sorted_sources[left:right] if str(value).strip()]
            source_counts.append(len(set(source_values)))

        metadata[f"embedding_window_start_date_{window}d"] = start_dates
        metadata[f"embedding_window_end_date_{window}d"] = end_dates
        metadata[f"embedding_news_count_{window}d"] = counts
        metadata[f"embedding_unique_source_count_{window}d"] = source_counts
        metadata[f"embedding_missing_{window}d"] = missing

        expected_col = None
        for candidate in [f"news_count_{window}d", f"news_count_{window}d_t_minus_1"]:
            if candidate in samples.columns:
                expected_col = candidate
                break
        if expected_col is not None:
            metadata[f"embedding_count_matches_stage4_{window}d"] = (
                metadata[f"embedding_news_count_{window}d"].astype(int)
                == samples[expected_col].astype(int)
            )

    return metadata, mean_vectors, decay_vectors


def build_summary(metadata: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_order = ["train", "validation", "test", "all"]
    for split in split_order:
        frame = metadata if split == "all" else metadata.loc[metadata["split"].eq(split)]
        if frame.empty:
            continue
        row: dict[str, Any] = {"split": split, "num_samples": int(len(frame))}
        for window in windows:
            count_col = f"embedding_news_count_{window}d"
            missing_col = f"embedding_missing_{window}d"
            match_col = f"embedding_count_matches_stage4_{window}d"
            row[f"mean_news_count_{window}d"] = float(frame[count_col].mean())
            row[f"min_news_count_{window}d"] = int(frame[count_col].min())
            row[f"max_news_count_{window}d"] = int(frame[count_col].max())
            row[f"missing_rate_{window}d"] = float(frame[missing_col].mean())
            if match_col in frame.columns:
                row[f"count_match_rate_{window}d"] = float(frame[match_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    windows = sorted({int(window) for window in args.windows})
    if not windows or any(window <= 0 for window in windows):
        raise ValueError("--windows must contain positive integers.")

    embedding_root = Path(args.embedding_root)
    sample_counts_path = resolve_sample_table(str(args.sample_counts))

    items, vectors, cache_manifest = load_embedding_artifacts(embedding_root)
    samples = read_sample_table(sample_counts_path)
    metadata, mean_vectors, decay_vectors = aggregate_windows(
        samples=samples,
        items=items,
        vectors=vectors,
        windows=windows,
    )
    summary = build_summary(metadata, windows)

    output_root = ROOT / "outputs/stage5/embedding_context" / str(args.output_name)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "stage5_news_embedding_context_metadata.csv"
    mean_path = output_root / "stage5_news_embedding_window_mean_vectors.npy"
    decay_path = output_root / "stage5_news_embedding_window_decay_mean_vectors.npy"

    metadata.to_csv(metadata_path, index=False)
    np.save(mean_path, mean_vectors.astype(np.float32))
    np.save(decay_path, decay_vectors.astype(np.float32))

    summary_path = REPORT_TABLES / "stage5_news_embedding_context_summary.csv"
    report_path = REPORT_TABLES / "stage5_news_embedding_context_report.md"
    manifest_path = DATA_INVENTORY / "stage5_news_embedding_context_manifest.json"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "status": "ok",
        "stage": "5-6",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embedding_provider": cache_manifest.get("provider", "openai"),
        "embedding_model": cache_manifest.get("model", "text-embedding-3-small"),
        "embedding_shape": list(vectors.shape),
        "num_samples": int(len(metadata)),
        "sample_table": str(sample_counts_path),
        "windows": windows,
        "aggregation": {
            "mean": "simple arithmetic mean over items in [t-window, t-1]",
            "time_decay_mean": "exponential half-life = window_days / 2, using item age in days",
            "strict_lag": "same-day news excluded; newest allowed news date is image_end_date - 1 day",
        },
        "outputs": {
            "metadata": str(metadata_path),
            "mean_vectors": str(mean_path),
            "time_decay_mean_vectors": str(decay_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)
    write_json(REPORT_TABLES / "stage5_news_embedding_context_manifest.json", manifest)

    all_row = summary.loc[summary["split"].eq("all")].iloc[0]
    report = f"""# 5-6 Trailing-Window Embedding Context

Status: completed.

## Inputs

- Embedding model: `{manifest["embedding_model"]}`
- Embedding matrix shape: `{manifest["embedding_shape"]}`
- Sample count: `{manifest["num_samples"]}`
- Windows: `{windows}`

## Aggregation

For each sample ending at date `t`, the script uses news dates:

```text
t-window <= news_date <= t-1
```

Same-day news is excluded. Two vector aggregations are exported:

- simple mean
- exponential time-decay mean with half-life `window / 2`

## Coverage

- 7d missing rate: `{all_row["missing_rate_7d"]:.4f}`
- 20d missing rate: `{all_row["missing_rate_20d"]:.4f}`
- 60d missing rate: `{all_row["missing_rate_60d"]:.4f}`
- 7d count match rate vs Stage4 audit: `{all_row.get("count_match_rate_7d", float("nan")):.4f}`
- 20d count match rate vs Stage4 audit: `{all_row.get("count_match_rate_20d", float("nan")):.4f}`
- 60d count match rate vs Stage4 audit: `{all_row.get("count_match_rate_60d", float("nan")):.4f}`

## Outputs

- Metadata: `{metadata_path}`
- Mean vectors: `{mean_path}`
- Time-decay mean vectors: `{decay_path}`
- Summary: `{summary_path}`
- Manifest: `{manifest_path}`

## Next

Proceed to `5-7`: train-only SVD/PCA dimension grid over the 7/20/60
embedding context vectors.
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
