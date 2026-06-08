#!/usr/bin/env python3
"""Fit train-only PCA/SVD reducers for Stage 5 embedding context vectors."""

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
        "--context-root",
        default=str(ROOT / "outputs/stage5/embedding_context/openai_text_embedding_3_small"),
        help="Directory containing 5-6 metadata and embedding context arrays.",
    )
    parser.add_argument("--dims", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--windows", type=int, nargs="+", default=[7, 20, 60])
    parser.add_argument(
        "--aggregations",
        nargs="+",
        default=["mean", "decay_mean"],
        choices=["mean", "decay_mean"],
        help="Vector aggregations exported by 5-6 to reduce.",
    )
    parser.add_argument(
        "--output-name",
        default="openai_text_embedding_3_small",
        help="Name used under outputs/stage5/embedding_svd.",
    )
    return parser.parse_args()


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_metadata(context_root: Path) -> pd.DataFrame:
    path = context_root / "stage5_news_embedding_context_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"5-6 metadata missing: {path}")
    metadata = pd.read_csv(path)
    required = {"split", "sample_index", "image_end_date", "label_end_date"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"5-6 metadata is missing required columns: {missing}")
    if not metadata["split"].eq("train").any():
        raise ValueError("5-6 metadata has no train split rows; cannot fit train-only reducer.")
    return metadata


def load_context_array(context_root: Path, aggregation: str) -> np.ndarray:
    file_name = {
        "mean": "stage5_news_embedding_window_mean_vectors.npy",
        "decay_mean": "stage5_news_embedding_window_decay_mean_vectors.npy",
    }[aggregation]
    path = context_root / file_name
    if not path.exists():
        raise FileNotFoundError(f"5-6 context array missing: {path}")
    array = np.load(path, mmap_mode="r")
    if array.ndim != 3:
        raise ValueError(f"Expected 3D context array, got shape {array.shape}: {path}")
    return array


def fit_train_centered_svd(train_matrix: np.ndarray, max_dim: int) -> dict[str, np.ndarray]:
    train = np.asarray(train_matrix, dtype=np.float64)
    train_mean = train.mean(axis=0)
    centered = train - train_mean
    if not np.isfinite(centered).all():
        raise ValueError("Train matrix contains NaN or inf values.")

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    max_rank = min(int(max_dim), vt.shape[0])
    components = vt[:max_rank].astype(np.float32)
    singular_values = singular_values[:max_rank].astype(np.float64)

    total_variance = float(np.var(centered, axis=0, ddof=1).sum())
    if total_variance <= 0:
        explained = np.zeros(max_rank, dtype=np.float64)
        explained_ratio = np.zeros(max_rank, dtype=np.float64)
    else:
        explained = (singular_values**2) / max(centered.shape[0] - 1, 1)
        explained_ratio = explained / total_variance

    return {
        "train_mean": train_mean.astype(np.float32),
        "components": components,
        "singular_values": singular_values.astype(np.float32),
        "explained_variance_ratio": explained_ratio.astype(np.float64),
        "total_variance": np.array(total_variance, dtype=np.float64),
    }


def transform(matrix: np.ndarray, train_mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    return (values - train_mean.astype(np.float32)) @ components.T.astype(np.float32)


def count_feature_frame(metadata: pd.DataFrame, windows: list[int]) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.DataFrame(index=metadata.index)
    feature_columns: list[str] = []
    train_mask = metadata["split"].eq("train").to_numpy()

    for window in windows:
        count_col = f"embedding_news_count_{window}d"
        missing_col = f"embedding_missing_{window}d"
        if count_col not in metadata.columns:
            raise ValueError(f"Metadata is missing count column: {count_col}")
        count = pd.to_numeric(metadata[count_col], errors="coerce").fillna(0.0).astype(float)
        train_count = count.loc[train_mask]
        mean = float(train_count.mean())
        std = float(train_count.std(ddof=0))
        if std <= 0 or not np.isfinite(std):
            std = 1.0

        raw_name = f"stage5_news_count_{window}d"
        z_name = f"stage5_news_count_z_{window}d"
        frame[raw_name] = count
        frame[z_name] = (count - mean) / std
        feature_columns.extend([raw_name, z_name])

        if missing_col in metadata.columns:
            miss_name = f"stage5_news_missing_{window}d"
            frame[miss_name] = metadata[missing_col].astype(bool).astype(int)
            feature_columns.append(miss_name)

    return frame, feature_columns


def base_metadata_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    columns = ["split", "sample_index", "image_end_date", "label_end_date"]
    optional = ["strict_policy", "same_day_news_count_excluded"]
    columns.extend([column for column in optional if column in metadata.columns])
    return metadata[columns].copy()


def main() -> None:
    args = parse_args()
    dims = sorted({int(value) for value in args.dims})
    windows = sorted({int(value) for value in args.windows})
    if not dims or any(value <= 0 for value in dims):
        raise ValueError("--dims must contain positive integers.")
    if not windows or any(value <= 0 for value in windows):
        raise ValueError("--windows must contain positive integers.")

    context_root = Path(args.context_root)
    metadata = read_metadata(context_root)
    train_mask = metadata["split"].eq("train").to_numpy()
    max_dim = max(dims)
    output_root = ROOT / "outputs/stage5/embedding_svd" / str(args.output_name)
    reducer_root = output_root / "reducers"
    feature_root = output_root / "features"
    reducer_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    DATA_INVENTORY.mkdir(parents=True, exist_ok=True)

    base_frame = base_metadata_frame(metadata)
    count_frame, count_feature_columns = count_feature_frame(metadata, windows)
    summary_rows: list[dict[str, Any]] = []
    output_features: dict[str, str] = {}

    for aggregation in args.aggregations:
        vectors = load_context_array(context_root, aggregation)
        if vectors.shape[0] != len(metadata):
            raise ValueError(
                f"Metadata/vector row mismatch for {aggregation}: "
                f"{len(metadata)} vs {vectors.shape[0]}"
            )
        if vectors.shape[1] != len(windows):
            raise ValueError(
                f"Window count mismatch for {aggregation}: expected {len(windows)}, "
                f"got {vectors.shape[1]}"
            )

        transformed_by_window: dict[int, np.ndarray] = {}
        for window_pos, window in enumerate(windows):
            matrix = np.asarray(vectors[:, window_pos, :], dtype=np.float32)
            fit = fit_train_centered_svd(matrix[train_mask], max_dim=max_dim)
            components = fit["components"]
            transformed = transform(matrix, fit["train_mean"], components).astype(np.float32)
            transformed_by_window[window] = transformed

            prefix = f"{aggregation}_{window}d"
            np.save(reducer_root / f"{prefix}_train_mean.npy", fit["train_mean"])
            np.save(reducer_root / f"{prefix}_components.npy", components)
            np.save(reducer_root / f"{prefix}_singular_values.npy", fit["singular_values"])
            np.save(
                reducer_root / f"{prefix}_explained_variance_ratio.npy",
                fit["explained_variance_ratio"],
            )

            cumulative = np.cumsum(fit["explained_variance_ratio"])
            for dim in dims:
                summary_rows.append(
                    {
                        "aggregation": aggregation,
                        "window_days": int(window),
                        "dimension": int(dim),
                        "fit_split": "train",
                        "train_rows": int(train_mask.sum()),
                        "all_rows": int(len(metadata)),
                        "input_embedding_dim": int(matrix.shape[1]),
                        "explained_variance_ratio": float(cumulative[dim - 1]),
                        "component_1_explained_variance_ratio": float(
                            fit["explained_variance_ratio"][0]
                        ),
                        "reducer_components_path": str(
                            reducer_root / f"{prefix}_components.npy"
                        ),
                    }
                )

        for dim in dims:
            svd_columns: dict[str, np.ndarray] = {}
            feature_columns = list(count_feature_columns)
            for window in windows:
                values = transformed_by_window[window][:, :dim]
                for component_idx in range(dim):
                    column = (
                        f"stage5_news_embed_{aggregation}_{window}d_"
                        f"svd{dim}_{component_idx:02d}"
                    )
                    svd_columns[column] = values[:, component_idx]
                    feature_columns.append(column)

            svd_frame = pd.DataFrame(svd_columns, index=metadata.index)
            feature_frame = pd.concat(
                [base_frame, count_frame, svd_frame],
                axis=1,
            ).copy()
            feature_frame["stage5_embedding_aggregation"] = aggregation
            feature_frame["stage5_embedding_svd_dim"] = int(dim)
            feature_frame["stage5_embedding_fit_split"] = "train"
            feature_frame["stage5_embedding_feature_count"] = int(len(feature_columns))
            feature_frame["stage5_embedding_feature_columns"] = "|".join(feature_columns)

            dim_dir = feature_root / aggregation / f"dim_{dim}"
            dim_dir.mkdir(parents=True, exist_ok=True)
            feature_path = dim_dir / "stage5_news_embedding_svd_context_features.csv"
            feature_frame.to_csv(feature_path, index=False)
            output_features[f"{aggregation}_dim_{dim}"] = str(feature_path)

    summary = pd.DataFrame(summary_rows)
    summary_path = REPORT_TABLES / "stage5_news_embedding_svd_grid_summary.csv"
    summary.to_csv(summary_path, index=False)

    combined_summary = (
        summary.groupby(["aggregation", "dimension"], as_index=False)
        .agg(
            mean_window_explained_variance_ratio=(
                "explained_variance_ratio",
                "mean",
            ),
            min_window_explained_variance_ratio=(
                "explained_variance_ratio",
                "min",
            ),
            max_window_explained_variance_ratio=(
                "explained_variance_ratio",
                "max",
            ),
        )
        .sort_values(["aggregation", "dimension"])
    )
    combined_summary_path = REPORT_TABLES / "stage5_news_embedding_svd_grid_combined_summary.csv"
    combined_summary.to_csv(combined_summary_path, index=False)

    manifest = {
        "status": "ok",
        "stage": "5-7",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "context_root": str(context_root),
        "output_root": str(output_root),
        "fit_split": "train",
        "num_rows": int(len(metadata)),
        "train_rows": int(train_mask.sum()),
        "windows": windows,
        "aggregations": list(args.aggregations),
        "dimensions": dims,
        "method": "train-centered PCA via numpy SVD; reducers fit on train split only",
        "count_features": count_feature_columns,
        "outputs": {
            "summary": str(summary_path),
            "combined_summary": str(combined_summary_path),
            "feature_tables": output_features,
            "reducer_root": str(reducer_root),
        },
    }
    manifest_path = DATA_INVENTORY / "stage5_news_embedding_svd_manifest.json"
    report_manifest_path = REPORT_TABLES / "stage5_news_embedding_svd_manifest.json"
    write_json(manifest_path, manifest)
    write_json(report_manifest_path, manifest)

    best_rows = combined_summary.loc[combined_summary["dimension"].isin(dims)].copy()
    report_lines = [
        "# 5-7 Train-Only SVD/PCA Dimension Grid",
        "",
        "Status: completed.",
        "",
        "## Protocol",
        "",
        "- Reducer fit split: `train` only.",
        "- Validation/test are transformed with the train-fitted mean and components.",
        "- Method: train-centered PCA implemented with NumPy SVD.",
        f"- Windows: `{windows}`.",
        f"- Aggregations: `{list(args.aggregations)}`.",
        f"- Dimensions: `{dims}`.",
        "",
        "## Mean Explained Variance Across 7/20/60 Windows",
        "",
        best_rows.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Outputs",
        "",
        f"- Summary: `{summary_path}`",
        f"- Combined summary: `{combined_summary_path}`",
        f"- Manifest: `{manifest_path}`",
        f"- Feature tables root: `{feature_root}`",
        f"- Reducer root: `{reducer_root}`",
        "",
        "## Next",
        "",
        "Use the best compact feature tables in `5-8A`: Stage2 frozen + "
        "embedding-only bounded FiLM.",
        "",
    ]
    report_path = REPORT_TABLES / "stage5_news_embedding_svd_grid_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
