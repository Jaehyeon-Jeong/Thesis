"""Build train-only TF-IDF/SVD vectors for Stage 4 news headline windows.

This is the 4-N4 news-context step. It reads the 4-N3 headline-window table,
fits TF-IDF and SVD only on train-split headline-window documents, then
transforms train/validation/test documents for each window.

Marker: train-only TF-IDF/SVD news vectorizer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from audit_stage4_news_alignment import NEWS_WINDOWS, SPLIT_ORDER
from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import get_stage4_model_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=None)
    parser.add_argument("--return-horizon", type=int, default=None)
    parser.add_argument("--window-days", type=int, nargs="+", default=list(NEWS_WINDOWS))
    parser.add_argument(
        "--input-prefix",
        default="stage4_news_headline_windows",
        help="4-N3 output prefix under outputs/stage4/news.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Optional explicit sample_headline_windows parquet/csv.gz path.",
    )
    parser.add_argument(
        "--output-prefix",
        default="stage4_news_tfidf_svd",
        help="Prefix for outputs/stage4/news and reports/tables files.",
    )
    parser.add_argument("--svd-dim", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--example-rows-per-split", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Build and write train-only TF-IDF/SVD news vectors."""

    args = parse_args()
    if args.svd_dim <= 0:
        raise ValueError("--svd-dim must be positive.")
    if args.max_features <= 0:
        raise ValueError("--max-features must be positive.")
    if args.min_df <= 0:
        raise ValueError("--min-df must be positive.")
    if args.ngram_min <= 0 or args.ngram_max < args.ngram_min:
        raise ValueError("--ngram-min/--ngram-max must define a valid range.")

    config = load_config(args.config)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    stage4_model = get_stage4_model_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
    return_horizon = int(args.return_horizon or stage4_model["primary_return_horizon"])
    window_days = tuple(sorted({int(day) for day in args.window_days}))
    if any(day <= 0 for day in window_days):
        raise ValueError("--window-days must contain only positive integers.")

    input_path = resolve_input_path(
        paths.output_root,
        input_prefix=str(args.input_prefix),
        explicit_path=Path(args.input_path) if args.input_path else None,
        image_window=image_window,
        return_horizon=return_horizon,
    )
    sample_windows = read_table(input_path)
    validate_sample_windows(sample_windows, window_days)
    sample_windows = sort_samples(sample_windows)

    fit_documents, fit_metadata = build_fit_documents(sample_windows, window_days)
    if not fit_documents:
        raise ValueError("No train documents available for TF-IDF/SVD fit.")

    vectorizer = build_vectorizer(args)
    train_tfidf = vectorizer.fit_transform(fit_documents)
    feature_names = vectorizer.get_feature_names_out()
    actual_svd_dim = min(int(args.svd_dim), max(1, train_tfidf.shape[0] - 1), max(1, train_tfidf.shape[1] - 1))
    if actual_svd_dim <= 0:
        raise ValueError(
            "TF-IDF matrix is too small for SVD: "
            f"shape={tuple(train_tfidf.shape)}, requested_dim={args.svd_dim}"
        )
    svd = TruncatedSVD(n_components=actual_svd_dim, random_state=42)
    svd.fit(train_tfidf)

    vector_frame = build_vector_frame(sample_windows, vectorizer, svd, window_days)
    summary = build_summary(vector_frame, window_days, actual_svd_dim)
    feature_summary = build_feature_summary(vector_frame, window_days, actual_svd_dim)
    top_terms = build_top_terms(svd, feature_names, int(args.top_terms))
    examples = build_examples(vector_frame, window_days, actual_svd_dim, int(args.example_rows_per_split))

    output_name = f"{args.output_prefix}_i{image_window}_r{return_horizon}"
    news_output_root = paths.output_root / "news" / output_name
    artifacts_root = news_output_root / "artifacts"
    news_output_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    paths.tables_root.mkdir(parents=True, exist_ok=True)

    vector_path = write_large_table(vector_frame, news_output_root / "news_tfidf_svd_features")
    vectorizer_path = artifacts_root / "tfidf_vectorizer.joblib"
    svd_path = artifacts_root / "truncated_svd.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(svd, svd_path)

    manifest = build_manifest(
        args=args,
        input_path=input_path,
        vector_path=vector_path,
        vectorizer_path=vectorizer_path,
        svd_path=svd_path,
        image_window=image_window,
        return_horizon=return_horizon,
        window_days=window_days,
        sample_windows=sample_windows,
        vector_frame=vector_frame,
        fit_metadata=fit_metadata,
        train_tfidf_shape=tuple(train_tfidf.shape),
        feature_names=feature_names,
        actual_svd_dim=actual_svd_dim,
        svd=svd,
    )

    manifest_path = paths.tables_root / f"{args.output_prefix}_manifest.json"
    summary_path = paths.tables_root / f"{args.output_prefix}_summary.csv"
    feature_summary_path = paths.tables_root / f"{args.output_prefix}_feature_summary.csv"
    top_terms_path = paths.tables_root / f"{args.output_prefix}_top_terms.csv"
    examples_path = paths.tables_root / f"{args.output_prefix}_examples.csv"

    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    summary.to_csv(summary_path, index=False)
    feature_summary.to_csv(feature_summary_path, index=False)
    top_terms.to_csv(top_terms_path, index=False)
    examples.to_csv(examples_path, index=False)

    print(
        json.dumps(
            {
                "status": "ok",
                "stage": "4-N4",
                "input_path": str(input_path),
                "image_window": image_window,
                "return_horizon": return_horizon,
                "window_days": list(window_days),
                "requested_svd_dim": int(args.svd_dim),
                "actual_svd_dim": int(actual_svd_dim),
                "train_fit_documents": len(fit_documents),
                "tfidf_shape": list(train_tfidf.shape),
                "sample_rows": int(len(vector_frame)),
                "written": {
                    "vector_features": str(vector_path),
                    "vectorizer": str(vectorizer_path),
                    "svd": str(svd_path),
                    "manifest": str(manifest_path),
                    "summary": str(summary_path),
                    "feature_summary": str(feature_summary_path),
                    "top_terms": str(top_terms_path),
                    "examples": str(examples_path),
                },
            },
            indent=2,
        )
    )


def resolve_input_path(
    output_root: Path,
    *,
    input_prefix: str,
    explicit_path: Path | None,
    image_window: int,
    return_horizon: int,
) -> Path:
    """Resolve the 4-N3 sample headline-window table path."""

    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {explicit_path}")
        return explicit_path

    base = output_root / "news" / f"{input_prefix}_i{image_window}_r{return_horizon}"
    candidates = [
        base / "sample_headline_windows.parquet",
        base / "sample_headline_windows.csv.gz",
        base / "sample_headline_windows.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "4-N3 sample headline-window table is missing. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def read_table(path: Path) -> pd.DataFrame:
    """Read parquet/csv/csv.gz table."""

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".gz" or path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input table format: {path}")


def validate_sample_windows(frame: pd.DataFrame, window_days: tuple[int, ...]) -> None:
    """Validate columns required by 4-N4."""

    required = {"split", "sample_index", "image_end_date", "label_end_date"}
    for window in window_days:
        prefix = f"{window}d"
        required.update(
            {
                f"headline_text_{prefix}",
                f"news_count_{prefix}",
                f"unique_source_count_{prefix}",
                f"headline_text_hash_{prefix}",
            }
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Input headline-window table is missing columns: " + ", ".join(missing))
    if "train" not in set(frame["split"].astype(str)):
        raise ValueError("Input headline-window table has no train split rows.")


def sort_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort samples by split/date/sample index."""

    result = frame.copy()
    result["_split_order"] = result["split"].map(SPLIT_ORDER).fillna(99).astype(int)
    result["image_end_date"] = pd.to_datetime(result["image_end_date"]).dt.date.astype(str)
    result["label_end_date"] = pd.to_datetime(result["label_end_date"]).dt.date.astype(str)
    result = result.sort_values(["_split_order", "image_end_date", "sample_index"]).drop(
        columns=["_split_order"]
    )
    return result.reset_index(drop=True)


def build_fit_documents(
    frame: pd.DataFrame,
    window_days: tuple[int, ...],
) -> tuple[list[str], pd.DataFrame]:
    """Collect train-only documents used to fit TF-IDF/SVD."""

    train = frame.loc[frame["split"].astype(str).eq("train")].copy()
    documents: list[str] = []
    rows: list[dict[str, Any]] = []
    for window in window_days:
        prefix = f"{window}d"
        text_column = f"headline_text_{prefix}"
        for _, row in train.iterrows():
            text = normalize_text(row.get(text_column, ""))
            documents.append(text)
            rows.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "image_end_date": row["image_end_date"],
                    "window_days": int(window),
                    "document_chars": int(len(text)),
                    "news_count": int(row[f"news_count_{prefix}"]),
                    "headline_text_hash": row[f"headline_text_hash_{prefix}"],
                }
            )
    return documents, pd.DataFrame(rows)


def build_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    """Create a fixed TF-IDF vectorizer."""

    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(int(args.ngram_min), int(args.ngram_max)),
        min_df=int(args.min_df),
        max_df=float(args.max_df),
        max_features=int(args.max_features),
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{1,}\b",
    )


def build_vector_frame(
    frame: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    svd: TruncatedSVD,
    window_days: tuple[int, ...],
) -> pd.DataFrame:
    """Transform headline windows into sample-level SVD vectors."""

    keep_columns = [
        "split",
        "sample_index",
        "image_end_date",
        "label_end_date",
        "strict_policy",
        "same_day_excluded",
    ]
    result = frame[[column for column in keep_columns if column in frame.columns]].copy()
    for window in window_days:
        prefix = f"{window}d"
        text_column = f"headline_text_{prefix}"
        docs = [normalize_text(value) for value in frame[text_column].tolist()]
        matrix = vectorizer.transform(docs)
        vectors = svd.transform(matrix)
        news_count = frame[f"news_count_{prefix}"].astype(int).values
        source_count = frame[f"unique_source_count_{prefix}"].astype(int).values
        window_data: dict[str, Any] = {
            f"news_count_{prefix}": news_count,
            f"unique_source_count_{prefix}": source_count,
            f"news_count_log1p_{prefix}": [math.log1p(float(value)) for value in news_count],
            f"unique_source_count_log1p_{prefix}": [
                math.log1p(float(value)) for value in source_count
            ],
            f"headline_text_hash_{prefix}": frame[f"headline_text_hash_{prefix}"].astype(str).values,
            f"news_svd_norm_{prefix}": (vectors**2).sum(axis=1) ** 0.5,
        }
        for index in range(vectors.shape[1]):
            window_data[f"news_svd_{prefix}_{index:02d}"] = vectors[:, index]
        result = pd.concat([result, pd.DataFrame(window_data, index=result.index)], axis=1)
    return result.copy()


def build_summary(
    frame: pd.DataFrame,
    window_days: tuple[int, ...],
    svd_dim: int,
) -> pd.DataFrame:
    """Build split-level count/vector summary."""

    rows: list[dict[str, Any]] = []
    for split_name, split_frame in [*frame.groupby("split", sort=False), ("all", frame)]:
        row: dict[str, Any] = {
            "split": split_name,
            "num_samples": int(len(split_frame)),
            "sample_date_min": split_frame["image_end_date"].min(),
            "sample_date_max": split_frame["image_end_date"].max(),
        }
        for window in window_days:
            prefix = f"{window}d"
            count = pd.to_numeric(split_frame[f"news_count_{prefix}"], errors="coerce")
            source_count = pd.to_numeric(split_frame[f"unique_source_count_{prefix}"], errors="coerce")
            norm = pd.to_numeric(split_frame[f"news_svd_norm_{prefix}"], errors="coerce")
            row[f"coverage_rate_{prefix}"] = float((count > 0).mean())
            row[f"mean_news_count_{prefix}"] = float(count.mean())
            row[f"median_news_count_{prefix}"] = float(count.median())
            row[f"mean_unique_source_count_{prefix}"] = float(source_count.mean())
            row[f"median_unique_source_count_{prefix}"] = float(source_count.median())
            row[f"mean_svd_norm_{prefix}"] = float(norm.mean())
            row[f"median_svd_norm_{prefix}"] = float(norm.median())
            for index in range(min(3, svd_dim)):
                values = pd.to_numeric(split_frame[f"news_svd_{prefix}_{index:02d}"], errors="coerce")
                row[f"mean_news_svd_{prefix}_{index:02d}"] = float(values.mean())
                row[f"std_news_svd_{prefix}_{index:02d}"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_summary(
    frame: pd.DataFrame,
    window_days: tuple[int, ...],
    svd_dim: int,
) -> pd.DataFrame:
    """Summarize every SVD feature by split/window."""

    rows: list[dict[str, Any]] = []
    for split_name, split_frame in [*frame.groupby("split", sort=False), ("all", frame)]:
        for window in window_days:
            prefix = f"{window}d"
            for index in range(svd_dim):
                column = f"news_svd_{prefix}_{index:02d}"
                values = pd.to_numeric(split_frame[column], errors="coerce")
                rows.append(
                    {
                        "split": split_name,
                        "window_days": int(window),
                        "feature": column,
                        "mean": _safe_float(values.mean()),
                        "std": _safe_float(values.std(ddof=1)),
                        "min": _safe_float(values.min()),
                        "max": _safe_float(values.max()),
                    }
                )
    return pd.DataFrame(rows)


def build_top_terms(
    svd: TruncatedSVD,
    feature_names: Any,
    top_terms: int,
) -> pd.DataFrame:
    """Build compact component-term table for interpretability."""

    rows: list[dict[str, Any]] = []
    names = list(feature_names)
    for index, component in enumerate(svd.components_):
        ranked_positive = component.argsort()[::-1][:top_terms]
        ranked_negative = component.argsort()[:top_terms]
        rows.append(
            {
                "component_index": int(index),
                "component_feature_suffix": f"{index:02d}",
                "explained_variance_ratio": _safe_float(svd.explained_variance_ratio_[index]),
                "top_positive_terms": " | ".join(names[item] for item in ranked_positive),
                "top_positive_weights": " | ".join(f"{component[item]:.6f}" for item in ranked_positive),
                "top_negative_terms": " | ".join(names[item] for item in ranked_negative),
                "top_negative_weights": " | ".join(f"{component[item]:.6f}" for item in ranked_negative),
            }
        )
    return pd.DataFrame(rows)


def build_examples(
    frame: pd.DataFrame,
    window_days: tuple[int, ...],
    svd_dim: int,
    rows_per_split: int,
) -> pd.DataFrame:
    """Build compact example rows for reports."""

    selected: list[pd.DataFrame] = []
    for _, split_frame in frame.groupby("split", sort=False):
        selected.append(split_frame.head(rows_per_split))
        selected.append(split_frame.tail(rows_per_split))
    examples = pd.concat(selected, ignore_index=True)
    examples = examples.drop_duplicates(["split", "sample_index"]).sort_values(
        ["split", "image_end_date", "sample_index"]
    )
    keep = ["split", "sample_index", "image_end_date", "label_end_date"]
    result = examples[keep].copy()
    for window in window_days:
        prefix = f"{window}d"
        result[f"news_count_{prefix}"] = examples[f"news_count_{prefix}"].astype(int).values
        result[f"unique_source_count_{prefix}"] = examples[f"unique_source_count_{prefix}"].astype(int).values
        result[f"news_svd_norm_{prefix}"] = examples[f"news_svd_norm_{prefix}"].values
        for index in range(min(5, svd_dim)):
            result[f"news_svd_{prefix}_{index:02d}"] = examples[f"news_svd_{prefix}_{index:02d}"].values
    return result


def build_manifest(
    *,
    args: argparse.Namespace,
    input_path: Path,
    vector_path: Path,
    vectorizer_path: Path,
    svd_path: Path,
    image_window: int,
    return_horizon: int,
    window_days: tuple[int, ...],
    sample_windows: pd.DataFrame,
    vector_frame: pd.DataFrame,
    fit_metadata: pd.DataFrame,
    train_tfidf_shape: tuple[int, int],
    feature_names: Any,
    actual_svd_dim: int,
    svd: TruncatedSVD,
) -> dict[str, Any]:
    """Build JSON manifest for the TF-IDF/SVD vectorizer run."""

    return {
        "status": "ok",
        "stage": "4-N4",
        "input": {
            "sample_headline_windows": str(input_path),
            "sha256": file_sha256(input_path),
            "num_rows": int(len(sample_windows)),
        },
        "stage4_samples": {
            "image_window": int(image_window),
            "return_horizon": int(return_horizon),
            "num_samples": int(len(vector_frame)),
            "split_counts": split_counts(vector_frame),
            "sample_date_min": vector_frame["image_end_date"].min(),
            "sample_date_max": vector_frame["image_end_date"].max(),
        },
        "alignment_policy": {
            "name": "strict_t_minus_1_calendar_date",
            "same_day_news": "excluded",
            "window_days": list(window_days),
        },
        "fit_policy": {
            "text_preprocessing": "fit TF-IDF vocabulary/IDF and SVD on train split only",
            "fit_split": "train",
            "fit_documents": int(len(fit_metadata)),
            "fit_samples": int(fit_metadata["sample_index"].nunique()),
            "fit_windows": sorted(int(value) for value in fit_metadata["window_days"].unique()),
            "fit_sample_date_min": fit_metadata["image_end_date"].min(),
            "fit_sample_date_max": fit_metadata["image_end_date"].max(),
        },
        "vectorizer": {
            "class": "sklearn.feature_extraction.text.TfidfVectorizer",
            "lowercase": True,
            "strip_accents": "unicode",
            "stop_words": "english",
            "ngram_range": [int(args.ngram_min), int(args.ngram_max)],
            "min_df": int(args.min_df),
            "max_df": float(args.max_df),
            "max_features": int(args.max_features),
            "sublinear_tf": True,
            "norm": "l2",
            "token_pattern": r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{1,}\b",
            "vocabulary_size": int(len(feature_names)),
            "vocabulary_hash": stable_hash("\n".join(feature_names)),
            "tfidf_fit_shape": list(train_tfidf_shape),
        },
        "svd": {
            "class": "sklearn.decomposition.TruncatedSVD",
            "requested_components": int(args.svd_dim),
            "actual_components": int(actual_svd_dim),
            "random_state": 42,
            "explained_variance_ratio_sum": _safe_float(svd.explained_variance_ratio_.sum()),
        },
        "outputs": {
            "news_tfidf_svd_features": str(vector_path),
            "tfidf_vectorizer": str(vectorizer_path),
            "truncated_svd": str(svd_path),
        },
        "output_columns": {
            "vector_prefixes": [f"news_svd_{window}d" for window in window_days],
            "count_features": [
                *[f"news_count_{window}d" for window in window_days],
                *[f"unique_source_count_{window}d" for window in window_days],
                *[f"news_count_log1p_{window}d" for window in window_days],
                *[f"unique_source_count_log1p_{window}d" for window in window_days],
            ],
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


def normalize_text(value: Any) -> str:
    """Return text safe for vectorization."""

    if pd.isna(value):
        return ""
    return str(value)


def split_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return split counts in stable order."""

    counts = frame["split"].value_counts().to_dict()
    ordered = {}
    for split in ["train", "validation", "test"]:
        if split in counts:
            ordered[split] = int(counts[split])
    for split, count in sorted(counts.items()):
        if split not in ordered:
            ordered[str(split)] = int(count)
    return ordered


def stable_hash(text: str) -> str:
    """Return a stable SHA256 hex digest for text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_float(value: Any) -> float | None:
    """Return None for NaN-like values, otherwise float."""

    if pd.isna(value):
        return None
    return float(value)


def _jsonable(value: Any) -> Any:
    """Convert common pandas/numpy values to JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
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
