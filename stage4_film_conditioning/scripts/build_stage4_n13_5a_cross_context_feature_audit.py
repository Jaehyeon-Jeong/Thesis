"""Build the Stage 4 N13-5A cross-context feature audit.

This is a diagnostic step before selected-combo FiLM. Feature selection signals
are computed on the train split only. Stage 2 error correlations are reported as
test diagnostics and are not used in the train-only candidate score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_PREFIX = "stage4_n13_5a_cross_context"
STAGE2_BUNDLE_NAME = "stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8"


@dataclass(frozen=True)
class FeatureSource:
    family: str
    path: Path
    prefixes: tuple[str, ...]


def _find_workspace_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    cwd = Path.cwd().resolve()
    if cwd.name == "stage4_film_conditioning":
        if cwd.parent.name == "Thesis":
            return cwd.parent.parent
        return cwd.parent
    for parent in [cwd, *cwd.parents]:
        if (parent / "stage4_film_conditioning").exists():
            return parent
    raise FileNotFoundError("Could not infer workspace root; pass --workspace-root.")


def _first_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    joined = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{joined}")


def _csv_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _first_existing_context_csv(candidates: list[Path], label: str, min_rows: int = 1000) -> Path:
    rejected: list[str] = []
    for path in candidates:
        if not path.exists():
            rejected.append(f"{path} [missing]")
            continue
        rows = _csv_row_count(path)
        if rows >= min_rows:
            return path
        rejected.append(f"{path} [rows={rows} < {min_rows}]")
    joined = "\n".join(rejected)
    raise FileNotFoundError(f"Could not find full-size {label}. Tried:\n{joined}")


def _default_sources(workspace_root: Path) -> dict[str, Path]:
    main_stage4 = workspace_root / "stage4_film_conditioning"
    n13_2 = workspace_root / "N13_2_results"
    n13_4 = workspace_root / "N13_4_results"
    stage2_bundle = workspace_root / STAGE2_BUNDLE_NAME
    return {
        "base": _first_existing_context_csv(
            [
                main_stage4
                / "stage4_p7_p8_result_bundle/outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60_fg_only/seed_42/context_features.csv",
                workspace_root
                / "stage4_v2_v8_p7_p8_analysis_bundle/outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60_fg_only/seed_42/context_features.csv",
                main_stage4
                / "outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60_fg_only/seed_42/context_features.csv",
            ],
            "base/F&G context_features.csv",
        ),
        "technical": _first_existing(
            [
                main_stage4
                / "outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_features.csv",
            ],
            "optional prebuilt technical context_features.csv",
        )
        if (
            main_stage4
            / "outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_features.csv"
        ).exists()
        and _csv_row_count(
            main_stage4
            / "outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_features.csv"
        )
        >= 1000
        else Path("__compute_from_btc_ohlcv__"),
        "btc_source": _first_existing(
            [
                workspace_root / "데이터셋/BTC _OHLCV/btc_1d_data_2018_to_2025.csv",
                workspace_root / "데이터셋/BTC _OHLCV/btc_1d_data_2018_to_2025.csv",
            ],
            "BTC daily OHLCV source",
        ),
        "news": _first_existing_context_csv(
            [
                main_stage4
                / "outputs/stage4/context/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60/seed_42/context_features.csv",
                workspace_root
                / "N10_results/outputs/stage4/context/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd32_w7_20_60/seed_42/context_features.csv",
            ],
            "news context_features.csv",
        ),
        "fsi": _first_existing_context_csv(
            [
                n13_2
                / "outputs/stage4/context/stage4_fsi_context_i60_ohlc_ma_vb_r20_ofr_fsi_lag1_w20_60/seed_42/context_features.csv",
                n13_2
                / "outputs/stage4/context/stage4_fsi_context_i60_ohlc_ma_vb_r20_ofr_fsi_lag1_w20_60_fsi_all/seed_42/context_features.csv",
            ],
            "FSI context_features.csv",
        ),
        "roro": _first_existing_context_csv(
            [
                n13_4
                / "outputs/stage4/context/stage4_roro_context_i60_ohlc_ma_vb_r20_public_roro_pca_lag1_w20_60/seed_42/context_features.csv",
                n13_4
                / "outputs/stage4/context/stage4_roro_context_i60_ohlc_ma_vb_r20_public_roro_pca_lag1_w20_60_roro_proxy_all/seed_42/context_features.csv",
            ],
            "RORO context_features.csv",
        ),
        "stage2_predictions_root": _first_existing(
            [
                stage2_bundle
                / "outputs/stage2/predictions/stage2_i60_ohlc_ma_vb_r20",
            ],
            "Stage 2 predictions root",
        ),
    }


def _metadata_columns() -> list[str]:
    return [
        "sample_index",
        "split",
        "Date",
        "image_end_date",
        "label_end_date",
        "start_index",
        "end_index",
        "label_end_index",
        "entry_close",
        "exit_close",
        "future_return",
        "label",
    ]


def _read_base(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "image_end_date" not in frame.columns and "Date" in frame.columns:
        frame["image_end_date"] = frame["Date"]
    columns = [column for column in _metadata_columns() if column in frame.columns]
    return frame[columns].copy()


def _feature_family(column: str) -> str:
    if column.startswith("fg_"):
        return "fear_greed"
    if column.startswith(("bb_", "mfi_", "rv_")):
        return "technical"
    if column.startswith("news_") or column.startswith("unique_source_"):
        return "news"
    if column.startswith("ofr_fsi_"):
        return "fsi"
    if column.startswith(("roro_", "riskoff_")):
        return "roro"
    return "other"


def _select_normalized_columns(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    selected: list[str] = []
    for column in frame.columns:
        if not column.endswith("_normalized"):
            continue
        if prefixes and not column.startswith(prefixes):
            continue
        selected.append(column)
    return selected


def _read_feature_source(source: FeatureSource) -> pd.DataFrame:
    frame = pd.read_csv(source.path)
    cols = ["sample_index", "split"] + _select_normalized_columns(frame, source.prefixes)
    missing = [column for column in ["sample_index", "split"] if column not in frame.columns]
    if missing:
        raise ValueError(f"{source.path} is missing required columns: {missing}")
    return frame[cols].copy()


def _compute_technical_context_from_btc(base: pd.DataFrame, btc_path: Path, context_window: int = 60) -> pd.DataFrame:
    ohlcv = pd.read_csv(btc_path).reset_index(drop=True)
    close = pd.to_numeric(ohlcv["Close"], errors="coerce")
    high = pd.to_numeric(ohlcv["High"], errors="coerce")
    low = pd.to_numeric(ohlcv["Low"], errors="coerce")
    volume = pd.to_numeric(ohlcv["Volume"], errors="coerce")

    middle = close.rolling(context_window, min_periods=context_window).mean()
    std = close.rolling(context_window, min_periods=context_window).std(ddof=0)
    upper = middle + 2.0 * std
    lower = middle - 2.0 * std
    band_width = upper - lower

    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    typical_diff = typical_price.diff()
    positive_flow = raw_money_flow.where(typical_diff > 0.0, 0.0)
    negative_flow = raw_money_flow.where(typical_diff < 0.0, 0.0)
    positive_sum = positive_flow.rolling(context_window, min_periods=context_window).sum()
    negative_sum = negative_flow.rolling(context_window, min_periods=context_window).sum()
    money_flow_ratio = positive_sum / negative_sum.replace(0.0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + money_flow_ratio)
    mfi = mfi.where(~((negative_sum == 0.0) & (positive_sum > 0.0)), 100.0)
    mfi = mfi.where(~((positive_sum == 0.0) & (negative_sum > 0.0)), 0.0)

    log_return = np.log(close / close.shift(1))
    rv_window = max(context_window - 1, 1)
    realized_volatility = log_return.rolling(rv_window, min_periods=rv_window).std(ddof=0)

    raw = pd.DataFrame(
        {
            "bb_percent_b_60": (close - lower) / band_width,
            "bb_bandwidth_60": band_width / middle,
            "mfi_60": mfi,
            "rv_60": realized_volatility,
        },
        index=ohlcv.index,
    )

    table = base[["sample_index", "split", "end_index"]].copy()
    table["end_index"] = pd.to_numeric(table["end_index"], errors="raise").astype(int)
    table = table.join(raw, on="end_index")
    _normalize_raw_technical_features(table, ["bb_percent_b_60", "bb_bandwidth_60", "mfi_60", "rv_60"])
    return table.drop(columns=["end_index"])


def _normalize_raw_technical_features(table: pd.DataFrame, features: list[str]) -> None:
    train_mask = table["split"].astype(str).eq("train")
    if not train_mask.any():
        raise ValueError("Cannot normalize technical context because train split is empty.")
    transforms = {
        "bb_percent_b_60": "identity",
        "bb_bandwidth_60": "log1p_nonnegative",
        "mfi_60": "divide_100",
        "rv_60": "log1p_nonnegative",
    }
    for feature in features:
        values = pd.to_numeric(table[feature], errors="coerce").astype(float)
        transform = transforms[feature]
        if transform == "divide_100":
            transformed = values / 100.0
        elif transform == "log1p_nonnegative":
            transformed = np.log1p(values.where(values >= 0.0))
        elif transform == "identity":
            transformed = values
        else:
            raise ValueError(f"Unsupported transform: {transform}")

        train_values = transformed.loc[train_mask].replace([np.inf, -np.inf], np.nan)
        median = float(train_values.median())
        if not np.isfinite(median):
            raise ValueError(f"Cannot fit technical median for all-missing feature: {feature}")
        imputed = transformed.replace([np.inf, -np.inf], np.nan).fillna(median)
        lower = float(imputed.loc[train_mask].quantile(0.01))
        upper = float(imputed.loc[train_mask].quantile(0.99))
        clipped = imputed.clip(lower=min(lower, upper), upper=max(lower, upper))
        mean = float(clipped.loc[train_mask].mean())
        std = float(clipped.loc[train_mask].std(ddof=0))
        if not np.isfinite(std) or std < 1.0e-8:
            std = 1.0
        table[f"{feature}_normalized"] = (clipped - mean) / std


def _merge_sources(base: pd.DataFrame, sources: list[FeatureSource]) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = base.copy()
    source_info: dict[str, Any] = {}
    for source in sources:
        frame = _read_feature_source(source)
        feature_columns = [column for column in frame.columns if column not in {"sample_index", "split"}]
        rename_map = {column: f"{source.family}__{column}" for column in feature_columns}
        frame = frame.rename(columns=rename_map)
        before_rows = len(matrix)
        matrix = matrix.merge(frame, on=["sample_index", "split"], how="left", validate="one_to_one")
        if len(matrix) != before_rows:
            raise RuntimeError(f"Merge changed row count for {source.family}")
        source_info[source.family] = {
            "path": str(source.path),
            "num_features": len(feature_columns),
            "features": [rename_map[column] for column in feature_columns],
        }
    return matrix, source_info


def _merge_precomputed_feature_frame(
    matrix: pd.DataFrame,
    family: str,
    frame: pd.DataFrame,
    source_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_columns = [
        column
        for column in frame.columns
        if column not in {"sample_index", "split"} and column.endswith("_normalized")
    ]
    rename_map = {column: f"{family}__{column}" for column in feature_columns}
    before_rows = len(matrix)
    merged = matrix.merge(
        frame[["sample_index", "split", *feature_columns]].rename(columns=rename_map),
        on=["sample_index", "split"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != before_rows:
        raise RuntimeError(f"Merge changed row count for {family}")
    return merged, {
        "path": source_label,
        "num_features": len(feature_columns),
        "features": [rename_map[column] for column in feature_columns],
    }


def _build_stage2_diagnostic(matrix: pd.DataFrame) -> pd.DataFrame:
    test = matrix[matrix["split"].eq("test")].copy()
    return pd.DataFrame(
        [
            {
                "num_test_rows": int(len(test)),
                "stage2_coverage": float(test["stage2_prob_up_mean"].notna().mean()),
                "stage2_correct_rate_mean": float(test["stage2_correct_rate"].mean()),
                "stage2_error_rate_mean": float(test["stage2_error_rate"].mean()),
                "stage2_prob_up_mean": float(test["stage2_prob_up_mean"].mean()),
                "stage2_pred_up_rate_mean": float(test["stage2_pred_up_rate"].mean()),
            }
        ]
    )


def _read_stage2_predictions(predictions_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed_dir in sorted(predictions_root.glob("seed_*")):
        seed_text = seed_dir.name.replace("seed_", "")
        pred_path = seed_dir / "test_predictions.csv"
        if not pred_path.exists():
            continue
        frame = pd.read_csv(pred_path)
        frame["stage2_run_seed"] = int(seed_text)
        rows.append(
            frame[
                [
                    "sample_index",
                    "prob_up",
                    "pred_class",
                    "correct",
                    "stage2_run_seed",
                ]
            ].copy()
        )
    if not rows:
        raise FileNotFoundError(f"No Stage 2 test_predictions.csv files under {predictions_root}")
    preds = pd.concat(rows, ignore_index=True)
    grouped = preds.groupby("sample_index", as_index=False).agg(
        stage2_prob_up_mean=("prob_up", "mean"),
        stage2_prob_up_std=("prob_up", "std"),
        stage2_pred_up_rate=("pred_class", "mean"),
        stage2_correct_rate=("correct", "mean"),
        stage2_seed_count=("stage2_run_seed", "nunique"),
    )
    grouped["stage2_error_rate"] = 1.0 - grouped["stage2_correct_rate"]
    grouped["stage2_margin_abs_mean"] = (grouped["stage2_prob_up_mean"] - 0.5).abs()
    return grouped


def _safe_corr(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    values = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3:
        return float("nan")
    if values.iloc[:, 0].nunique(dropna=True) < 2 or values.iloc[:, 1].nunique(dropna=True) < 2:
        return float("nan")
    return float(values.iloc[:, 0].corr(values.iloc[:, 1], method=method))


def _build_feature_audit(matrix: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    train = matrix[matrix["split"].eq("train")].copy()
    test = matrix[matrix["split"].eq("test")].copy()
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        family = feature.split("__", 1)[0]
        train_feature = pd.to_numeric(train[feature], errors="coerce")
        test_feature = pd.to_numeric(test[feature], errors="coerce")
        corr_label = _safe_corr(train_feature, train["label"], method="pearson")
        corr_future = _safe_corr(train_feature, train["future_return"], method="pearson")
        corr_future_spearman = _safe_corr(train_feature, train["future_return"], method="spearman")
        corr_stage2_error = _safe_corr(test_feature, test["stage2_error_rate"], method="pearson") if "stage2_error_rate" in test else float("nan")
        corr_stage2_prob = _safe_corr(test_feature, test["stage2_prob_up_mean"], method="pearson") if "stage2_prob_up_mean" in test else float("nan")
        corr_stage2_uncertainty = _safe_corr(test_feature, 1.0 - test["stage2_margin_abs_mean"] * 2.0, method="pearson") if "stage2_margin_abs_mean" in test else float("nan")
        train_only_score = np.nanmax([abs(corr_label), abs(corr_future), abs(corr_future_spearman)])
        rows.append(
            {
                "feature": feature,
                "family": family,
                "source_feature": feature.split("__", 1)[1] if "__" in feature else feature,
                "num_rows_all": int(matrix[feature].shape[0]),
                "missing_rate_all": float(pd.to_numeric(matrix[feature], errors="coerce").isna().mean()),
                "missing_rate_train": float(train_feature.isna().mean()),
                "train_mean": float(train_feature.mean(skipna=True)),
                "train_std": float(train_feature.std(skipna=True)),
                "train_unique": int(train_feature.nunique(dropna=True)),
                "train_corr_label": corr_label,
                "train_abs_corr_label": abs(corr_label) if pd.notna(corr_label) else float("nan"),
                "train_corr_future_return": corr_future,
                "train_abs_corr_future_return": abs(corr_future) if pd.notna(corr_future) else float("nan"),
                "train_spearman_future_return": corr_future_spearman,
                "train_abs_spearman_future_return": abs(corr_future_spearman) if pd.notna(corr_future_spearman) else float("nan"),
                "test_corr_stage2_error_rate": corr_stage2_error,
                "test_abs_corr_stage2_error_rate": abs(corr_stage2_error) if pd.notna(corr_stage2_error) else float("nan"),
                "test_corr_stage2_prob_up_mean": corr_stage2_prob,
                "test_abs_corr_stage2_prob_up_mean": abs(corr_stage2_prob) if pd.notna(corr_stage2_prob) else float("nan"),
                "test_corr_stage2_uncertainty": corr_stage2_uncertainty,
                "test_abs_corr_stage2_uncertainty": abs(corr_stage2_uncertainty) if pd.notna(corr_stage2_uncertainty) else float("nan"),
                "train_only_candidate_score": float(train_only_score) if pd.notna(train_only_score) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("train_only_candidate_score", ascending=False)


def _build_family_summary(feature_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in feature_audit.groupby("family"):
        rows.append(
            {
                "family": family,
                "num_features": int(len(group)),
                "max_train_only_candidate_score": float(group["train_only_candidate_score"].max()),
                "mean_train_only_candidate_score": float(group["train_only_candidate_score"].mean()),
                "best_train_feature": str(group.sort_values("train_only_candidate_score", ascending=False).iloc[0]["feature"]),
                "max_train_abs_corr_label": float(group["train_abs_corr_label"].max()),
                "max_train_abs_corr_future_return": float(group["train_abs_corr_future_return"].max()),
                "max_test_abs_corr_stage2_error_rate": float(group["test_abs_corr_stage2_error_rate"].max()),
                "max_test_abs_corr_stage2_prob_up_mean": float(group["test_abs_corr_stage2_prob_up_mean"].max()),
                "mean_missing_rate_train": float(group["missing_rate_train"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("max_train_only_candidate_score", ascending=False)


def _build_redundancy_pairs(matrix: pd.DataFrame, feature_columns: list[str], threshold: float) -> pd.DataFrame:
    train = matrix[matrix["split"].eq("train")]
    numeric = train[feature_columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="pearson").abs()
    rows: list[dict[str, Any]] = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= threshold:
                rows.append(
                    {
                        "feature_left": left,
                        "family_left": left.split("__", 1)[0],
                        "feature_right": right,
                        "family_right": right.split("__", 1)[0],
                        "abs_corr_train": float(value),
                        "cross_family": left.split("__", 1)[0] != right.split("__", 1)[0],
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_corr_train", ascending=False)


def _select_candidates(
    feature_audit: pd.DataFrame,
    redundancy_pairs: pd.DataFrame,
    per_family: int,
    max_total: int,
    max_abs_corr: float,
) -> pd.DataFrame:
    blocked_pairs: set[tuple[str, str]] = set()
    if not redundancy_pairs.empty:
        for _, row in redundancy_pairs.iterrows():
            if row["abs_corr_train"] >= max_abs_corr:
                a, b = str(row["feature_left"]), str(row["feature_right"])
                blocked_pairs.add((a, b))
                blocked_pairs.add((b, a))

    selected: list[pd.Series] = []
    family_counts: dict[str, int] = {}
    candidates = feature_audit.copy()
    candidates = candidates[
        candidates["missing_rate_train"].le(0.02)
        & candidates["train_std"].gt(1e-8)
        & candidates["train_only_candidate_score"].notna()
    ].sort_values("train_only_candidate_score", ascending=False)

    for _, row in candidates.iterrows():
        feature = str(row["feature"])
        family = str(row["family"])
        if family_counts.get(family, 0) >= per_family:
            continue
        if any((feature, str(existing["feature"])) in blocked_pairs for existing in selected):
            continue
        selected.append(row)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= max_total:
            break

    if not selected:
        return pd.DataFrame(columns=list(feature_audit.columns) + ["selected_rank", "selection_note"])
    selected_df = pd.DataFrame(selected).reset_index(drop=True)
    selected_df.insert(0, "selected_rank", np.arange(1, len(selected_df) + 1))
    selected_df["selection_note"] = (
        "Selected by train-only label/future-return score, with within/cross-feature redundancy cap."
    )
    return selected_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("reports/tables"))
    parser.add_argument("--redundancy-threshold", type=float, default=0.85)
    parser.add_argument("--selection-max-abs-corr", type=float, default=0.85)
    parser.add_argument("--selection-per-family", type=int, default=3)
    parser.add_argument("--selection-max-total", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = _find_workspace_root(args.workspace_root)
    source_paths = _default_sources(workspace_root)

    base = _read_base(source_paths["base"])
    sources = [
        FeatureSource("fear_greed", source_paths["base"], ("fg_",)),
        FeatureSource("news", source_paths["news"], ("news_", "unique_source_")),
        FeatureSource("fsi", source_paths["fsi"], ("ofr_fsi_",)),
        FeatureSource("roro", source_paths["roro"], ("roro_", "riskoff_")),
    ]
    matrix, source_info = _merge_sources(base, sources)
    if str(source_paths["technical"]) == "__compute_from_btc_ohlcv__":
        technical = _compute_technical_context_from_btc(base, source_paths["btc_source"], context_window=60)
        matrix, technical_info = _merge_precomputed_feature_frame(
            matrix,
            "technical",
            technical,
            source_label=f"computed_from:{source_paths['btc_source']}",
        )
    else:
        technical = _read_feature_source(FeatureSource("technical", source_paths["technical"], ("bb_", "mfi_", "rv_")))
        matrix, technical_info = _merge_precomputed_feature_frame(
            matrix,
            "technical",
            technical,
            source_label=str(source_paths["technical"]),
        )
    source_info["technical"] = technical_info

    stage2 = _read_stage2_predictions(source_paths["stage2_predictions_root"])
    matrix = matrix.merge(stage2, on="sample_index", how="left", validate="one_to_one")

    feature_columns = [
        column
        for column in matrix.columns
        if "__" in column and column.endswith("_normalized")
    ]
    feature_audit = _build_feature_audit(matrix, feature_columns)
    family_summary = _build_family_summary(feature_audit)
    redundancy_pairs = _build_redundancy_pairs(matrix, feature_columns, args.redundancy_threshold)
    selected = _select_candidates(
        feature_audit,
        redundancy_pairs,
        per_family=args.selection_per_family,
        max_total=args.selection_max_total,
        max_abs_corr=args.selection_max_abs_corr,
    )

    stage2_diag = _build_stage2_diagnostic(matrix)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "feature_audit": output_root / f"{OUTPUT_PREFIX}_feature_audit.csv",
        "family_summary": output_root / f"{OUTPUT_PREFIX}_family_summary.csv",
        "redundancy_pairs": output_root / f"{OUTPUT_PREFIX}_redundancy_pairs.csv",
        "selected_candidates": output_root / f"{OUTPUT_PREFIX}_selected_feature_candidates.csv",
        "stage2_error_diagnostics": output_root / f"{OUTPUT_PREFIX}_stage2_error_diagnostics.csv",
        "manifest": output_root / f"{OUTPUT_PREFIX}_manifest.json",
    }
    feature_audit.to_csv(paths["feature_audit"], index=False)
    family_summary.to_csv(paths["family_summary"], index=False)
    redundancy_pairs.to_csv(paths["redundancy_pairs"], index=False)
    selected.to_csv(paths["selected_candidates"], index=False)
    stage2_diag.to_csv(paths["stage2_error_diagnostics"], index=False)

    manifest = {
        "status": "ok",
        "stage": "4-N13-5A",
        "workspace_root": str(workspace_root),
        "num_rows": int(len(matrix)),
        "split_counts": matrix["split"].value_counts().to_dict(),
        "num_features": int(len(feature_columns)),
        "source_info": source_info,
        "btc_source": str(source_paths["btc_source"]),
        "stage2_predictions_root": str(source_paths["stage2_predictions_root"]),
        "selection_policy": {
            "feature_selection_split": "train only",
            "train_only_candidate_score": "max(abs(corr(label)), abs(corr(future_return)), abs(spearman(future_return)))",
            "stage2_error_correlation": "test diagnostic only, not used for selected candidates",
            "redundancy_threshold": args.redundancy_threshold,
            "selection_max_abs_corr": args.selection_max_abs_corr,
            "selection_per_family": args.selection_per_family,
            "selection_max_total": args.selection_max_total,
        },
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
