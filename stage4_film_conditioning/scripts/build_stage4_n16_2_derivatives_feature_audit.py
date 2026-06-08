"""Build the Stage 4 N16-2 derivatives feature audit.

N16-2 is a diagnostic step before running more FiLM grids. Feature selection
signals are computed on the train split only. Stage 2 error correlations are
reported as test diagnostics and are not used as training/selection evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MARKER = "N16-2 train-only derivatives feature audit"
OUTPUT_PREFIX = "stage4_n16_2_derivatives"
DEFAULT_CONTEXT_NAME = "stage4_derivatives_context_i60_ohlc_ma_vb_r20_n16_derivatives_all"
DEFAULT_STAGE2_PREDICTION_EXPERIMENT = "stage2_i60_ohlc_ma_vb_r20"


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


def _first_existing(candidates: list[Path], label: str, required: bool = True) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    if not required:
        return None
    joined = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{joined}")


def _context_dir(project_root: Path, context_name: str, seed: int) -> Path:
    return project_root / "outputs/stage4/context" / context_name / f"seed_{seed}"


def _feature_group(feature: str) -> str:
    if feature.startswith("funding_rate_"):
        return "bitmex_funding"
    if feature.startswith("bitmex_"):
        return "bitmex_activity"
    if feature.startswith("cot_"):
        return "cftc_cme_oi_positioning"
    return "other"


def _read_derivatives_context(project_root: Path, context_name: str, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = _context_dir(project_root, context_name, seed)
    context_path = root / "context_features.csv"
    scaler_path = root / "context_scaler.json"
    if not context_path.exists():
        raise FileNotFoundError(f"Missing derivatives context_features.csv: {context_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing derivatives context_scaler.json: {scaler_path}")

    frame = pd.read_csv(context_path)
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    feature_order = list(scaler.get("feature_order", []))
    normalized_columns = list(scaler.get("normalized_feature_columns", []))
    if not feature_order or not normalized_columns:
        raise ValueError(f"Scaler does not include feature order/normalized columns: {scaler_path}")

    missing = [column for column in normalized_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Context table is missing normalized columns: {missing[:10]}")
    return frame, scaler


def _read_prior_context(path: Path | None, family: str, prefixes: tuple[str, ...]) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    required = {"sample_index", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing required columns: {sorted(required - set(frame.columns))}")
    columns = [
        column
        for column in frame.columns
        if column.endswith("_normalized") and (not prefixes or column.startswith(prefixes))
    ]
    renamed = {column: f"{family}__{column}" for column in columns}
    return frame[["sample_index", "split", *columns]].rename(columns=renamed)


def _safe_corr(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    values = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3:
        return float("nan")
    if values.iloc[:, 0].nunique(dropna=True) < 2 or values.iloc[:, 1].nunique(dropna=True) < 2:
        return float("nan")
    return float(values.iloc[:, 0].corr(values.iloc[:, 1], method=method))


def _rank_auc(feature: pd.Series, label: pd.Series) -> float:
    values = pd.concat([feature, label], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3:
        return float("nan")
    x = pd.to_numeric(values.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(values.iloc[:, 1], errors="coerce")
    mask = x.notna() & y.notna()
    x = x.loc[mask]
    y = y.loc[mask]
    positives = y.eq(1)
    negatives = y.eq(0)
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = x.rank(method="average")
    sum_ranks_pos = float(ranks.loc[positives].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _read_stage2_predictions(predictions_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed_dir in sorted(predictions_root.glob("seed_*")):
        pred_path = seed_dir / "test_predictions.csv"
        if not pred_path.exists():
            continue
        seed = int(seed_dir.name.replace("seed_", ""))
        frame = pd.read_csv(pred_path)
        needed = ["sample_index", "prob_up", "pred_class", "correct"]
        missing = [column for column in needed if column not in frame.columns]
        if missing:
            raise ValueError(f"{pred_path} is missing columns: {missing}")
        part = frame[needed].copy()
        part["stage2_run_seed"] = seed
        rows.append(part)
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
    grouped["stage2_uncertainty_mean"] = 1.0 - 2.0 * (grouped["stage2_prob_up_mean"] - 0.5).abs()
    return grouped


def _build_feature_audit(
    frame: pd.DataFrame,
    feature_order: list[str],
    normalized_columns: list[str],
) -> pd.DataFrame:
    train = frame[frame["split"].eq("train")].copy()
    validation = frame[frame["split"].eq("validation")].copy()
    test = frame[frame["split"].eq("test")].copy()
    rows: list[dict[str, Any]] = []

    for feature, column in zip(feature_order, normalized_columns, strict=True):
        all_values = pd.to_numeric(frame[column], errors="coerce")
        train_values = pd.to_numeric(train[column], errors="coerce")
        validation_values = pd.to_numeric(validation[column], errors="coerce")
        test_values = pd.to_numeric(test[column], errors="coerce")
        train_auc = _rank_auc(train_values, train["label"])
        train_auc_lift = train_auc - 0.5 if pd.notna(train_auc) else float("nan")
        corr_label = _safe_corr(train_values, train["label"], method="pearson")
        spearman_label = _safe_corr(train_values, train["label"], method="spearman")
        corr_future = _safe_corr(train_values, train["future_return"], method="pearson")
        spearman_future = _safe_corr(train_values, train["future_return"], method="spearman")
        spearman_time = (
            _safe_corr(train_values, train["end_index"], method="spearman")
            if "end_index" in train.columns
            else float("nan")
        )
        corr_error = (
            _safe_corr(test_values, test["stage2_error_rate"], method="spearman")
            if "stage2_error_rate" in test.columns
            else float("nan")
        )
        corr_uncertainty = (
            _safe_corr(test_values, test["stage2_uncertainty_mean"], method="spearman")
            if "stage2_uncertainty_mean" in test.columns
            else float("nan")
        )
        score_parts = [
            abs(corr_label) if pd.notna(corr_label) else float("nan"),
            abs(spearman_label) if pd.notna(spearman_label) else float("nan"),
            abs(corr_future) if pd.notna(corr_future) else float("nan"),
            abs(spearman_future) if pd.notna(spearman_future) else float("nan"),
            2.0 * abs(train_auc_lift) if pd.notna(train_auc_lift) else float("nan"),
        ]
        finite_scores = [value for value in score_parts if np.isfinite(value)]
        rows.append(
            {
                "feature": feature,
                "normalized_column": column,
                "group": _feature_group(feature),
                "missing_rate_all": float(all_values.isna().mean()),
                "missing_rate_train": float(train_values.isna().mean()),
                "missing_rate_validation": float(validation_values.isna().mean()),
                "missing_rate_test": float(test_values.isna().mean()),
                "train_mean": float(train_values.mean(skipna=True)),
                "train_std": float(train_values.std(skipna=True)),
                "train_unique": int(train_values.nunique(dropna=True)),
                "train_corr_label": corr_label,
                "train_spearman_label": spearman_label,
                "train_corr_future_return": corr_future,
                "train_spearman_future_return": spearman_future,
                "train_spearman_end_index": spearman_time,
                "train_abs_spearman_end_index": abs(spearman_time) if pd.notna(spearman_time) else float("nan"),
                "train_univariate_auc": train_auc,
                "train_univariate_auc_lift": train_auc_lift,
                "train_abs_auc_lift": abs(train_auc_lift) if pd.notna(train_auc_lift) else float("nan"),
                "test_spearman_stage2_error_rate": corr_error,
                "test_abs_spearman_stage2_error_rate": abs(corr_error) if pd.notna(corr_error) else float("nan"),
                "test_spearman_stage2_uncertainty": corr_uncertainty,
                "test_abs_spearman_stage2_uncertainty": abs(corr_uncertainty) if pd.notna(corr_uncertainty) else float("nan"),
                "train_only_candidate_score": max(finite_scores) if finite_scores else float("nan"),
            }
        )
    audit = pd.DataFrame(rows)
    return audit.sort_values("train_only_candidate_score", ascending=False)


def _build_group_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, chunk in audit.groupby("group", dropna=False):
        ordered = chunk.sort_values("train_only_candidate_score", ascending=False)
        rows.append(
            {
                "group": group,
                "num_features": int(len(chunk)),
                "best_feature": str(ordered.iloc[0]["feature"]),
                "max_train_only_candidate_score": float(ordered["train_only_candidate_score"].max()),
                "mean_train_only_candidate_score": float(ordered["train_only_candidate_score"].mean()),
                "max_train_abs_auc_lift": float(ordered["train_abs_auc_lift"].max()),
                "max_abs_train_spearman_future_return": float(ordered["train_spearman_future_return"].abs().max()),
                "max_abs_train_spearman_end_index": float(ordered["train_abs_spearman_end_index"].max()),
                "max_test_abs_spearman_stage2_error_rate": float(ordered["test_abs_spearman_stage2_error_rate"].max()),
                "mean_missing_rate_train": float(ordered["missing_rate_train"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("max_train_only_candidate_score", ascending=False)


def _build_internal_redundancy(
    frame: pd.DataFrame,
    feature_order: list[str],
    normalized_columns: list[str],
    threshold: float,
) -> pd.DataFrame:
    train = frame[frame["split"].eq("train")]
    numeric = train[normalized_columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="pearson").abs()
    column_to_feature = dict(zip(normalized_columns, feature_order, strict=True))
    rows: list[dict[str, Any]] = []
    columns = list(corr.columns)
    for i, left_col in enumerate(columns):
        for right_col in columns[i + 1 :]:
            value = corr.loc[left_col, right_col]
            if pd.notna(value) and value >= threshold:
                left = column_to_feature[left_col]
                right = column_to_feature[right_col]
                rows.append(
                    {
                        "feature_left": left,
                        "group_left": _feature_group(left),
                        "feature_right": right,
                        "group_right": _feature_group(right),
                        "abs_corr_train": float(value),
                        "cross_group": _feature_group(left) != _feature_group(right),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_left",
                "group_left",
                "feature_right",
                "group_right",
                "abs_corr_train",
                "cross_group",
            ]
        )
    return pd.DataFrame(rows).sort_values("abs_corr_train", ascending=False)


def _build_prior_redundancy(
    frame: pd.DataFrame,
    feature_order: list[str],
    normalized_columns: list[str],
    prior_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    if not prior_frames:
        return pd.DataFrame(
            columns=[
                "feature",
                "group",
                "max_prior_abs_corr_train",
                "max_prior_feature",
                "prior_family",
            ]
        )
    matrix = frame[["sample_index", "split", *normalized_columns]].copy()
    for prior in prior_frames:
        before = len(matrix)
        matrix = matrix.merge(prior, on=["sample_index", "split"], how="left", validate="one_to_one")
        if len(matrix) != before:
            raise RuntimeError("Prior-context merge changed row count.")

    train = matrix[matrix["split"].eq("train")]
    prior_columns = [column for column in train.columns if "__" in column and column.endswith("_normalized")]
    rows: list[dict[str, Any]] = []
    for feature, column in zip(feature_order, normalized_columns, strict=True):
        values = pd.to_numeric(train[column], errors="coerce")
        best_corr = float("nan")
        best_prior = ""
        for prior_column in prior_columns:
            corr = _safe_corr(values, pd.to_numeric(train[prior_column], errors="coerce"), method="pearson")
            abs_corr = abs(corr) if pd.notna(corr) else float("nan")
            if pd.notna(abs_corr) and (pd.isna(best_corr) or abs_corr > best_corr):
                best_corr = float(abs_corr)
                best_prior = prior_column
        rows.append(
            {
                "feature": feature,
                "group": _feature_group(feature),
                "max_prior_abs_corr_train": best_corr,
                "max_prior_feature": best_prior,
                "prior_family": best_prior.split("__", 1)[0] if "__" in best_prior else "",
            }
        )
    return pd.DataFrame(rows).sort_values("max_prior_abs_corr_train", ascending=False)


def _build_selected_candidates(
    audit: pd.DataFrame,
    internal_redundancy: pd.DataFrame,
    prior_redundancy: pd.DataFrame,
    max_total: int,
    max_per_group: int,
    max_internal_abs_corr: float,
    max_prior_abs_corr: float,
) -> pd.DataFrame:
    blocked: set[tuple[str, str]] = set()
    if not internal_redundancy.empty:
        high = internal_redundancy[internal_redundancy["abs_corr_train"].ge(max_internal_abs_corr)]
        for _, row in high.iterrows():
            left = str(row["feature_left"])
            right = str(row["feature_right"])
            blocked.add((left, right))
            blocked.add((right, left))

    prior_lookup = dict(
        zip(
            prior_redundancy["feature"],
            prior_redundancy["max_prior_abs_corr_train"],
            strict=False,
        )
    )
    selected: list[pd.Series] = []
    group_counts: dict[str, int] = {}
    candidates = audit[
        audit["train_only_candidate_score"].notna()
        & audit["missing_rate_train"].le(0.05)
        & audit["train_std"].gt(1.0e-8)
    ].sort_values("train_only_candidate_score", ascending=False)
    for _, row in candidates.iterrows():
        feature = str(row["feature"])
        group = str(row["group"])
        if group_counts.get(group, 0) >= max_per_group:
            continue
        prior_corr = prior_lookup.get(feature, float("nan"))
        if pd.notna(prior_corr) and prior_corr > max_prior_abs_corr:
            continue
        if any((feature, str(existing["feature"])) in blocked for existing in selected):
            continue
        selected.append(row)
        group_counts[group] = group_counts.get(group, 0) + 1
        if len(selected) >= max_total:
            break
    if not selected:
        return pd.DataFrame(columns=["selected_rank", *audit.columns])
    out = pd.DataFrame(selected).reset_index(drop=True)
    out.insert(0, "selected_rank", np.arange(1, len(out) + 1))
    out["selection_note"] = (
        "Train-only score with missing-rate, internal-redundancy, and prior-context redundancy filters."
    )
    return out


def _stage2_diag(frame: pd.DataFrame) -> pd.DataFrame:
    test = frame[frame["split"].eq("test")]
    return pd.DataFrame(
        [
            {
                "num_test_rows": int(len(test)),
                "stage2_coverage": float(test["stage2_error_rate"].notna().mean()),
                "stage2_correct_rate_mean": float(test["stage2_correct_rate"].mean()),
                "stage2_error_rate_mean": float(test["stage2_error_rate"].mean()),
                "stage2_prob_up_mean": float(test["stage2_prob_up_mean"].mean()),
                "stage2_pred_up_rate_mean": float(test["stage2_pred_up_rate"].mean()),
                "stage2_seed_count_min": int(test["stage2_seed_count"].min()),
                "stage2_seed_count_max": int(test["stage2_seed_count"].max()),
            }
        ]
    )


def _write_report(
    path: Path,
    audit: pd.DataFrame,
    group_summary: pd.DataFrame,
    selected: pd.DataFrame,
    internal_redundancy: pd.DataFrame,
    prior_redundancy: pd.DataFrame,
    stage2_diag: pd.DataFrame,
) -> None:
    top = audit.head(10)
    lines = [
        "# 4-N16-2 Derivatives Feature Audit",
        "",
        "## Status",
        "",
        "Completed locally.",
        "",
        "Feature-ranking signals below are train-only. Stage 2 error correlations are test diagnostics only.",
        "",
        "## Stage 2 Diagnostic",
        "",
        stage2_diag.to_markdown(index=False),
        "",
        "## Group Summary",
        "",
        group_summary.to_markdown(index=False),
        "",
        "## Top Train-Only Features",
        "",
        top[
            [
                "feature",
                "group",
                "train_only_candidate_score",
                "train_univariate_auc",
                "train_spearman_future_return",
                "train_spearman_end_index",
                "test_spearman_stage2_error_rate",
            ]
        ].to_markdown(index=False),
        "",
        "## Selected Candidate Features",
        "",
        selected[
            [
                "selected_rank",
                "feature",
                "group",
                "train_only_candidate_score",
                "train_univariate_auc",
                "train_spearman_future_return",
                "train_spearman_end_index",
            ]
        ].to_markdown(index=False)
        if not selected.empty
        else "No selected candidates after filters.",
        "",
        "## Redundancy Notes",
        "",
        f"Internal high-correlation pairs: `{len(internal_redundancy)}`.",
        f"Prior-context max absolute correlation median: `{prior_redundancy['max_prior_abs_corr_train'].median():.4f}`.",
        "",
        "## Decision",
        "",
        "Use this audit to prioritize N16-3 feature-set grids. A strong candidate should have low missing rate, non-trivial train-only label/future-return signal, and not be purely redundant with previous F&G/technical context.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--context-name", default=DEFAULT_CONTEXT_NAME)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("reports/tables"))
    parser.add_argument("--stage2-predictions-root", type=Path, default=None)
    parser.add_argument("--prior-fg-context", type=Path, default=None)
    parser.add_argument("--prior-technical-context", type=Path, default=None)
    parser.add_argument("--internal-redundancy-threshold", type=float, default=0.85)
    parser.add_argument("--selection-max-internal-abs-corr", type=float, default=0.85)
    parser.add_argument("--selection-max-prior-abs-corr", type=float, default=0.85)
    parser.add_argument("--selection-max-total", type=int, default=12)
    parser.add_argument("--selection-max-per-group", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = _find_workspace_root(args.workspace_root)
    project_root = args.project_root
    if not project_root.is_absolute():
        project_root = (Path.cwd() / project_root).resolve()

    if args.stage2_predictions_root is None:
        stage2_predictions_root = _first_existing(
            [
                workspace_root
                / "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15/outputs/stage2/predictions"
                / DEFAULT_STAGE2_PREDICTION_EXPERIMENT,
                workspace_root
                / "stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8/outputs/stage2/predictions"
                / DEFAULT_STAGE2_PREDICTION_EXPERIMENT,
            ],
            "Stage 2 prediction root",
        )
    else:
        stage2_predictions_root = args.stage2_predictions_root.expanduser().resolve()

    fg_context = args.prior_fg_context or _first_existing(
        [
            project_root / "outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60_fg_only/seed_42/context_features.csv",
            project_root
            / "stage4_p7_p8_result_bundle/outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60_fg_only/seed_42/context_features.csv",
            workspace_root
            / "stage4_v2_v8_p7_p8_analysis_bundle/outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60_fg_only/seed_42/context_features.csv",
        ],
        "F&G prior context",
        required=False,
    )
    technical_context = args.prior_technical_context or _first_existing(
        [
            project_root / "outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_features.csv",
        ],
        "technical prior context",
        required=False,
    )

    frame, scaler = _read_derivatives_context(project_root, args.context_name, args.run_seed)
    feature_order = list(scaler["feature_order"])
    normalized_columns = list(scaler["normalized_feature_columns"])
    stage2 = _read_stage2_predictions(stage2_predictions_root)
    frame = frame.merge(stage2, on="sample_index", how="left", validate="one_to_one")

    audit = _build_feature_audit(frame, feature_order, normalized_columns)
    group_summary = _build_group_summary(audit)
    internal_redundancy = _build_internal_redundancy(
        frame,
        feature_order,
        normalized_columns,
        threshold=args.internal_redundancy_threshold,
    )
    prior_frames = [
        prior
        for prior in [
            _read_prior_context(fg_context, "fear_greed", ("fg_",)) if fg_context else None,
            _read_prior_context(technical_context, "technical", ("bb_", "mfi_", "rv_")) if technical_context else None,
        ]
        if prior is not None
    ]
    prior_redundancy = _build_prior_redundancy(frame, feature_order, normalized_columns, prior_frames)
    selected = _build_selected_candidates(
        audit,
        internal_redundancy,
        prior_redundancy,
        max_total=args.selection_max_total,
        max_per_group=args.selection_max_per_group,
        max_internal_abs_corr=args.selection_max_internal_abs_corr,
        max_prior_abs_corr=args.selection_max_prior_abs_corr,
    )
    stage2_diag = _stage2_diag(frame)

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "feature_audit": output_root / f"{OUTPUT_PREFIX}_feature_audit.csv",
        "group_summary": output_root / f"{OUTPUT_PREFIX}_group_summary.csv",
        "internal_redundancy_pairs": output_root / f"{OUTPUT_PREFIX}_internal_redundancy_pairs.csv",
        "prior_context_redundancy": output_root / f"{OUTPUT_PREFIX}_prior_context_redundancy.csv",
        "selected_candidates": output_root / f"{OUTPUT_PREFIX}_selected_feature_candidates.csv",
        "stage2_error_diagnostics": output_root / f"{OUTPUT_PREFIX}_stage2_error_diagnostics.csv",
        "report": output_root / f"{OUTPUT_PREFIX}_feature_audit_report.md",
        "manifest": output_root / f"{OUTPUT_PREFIX}_manifest.json",
    }
    audit.to_csv(paths["feature_audit"], index=False)
    group_summary.to_csv(paths["group_summary"], index=False)
    internal_redundancy.to_csv(paths["internal_redundancy_pairs"], index=False)
    prior_redundancy.to_csv(paths["prior_context_redundancy"], index=False)
    selected.to_csv(paths["selected_candidates"], index=False)
    stage2_diag.to_csv(paths["stage2_error_diagnostics"], index=False)
    _write_report(paths["report"], audit, group_summary, selected, internal_redundancy, prior_redundancy, stage2_diag)

    manifest = {
        "status": "ok",
        "stage": "4-N16-2",
        "marker": MARKER,
        "workspace_root": str(workspace_root),
        "project_root": str(project_root),
        "context_name": args.context_name,
        "run_seed": args.run_seed,
        "split_counts": frame["split"].value_counts().to_dict(),
        "num_derivatives_features": int(len(feature_order)),
        "stage2_predictions_root": str(stage2_predictions_root),
        "prior_contexts": {
            "fear_greed": str(fg_context) if fg_context else None,
            "technical": str(technical_context) if technical_context else None,
        },
        "selection_policy": {
            "feature_selection_split": "train only",
            "train_only_candidate_score": (
                "max(abs(pearson/spearman label), abs(pearson/spearman future_return), "
                "2*abs(univariate_auc-0.5))"
            ),
            "stage2_error_correlation": "test diagnostic only",
            "time_trend_correlation": "train end_index Spearman diagnostic only",
            "internal_redundancy_threshold": args.internal_redundancy_threshold,
            "selection_max_internal_abs_corr": args.selection_max_internal_abs_corr,
            "selection_max_prior_abs_corr": args.selection_max_prior_abs_corr,
            "selection_max_total": args.selection_max_total,
            "selection_max_per_group": args.selection_max_per_group,
        },
        "top_features": audit.head(10)[["feature", "group", "train_only_candidate_score"]].to_dict("records"),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
