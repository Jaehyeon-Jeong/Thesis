#!/usr/bin/env python
"""Analyze Stage 4 P7/P8 seed-collapse and validation threshold calibration.

This diagnostic does not train a model. It reads existing Stage 4 prediction
CSVs, compares default threshold behavior, calibrates a threshold on validation,
and applies that threshold to test predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import (
    get_context_config,
    make_stage4_experiment_name,
    validate_context_method,
)


DEFAULT_METHODS = ("film_full", "film_full_bounded_last_block")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_SPLITS = ("validation", "test")
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_kaggle.yaml")
    parser.add_argument("--image-window", type=int, default=60)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--context-window", type=int, default=60)
    parser.add_argument("--context-suffix", default="")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--run-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--calibration-metric",
        default="balanced_accuracy",
        choices=["balanced_accuracy", "f1", "accuracy", "youden_j"],
    )
    parser.add_argument(
        "--collapse-low",
        type=float,
        default=0.15,
        help="Predicted positive rate below this value is a Down-collapse warning.",
    )
    parser.add_argument(
        "--collapse-high",
        type=float,
        default=0.85,
        help="Predicted positive rate above this value is an Up-collapse warning.",
    )
    parser.add_argument(
        "--output-prefix",
        default="stage4_v2_v8_p7_p8_seed_collapse",
    )
    return parser.parse_args()


def main() -> None:
    """Run the seed-collapse diagnostic."""

    args = parse_args()
    config = load_config(args.config)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    context_config = get_context_config(config)
    context_suffix = str(
        args.context_suffix or context_config.get("feature_set_name", "")
    ).strip()
    methods = [validate_context_method(method) for method in args.methods]
    splits = [str(split) for split in args.splits]

    prediction_frames: dict[tuple[str, int, str], pd.DataFrame] = {}
    missing_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []

    for method in methods:
        experiment_name = make_stage4_experiment_name(
            args.image_window,
            args.image_spec,
            args.return_horizon,
            method,
            args.context_window,
            experiment_suffix=context_suffix,
        )
        for run_seed in args.run_seeds:
            for split in splits:
                prediction_path = (
                    paths.predictions_root
                    / experiment_name
                    / f"seed_{int(run_seed)}"
                    / f"{split}_predictions.csv"
                )
                if not prediction_path.exists():
                    missing_rows.append(
                        {
                            "context_method": method,
                            "experiment_name": experiment_name,
                            "run_seed": int(run_seed),
                            "split": split,
                            "prediction_path": str(prediction_path),
                            "reason": "missing_prediction_csv",
                        }
                    )
                    continue
                frame = pd.read_csv(prediction_path)
                prediction_frames[(method, int(run_seed), split)] = frame
                default_pred = _prediction_from_threshold(frame["prob_up"], 0.5)
                metrics = _metrics_from_predictions(
                    frame["label"],
                    frame["prob_up"],
                    default_pred,
                )
                metric_rows.append(
                    {
                        "context_method": method,
                        "experiment_name": experiment_name,
                        "run_seed": int(run_seed),
                        "split": split,
                        "threshold": 0.5,
                        "threshold_source": "default_0_5",
                        **_collapse_fields(
                            metrics["predicted_positive_rate"],
                            args.collapse_low,
                            args.collapse_high,
                        ),
                        **metrics,
                    }
                )
                quantile_rows.append(
                    {
                        "context_method": method,
                        "experiment_name": experiment_name,
                        "run_seed": int(run_seed),
                        "split": split,
                        **_probability_quantiles(frame["prob_up"]),
                    }
                )

    calibration_rows = _build_calibration_rows(
        prediction_frames,
        methods=methods,
        run_seeds=args.run_seeds,
        metric_name=args.calibration_metric,
        collapse_low=args.collapse_low,
        collapse_high=args.collapse_high,
    )
    pairwise_rows = _build_pairwise_rows(
        prediction_frames,
        methods=methods,
        run_seeds=args.run_seeds,
        splits=splits,
    )

    tables_root = paths.tables_root
    tables_root.mkdir(parents=True, exist_ok=True)
    metric_df = pd.DataFrame(metric_rows)
    quantile_df = pd.DataFrame(quantile_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    missing_df = pd.DataFrame(missing_rows)

    written = {
        "default_metrics": tables_root / f"{args.output_prefix}_default_metrics.csv",
        "probability_quantiles": tables_root / f"{args.output_prefix}_probability_quantiles.csv",
        "threshold_calibration": tables_root / f"{args.output_prefix}_threshold_calibration.csv",
        "pairwise_comparison": tables_root / f"{args.output_prefix}_pairwise_comparison.csv",
        "missing_predictions": tables_root / f"{args.output_prefix}_missing_predictions.csv",
        "markdown_report": tables_root / f"{args.output_prefix}_report.md",
    }
    metric_df.to_csv(written["default_metrics"], index=False)
    quantile_df.to_csv(written["probability_quantiles"], index=False)
    calibration_df.to_csv(written["threshold_calibration"], index=False)
    pairwise_df.to_csv(written["pairwise_comparison"], index=False)
    missing_df.to_csv(written["missing_predictions"], index=False)
    written["markdown_report"].write_text(
        _build_markdown_report(
            metric_df=metric_df,
            calibration_df=calibration_df,
            pairwise_df=pairwise_df,
            missing_df=missing_df,
            args=args,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "methods": methods,
                "run_seeds": [int(seed) for seed in args.run_seeds],
                "splits": splits,
                "num_default_metric_rows": int(len(metric_df)),
                "num_calibration_rows": int(len(calibration_df)),
                "num_pairwise_rows": int(len(pairwise_df)),
                "num_missing_predictions": int(len(missing_df)),
                "written": {key: str(value) for key, value in written.items()},
            },
            indent=2,
        )
    )


def _prediction_from_threshold(prob_up: Iterable[float], threshold: float) -> np.ndarray:
    """Return class predictions from `prob_up >= threshold`."""

    values = np.asarray(list(prob_up), dtype=float)
    return (values >= float(threshold)).astype(int)


def _metrics_from_predictions(
    labels: Iterable[int],
    prob_up: Iterable[float],
    predictions: Iterable[int],
) -> dict[str, float | int]:
    """Compute binary classification metrics without relying on sklearn."""

    y_true = np.asarray(list(labels), dtype=int)
    y_score = np.asarray(list(prob_up), dtype=float)
    y_pred = np.asarray(list(predictions), dtype=int)
    if len(y_true) != len(y_score) or len(y_true) != len(y_pred):
        raise ValueError("labels, prob_up, and predictions must have matching lengths.")
    if len(y_true) == 0:
        return _empty_metrics()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    positive_count = int((y_true == 1).sum())
    negative_count = int((y_true == 0).sum())
    accuracy = (tp + tn) / len(y_true)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2
    brier = float(np.mean((y_score - y_true) ** 2))
    roc_auc = _roc_auc(y_true, y_score)
    average_precision = _average_precision(y_true, y_score)
    pred_pos_rate = float(y_pred.mean())
    pos_rate = float(y_true.mean())

    return {
        "num_samples": int(len(y_true)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": pos_rate,
        "predicted_positive_rate": pred_pos_rate,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "brier_score": brier,
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "youden_j": float(recall + specificity - 1.0),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _empty_metrics() -> dict[str, float | int]:
    """Return an empty metric row shape."""

    return {
        "num_samples": 0,
        "positive_count": 0,
        "negative_count": 0,
        "positive_rate": math.nan,
        "predicted_positive_rate": math.nan,
        "accuracy": math.nan,
        "balanced_accuracy": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "specificity": math.nan,
        "f1": math.nan,
        "brier_score": math.nan,
        "roc_auc": math.nan,
        "average_precision": math.nan,
        "youden_j": math.nan,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    """Return a safe floating division."""

    return float(numerator / denominator) if denominator else 0.0


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC using average ranks for ties."""

    positive = y_true == 1
    negative = y_true == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[positive].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average precision from the precision-recall ranking."""

    positives = int((y_true == 1).sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    cumulative_tp = np.cumsum(sorted_true == 1)
    ranks = np.arange(1, len(sorted_true) + 1)
    precision_at_k = cumulative_tp / ranks
    return float(precision_at_k[sorted_true == 1].sum() / positives)


def _probability_quantiles(prob_up: Iterable[float]) -> dict[str, float]:
    """Return compact probability distribution summary."""

    values = np.asarray(list(prob_up), dtype=float)
    row: dict[str, float] = {
        "prob_up_mean": float(np.mean(values)) if len(values) else math.nan,
        "prob_up_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "prob_up_min": float(np.min(values)) if len(values) else math.nan,
        "prob_up_max": float(np.max(values)) if len(values) else math.nan,
    }
    if len(values):
        quantiles = np.quantile(values, QUANTILES)
        for quantile, value in zip(QUANTILES, quantiles, strict=True):
            row[f"prob_up_q{int(round(quantile * 100)):02d}"] = float(value)
    return row


def _collapse_fields(
    predicted_positive_rate: float,
    low: float,
    high: float,
) -> dict[str, Any]:
    """Return collapse flags from predicted positive rate."""

    if math.isnan(float(predicted_positive_rate)):
        direction = "unknown"
        flag = False
    elif predicted_positive_rate <= low:
        direction = "mostly_down"
        flag = True
    elif predicted_positive_rate >= high:
        direction = "mostly_up"
        flag = True
    else:
        direction = "none"
        flag = False
    return {
        "collapse_flag": bool(flag),
        "collapse_direction": direction,
    }


def _build_calibration_rows(
    prediction_frames: dict[tuple[str, int, str], pd.DataFrame],
    methods: list[str],
    run_seeds: list[int],
    metric_name: str,
    collapse_low: float,
    collapse_high: float,
) -> list[dict[str, Any]]:
    """Calibrate thresholds on validation and apply them to test."""

    rows: list[dict[str, Any]] = []
    for method in methods:
        for run_seed in run_seeds:
            validation = prediction_frames.get((method, int(run_seed), "validation"))
            test = prediction_frames.get((method, int(run_seed), "test"))
            if validation is None or test is None:
                continue
            threshold_row = _select_validation_threshold(validation, metric_name)
            threshold = float(threshold_row["threshold"])
            val_pred = _prediction_from_threshold(validation["prob_up"], threshold)
            test_pred = _prediction_from_threshold(test["prob_up"], threshold)
            val_metrics = _metrics_from_predictions(
                validation["label"],
                validation["prob_up"],
                val_pred,
            )
            test_metrics = _metrics_from_predictions(
                test["label"],
                test["prob_up"],
                test_pred,
            )
            default_test_metrics = _metrics_from_predictions(
                test["label"],
                test["prob_up"],
                _prediction_from_threshold(test["prob_up"], 0.5),
            )
            rows.append(
                {
                    "context_method": method,
                    "run_seed": int(run_seed),
                    "calibration_metric": metric_name,
                    "threshold": threshold,
                    "validation_threshold_score": float(threshold_row["score"]),
                    "validation_predicted_positive_rate": val_metrics[
                        "predicted_positive_rate"
                    ],
                    "test_default_accuracy": default_test_metrics["accuracy"],
                    "test_calibrated_accuracy": test_metrics["accuracy"],
                    "test_accuracy_delta": (
                        test_metrics["accuracy"] - default_test_metrics["accuracy"]
                    ),
                    "test_default_balanced_accuracy": default_test_metrics[
                        "balanced_accuracy"
                    ],
                    "test_calibrated_balanced_accuracy": test_metrics[
                        "balanced_accuracy"
                    ],
                    "test_balanced_accuracy_delta": (
                        test_metrics["balanced_accuracy"]
                        - default_test_metrics["balanced_accuracy"]
                    ),
                    "test_default_f1": default_test_metrics["f1"],
                    "test_calibrated_f1": test_metrics["f1"],
                    "test_f1_delta": test_metrics["f1"] - default_test_metrics["f1"],
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_default_predicted_positive_rate": default_test_metrics[
                        "predicted_positive_rate"
                    ],
                    "test_calibrated_predicted_positive_rate": test_metrics[
                        "predicted_positive_rate"
                    ],
                    **{
                        f"test_calibrated_{key}": value
                        for key, value in _collapse_fields(
                            float(test_metrics["predicted_positive_rate"]),
                            collapse_low,
                            collapse_high,
                        ).items()
                    },
                    "test_calibrated_tp": test_metrics["tp"],
                    "test_calibrated_tn": test_metrics["tn"],
                    "test_calibrated_fp": test_metrics["fp"],
                    "test_calibrated_fn": test_metrics["fn"],
                }
            )
    return rows


def _select_validation_threshold(
    validation: pd.DataFrame,
    metric_name: str,
) -> dict[str, float]:
    """Select a threshold on validation for a requested metric."""

    y_true = validation["label"].to_numpy(dtype=int)
    prob_up = validation["prob_up"].to_numpy(dtype=float)
    unique = np.unique(prob_up[np.isfinite(prob_up)])
    if len(unique) <= 1:
        candidates = np.array([0.5])
    else:
        midpoints = (unique[:-1] + unique[1:]) / 2
        candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], midpoints)))

    positive_rate = float(np.mean(y_true))
    rows = []
    for threshold in candidates:
        pred = _prediction_from_threshold(prob_up, float(threshold))
        metrics = _metrics_from_predictions(y_true, prob_up, pred)
        score = (
            metrics["youden_j"]
            if metric_name == "youden_j"
            else metrics[metric_name]
        )
        rows.append(
            {
                "threshold": float(threshold),
                "score": float(score),
                "predicted_positive_rate": float(metrics["predicted_positive_rate"]),
                "positive_rate_gap": abs(
                    float(metrics["predicted_positive_rate"]) - positive_rate
                ),
                "threshold_gap_from_default": abs(float(threshold) - 0.5),
            }
        )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["score", "positive_rate_gap", "threshold_gap_from_default"],
        ascending=[False, True, True],
    )
    return {
        "threshold": float(frame.iloc[0]["threshold"]),
        "score": float(frame.iloc[0]["score"]),
    }


def _build_pairwise_rows(
    prediction_frames: dict[tuple[str, int, str], pd.DataFrame],
    methods: list[str],
    run_seeds: list[int],
    splits: list[str],
) -> list[dict[str, Any]]:
    """Build pairwise comparison rows for the first two methods."""

    if len(methods) < 2:
        return []
    left_method, right_method = methods[0], methods[1]
    rows: list[dict[str, Any]] = []
    for run_seed in run_seeds:
        for split in splits:
            left = prediction_frames.get((left_method, int(run_seed), split))
            right = prediction_frames.get((right_method, int(run_seed), split))
            if left is None or right is None:
                continue
            key = _merge_key(left, right)
            merged = left.merge(
                right,
                on=key,
                suffixes=("_left", "_right"),
            )
            if merged.empty:
                continue
            label_col = "label_left" if "label_left" in merged else "label"
            labels = merged[label_col].to_numpy(dtype=int)
            left_prob = merged["prob_up_left"].to_numpy(dtype=float)
            right_prob = merged["prob_up_right"].to_numpy(dtype=float)
            left_pred = _prediction_from_threshold(left_prob, 0.5)
            right_pred = _prediction_from_threshold(right_prob, 0.5)
            left_metrics = _metrics_from_predictions(labels, left_prob, left_pred)
            right_metrics = _metrics_from_predictions(labels, right_prob, right_pred)
            p7_up_p8_down = int(((left_pred == 1) & (right_pred == 0)).sum())
            p7_down_p8_up = int(((left_pred == 0) & (right_pred == 1)).sum())
            same_up = int(((left_pred == 1) & (right_pred == 1)).sum())
            same_down = int(((left_pred == 0) & (right_pred == 0)).sum())
            rows.append(
                {
                    "left_method": left_method,
                    "right_method": right_method,
                    "run_seed": int(run_seed),
                    "split": split,
                    "num_samples": int(len(merged)),
                    "merge_key": ",".join(key),
                    "prediction_agreement_rate": float(np.mean(left_pred == right_pred)),
                    "prob_up_correlation": _correlation(left_prob, right_prob),
                    "right_minus_left_prob_up_mean": float(np.mean(right_prob - left_prob)),
                    "right_minus_left_prob_up_std": float(
                        np.std(right_prob - left_prob, ddof=1)
                    ),
                    "left_accuracy": left_metrics["accuracy"],
                    "right_accuracy": right_metrics["accuracy"],
                    "right_minus_left_accuracy": (
                        right_metrics["accuracy"] - left_metrics["accuracy"]
                    ),
                    "left_roc_auc": left_metrics["roc_auc"],
                    "right_roc_auc": right_metrics["roc_auc"],
                    "right_minus_left_roc_auc": (
                        right_metrics["roc_auc"] - left_metrics["roc_auc"]
                    ),
                    "left_predicted_positive_rate": left_metrics[
                        "predicted_positive_rate"
                    ],
                    "right_predicted_positive_rate": right_metrics[
                        "predicted_positive_rate"
                    ],
                    "p7_up_p8_down_count": p7_up_p8_down,
                    "p7_down_p8_up_count": p7_down_p8_up,
                    "same_up_count": same_up,
                    "same_down_count": same_down,
                    "p7_up_p8_down_label_positive_rate": _group_positive_rate(
                        labels,
                        (left_pred == 1) & (right_pred == 0),
                    ),
                    "p7_down_p8_up_label_positive_rate": _group_positive_rate(
                        labels,
                        (left_pred == 0) & (right_pred == 1),
                    ),
                }
            )
    return rows


def _merge_key(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Choose stable prediction merge keys."""

    for candidates in (
        ["sample_index", "Date"],
        ["sample_index"],
        ["Date", "label_end_date"],
        ["Date"],
    ):
        if all(column in left.columns and column in right.columns for column in candidates):
            return candidates
    raise KeyError("Could not find a stable key to merge prediction frames.")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation or NaN for degenerate input."""

    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _group_positive_rate(labels: np.ndarray, mask: np.ndarray) -> float:
    """Return label positive rate for a boolean group."""

    if int(mask.sum()) == 0:
        return math.nan
    return float(np.mean(labels[mask]))


def _build_markdown_report(
    *,
    metric_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    """Build a compact Markdown report."""

    lines = [
        "# 4-V8 P7/P8 Seed-Collapse Diagnostic",
        "",
        "## Purpose",
        "",
        "- Compare P7 `film_full` and P8 `film_full_bounded_last_block` without retraining.",
        "- Check whether seed collapse is caused by threshold/class calibration rather than loss of ranking signal.",
        "- Calibrate a threshold on validation and apply it to test.",
        "",
        "## Settings",
        "",
        f"- Image: `I{args.image_window}/R{args.return_horizon}/{args.image_spec}`",
        f"- Context window: `{args.context_window}`",
        f"- Methods: `{', '.join(args.methods)}`",
        f"- Seeds: `{', '.join(str(seed) for seed in args.run_seeds)}`",
        f"- Calibration metric: `{args.calibration_metric}`",
        "",
    ]
    if not missing_df.empty:
        lines.extend(
            [
                "## Missing Predictions",
                "",
                _frame_to_markdown(missing_df),
                "",
            ]
        )
    if not metric_df.empty:
        compact_cols = [
            "context_method",
            "run_seed",
            "split",
            "collapse_flag",
            "collapse_direction",
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "f1",
            "predicted_positive_rate",
            "tp",
            "tn",
            "fp",
            "fn",
        ]
        compact_cols = [column for column in compact_cols if column in metric_df.columns]
        lines.extend(
            [
                "## Default Threshold Summary",
                "",
                _frame_to_markdown(
                    metric_df[compact_cols].sort_values(
                        ["split", "run_seed", "context_method"]
                    )
                ),
                "",
            ]
        )
    if not calibration_df.empty:
        compact_cols = [
            "context_method",
            "run_seed",
            "threshold",
            "validation_threshold_score",
            "test_default_accuracy",
            "test_calibrated_accuracy",
            "test_accuracy_delta",
            "test_default_balanced_accuracy",
            "test_calibrated_balanced_accuracy",
            "test_balanced_accuracy_delta",
            "test_default_predicted_positive_rate",
            "test_calibrated_predicted_positive_rate",
            "test_calibrated_collapse_flag",
            "test_calibrated_collapse_direction",
        ]
        compact_cols = [column for column in compact_cols if column in calibration_df.columns]
        lines.extend(
            [
                "## Validation Threshold Calibration",
                "",
                _frame_to_markdown(
                    calibration_df[compact_cols].sort_values(
                        ["run_seed", "context_method"]
                    )
                ),
                "",
            ]
        )
    if not pairwise_df.empty:
        compact_cols = [
            "run_seed",
            "split",
            "prediction_agreement_rate",
            "prob_up_correlation",
            "right_minus_left_prob_up_mean",
            "left_predicted_positive_rate",
            "right_predicted_positive_rate",
            "p7_up_p8_down_count",
            "p7_down_p8_up_count",
            "p7_up_p8_down_label_positive_rate",
            "p7_down_p8_up_label_positive_rate",
        ]
        compact_cols = [column for column in compact_cols if column in pairwise_df.columns]
        lines.extend(
            [
                "## Pairwise P7/P8 Comparison",
                "",
                _frame_to_markdown(
                    pairwise_df[compact_cols].sort_values(["split", "run_seed"])
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Next Decision Rule",
            "",
            "- If validation threshold calibration fixes seed 43/44 on test, prioritize calibrated evaluation/checkpoint criteria.",
            "- If ROC-AUC is good but class distribution collapses, avoid blind gamma/beta scale search.",
            "- If both validation and test collapse in the same direction, inspect train history and gamma/beta modulation exports for those seeds.",
            "",
        ]
    )
    return "\n".join(lines)


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    """Return a Markdown table with a dependency-free fallback."""

    try:
        return frame.to_markdown(index=False)
    except Exception:
        return "```text\n" + frame.to_string(index=False) + "\n```"


if __name__ == "__main__":
    main()
