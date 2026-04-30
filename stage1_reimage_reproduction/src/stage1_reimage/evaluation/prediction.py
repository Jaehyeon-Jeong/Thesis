"""Prediction CSV and metric utilities for Stage 1.

Source context:
    The Stage 1 plan fixes a GitHub-style CNN that returns two logits. Paper
    interpretation uses the softmax Up probability, so this module applies
    `softmax(logits, dim=1)` only after training. The `prob_up >= 0.5` tie rule
    is an explicit implementation convention recorded in config and docs; it is
    not a separately reported paper detail.

How to read this file:
    Training creates checkpoints. This file turns a checkpoint into human-readable
    prediction rows and metric JSONs. It is where logits become probabilities.

Leakage rule:
    Future returns are used only as labels/evaluation metadata. They are never
    passed into the model as inputs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from stage1_reimage.config import get_config_section

PREDICTION_ID_COLUMNS: tuple[str, ...] = (
    "Date",
    "StockID",
    "year",
    "local_row",
    "global_index",
    "split",
    "experiment_name",
    "image_window",
    "target_horizon",
    "target_return_name",
)

SEED_PREDICTION_COLUMNS: tuple[str, ...] = (
    *PREDICTION_ID_COLUMNS,
    "target_return",
    "label",
    "MarketCap",
    "EWMA_vol",
    "Ret_5d",
    "Ret_20d",
    "Ret_60d",
    "Ret_month",
    "run_seed",
    "checkpoint_path",
    "logit_down",
    "logit_up",
    "prob_down",
    "prob_up",
    "pred_class",
    "correct",
)


@dataclass(frozen=True)
class EvaluationSettings:
    """Stage 1 evaluation settings parsed from config.

    These settings control how logits are converted into predicted classes and
    how correlation diagnostics are computed.
    """

    threshold: float
    tie_break_class: int
    probability_source: str
    average_probabilities_across_seeds: bool
    batch_size: int
    min_correlation_group_size: int


def evaluation_settings_from_config(config: Mapping[str, Any]) -> EvaluationSettings:
    """Build `EvaluationSettings` from the `evaluation` config section.

    Output:
        Settings object passed into `predict_loader()` and averaging functions.
    """

    evaluation_config = get_config_section(config, "evaluation")
    training_config = get_config_section(config, "training")
    return EvaluationSettings(
        threshold=float(evaluation_config.get("threshold", 0.5)),
        tie_break_class=int(evaluation_config.get("tie_break_class", 1)),
        probability_source=str(evaluation_config.get("probability_source", "softmax_logits")),
        average_probabilities_across_seeds=bool(
            evaluation_config.get("average_probabilities_across_seeds", True)
        ),
        batch_size=int(evaluation_config.get("batch_size", training_config.get("batch_size", 128))),
        min_correlation_group_size=int(evaluation_config.get("min_correlation_group_size", 3)),
    )


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """Load a Stage 1 checkpoint into `model` and return checkpoint metadata.

    Input:
        Empty `StockCNNI20` model and path to `best.pt`.

    Effect:
        The model object is modified in-place so its parameters match the saved
        checkpoint. After this, `model(images)` uses learned weights.
    """

    checkpoint_file = Path(checkpoint_path).expanduser()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    checkpoint = _torch_load_checkpoint(checkpoint_file, map_location=torch.device(device))
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_file}")
    # `model_state_dict` contains tensors for every Conv/BatchNorm/Linear
    # parameter. Loading it replaces the freshly initialized model weights.
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint


def predict_loader(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    checkpoint_path: str | Path,
    experiment_name: str,
    image_window: str,
    target_horizon: str,
    run_seed: int,
    split_name: str,
    settings: EvaluationSettings,
    device: torch.device | str,
) -> pd.DataFrame:
    """Run model prediction and return a prediction-output frame.

    Input:
        batch image tensor: `(batch_size, 1, 64, 60)`.
        model output: logits with shape `(batch_size, 2)`.

    Output:
        One row per image with metadata, logits, softmax probabilities,
        predicted class, and correctness.
    """

    if settings.probability_source != "softmax_logits":
        raise ValueError(f"Unsupported probability_source: {settings.probability_source}")

    device = torch.device(device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in data_loader:
            # Evaluation uses the same image tensor shape as training:
            # images `(B, 1, 64, 60)`, labels `(B,)`.
            # Metadata is kept separately and is not passed to the model.
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.long)

            # Forward pass:
            #   images `(B, 1, 64, 60)` -> logits `(B, 2)`.
            logits = model(images)

            # Convert logits to probabilities only for interpretation/output.
            # Column 0 is Down/non-positive, column 1 is Up.
            probabilities = torch.softmax(logits, dim=1)

            logit_array = logits.detach().cpu().numpy()
            probability_array = probabilities.detach().cpu().numpy()
            label_array = labels.detach().cpu().numpy()
            # `pred_array` shape is `(B,)`; values are 0 or 1.
            pred_array = _predict_from_prob_up(
                probability_array[:, 1],
                threshold=settings.threshold,
                tie_break_class=settings.tie_break_class,
            )
            # DataLoader collates metadata into a dict of batched values. Convert
            # it back to one metadata dictionary per prediction row.
            metadata_rows = _metadata_batch_to_records(batch["metadata"])

            for row_index, metadata in enumerate(metadata_rows):
                # Build one CSV row for one stock-date image. This row combines
                # original metadata, model scores, probabilities, and correctness.
                prediction_row = dict(metadata)
                prediction_row["split"] = split_name
                prediction_row["experiment_name"] = experiment_name
                prediction_row["image_window"] = image_window
                prediction_row["target_horizon"] = target_horizon
                prediction_row["run_seed"] = int(run_seed)
                prediction_row["checkpoint_path"] = str(checkpoint_path)
                prediction_row["logit_down"] = float(logit_array[row_index, 0])
                prediction_row["logit_up"] = float(logit_array[row_index, 1])
                prediction_row["prob_down"] = float(probability_array[row_index, 0])
                prediction_row["prob_up"] = float(probability_array[row_index, 1])
                prediction_row["pred_class"] = int(pred_array[row_index])
                prediction_row["label"] = int(label_array[row_index])
                prediction_row["correct"] = int(pred_array[row_index] == label_array[row_index])
                rows.append(prediction_row)

    prediction_frame = pd.DataFrame(rows)
    return _order_columns(prediction_frame, SEED_PREDICTION_COLUMNS)


def compute_classification_metrics(
    predictions: pd.DataFrame,
    probability_column: str = "prob_up",
    prediction_column: str = "pred_class",
) -> dict[str, Any]:
    """Compute binary classification metrics from prediction rows.

    Input:
        Prediction DataFrame with `label`, probability column, and `pred_class`.

    Output:
        JSON-ready dictionary for `<split>_metrics.json`.
    """

    _require_columns(predictions, ["label", probability_column, prediction_column])
    labels = predictions["label"].astype(int).to_numpy()
    probabilities = predictions[probability_column].astype(float).to_numpy()
    pred_classes = predictions[prediction_column].astype(int).to_numpy()

    sample_count = int(labels.size)
    if sample_count == 0:
        raise ValueError("Cannot compute metrics for zero predictions.")

    positive_count = int(labels.sum())
    negative_count = int(sample_count - positive_count)
    predicted_positive_count = int(pred_classes.sum())
    predicted_negative_count = int(sample_count - predicted_positive_count)

    # Confusion matrix terms:
    #   TP: predicted Up and actual Up
    #   TN: predicted Down/non-positive and actual Down/non-positive
    true_positive = int(((labels == 1) & (pred_classes == 1)).sum())
    true_negative = int(((labels == 0) & (pred_classes == 0)).sum())
    false_positive = int(((labels == 0) & (pred_classes == 1)).sum())
    false_negative = int(((labels == 1) & (pred_classes == 0)).sum())

    warnings: dict[str, str] = {}
    accuracy = float((pred_classes == labels).mean())
    majority_class_accuracy = float(max(positive_count, negative_count) / sample_count)
    precision = _safe_ratio(
        true_positive,
        true_positive + false_positive,
        "precision",
        warnings,
    )
    recall = _safe_ratio(
        true_positive,
        true_positive + false_negative,
        "recall",
        warnings,
    )
    if precision is None or recall is None or (precision + recall) == 0.0:
        f1 = None
        warnings["f1"] = "Undefined because precision and/or recall is undefined or zero."
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))

    # Probability diagnostics use `prob_up`, not hard class predictions.
    brier_score = float(np.mean(np.square(probabilities - labels)))
    log_loss_value = _binary_log_loss(labels, probabilities)
    roc_auc = _sklearn_metric("roc_auc", labels, probabilities, warnings)
    average_precision = _sklearn_metric("average_precision", labels, probabilities, warnings)

    return {
        "num_samples": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": float(positive_count / sample_count),
        "negative_rate": float(negative_count / sample_count),
        "predicted_positive_count": predicted_positive_count,
        "predicted_negative_count": predicted_negative_count,
        "predicted_positive_rate": float(predicted_positive_count / sample_count),
        "accuracy": accuracy,
        "majority_class_accuracy": majority_class_accuracy,
        "accuracy_minus_majority_class_accuracy": float(
            accuracy - majority_class_accuracy
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier_score": brier_score,
        "log_loss": log_loss_value,
        "confusion_matrix": {
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
        },
        "metric_warnings": warnings,
    }


def compute_correlation_metrics(
    predictions: pd.DataFrame,
    probability_column: str = "prob_up",
    return_column: str = "target_return",
    date_column: str = "Date",
    min_group_size: int = 3,
) -> dict[str, Any]:
    """Compute prediction-return correlation diagnostics.

    This checks whether larger Up probabilities are associated with larger
    realized future returns. It is not the same as classification accuracy.
    """

    _require_columns(predictions, [probability_column, return_column, date_column])
    frame = predictions[[date_column, probability_column, return_column]].dropna().copy()
    frame[probability_column] = frame[probability_column].astype(float)
    frame[return_column] = frame[return_column].astype(float)

    # Global correlation pools all stock-date rows together.
    global_pearson = _series_corr(frame[probability_column], frame[return_column], "pearson")
    global_spearman = _series_corr(frame[probability_column], frame[return_column], "spearman")
    datewise_pearson: list[float] = []
    datewise_spearman: list[float] = []
    skipped_dates = 0

    # Date-wise correlation asks: on the same date, do stocks with higher
    # predicted Up probability have higher realized returns?
    for _, group in frame.groupby(date_column, sort=True):
        if len(group) < min_group_size:
            skipped_dates += 1
            continue
        pearson_value = _series_corr(group[probability_column], group[return_column], "pearson")
        spearman_value = _series_corr(group[probability_column], group[return_column], "spearman")
        if pearson_value is None:
            skipped_dates += 1
        else:
            datewise_pearson.append(float(pearson_value))
        if spearman_value is not None:
            datewise_spearman.append(float(spearman_value))

    return {
        "num_rows_used": int(len(frame)),
        "probability_column": probability_column,
        "return_column": return_column,
        "global_pearson_prob_return": global_pearson,
        "global_spearman_prob_return": global_spearman,
        "datewise": {
            "min_group_size": int(min_group_size),
            "num_dates_total": int(frame[date_column].nunique()),
            "num_dates_used_pearson": int(len(datewise_pearson)),
            "num_dates_used_spearman": int(len(datewise_spearman)),
            "num_dates_skipped": int(skipped_dates),
            "mean_pearson": _array_stat(datewise_pearson, "mean"),
            "std_pearson": _array_stat(datewise_pearson, "std"),
            "median_pearson": _array_stat(datewise_pearson, "median"),
            "mean_spearman": _array_stat(datewise_spearman, "mean"),
            "std_spearman": _array_stat(datewise_spearman, "std"),
            "median_spearman": _array_stat(datewise_spearman, "median"),
        },
    }


def write_evaluation_outputs(
    predictions: pd.DataFrame,
    classification_metrics: Mapping[str, Any],
    correlation_metrics: Mapping[str, Any],
    predictions_dir: str | Path,
    metrics_dir: str | Path,
    split_name: str,
) -> dict[str, str]:
    """Write predictions and metric JSON files for one split.

    Files written:
        `<split>_predictions.csv`
        `<split>_metrics.json`
        `<split>_correlation_metrics.json`
    """

    prediction_path = Path(predictions_dir)
    metric_path = Path(metrics_dir)
    prediction_path.mkdir(parents=True, exist_ok=True)
    metric_path.mkdir(parents=True, exist_ok=True)

    prediction_file = prediction_path / f"{split_name}_predictions.csv"
    metrics_file = metric_path / f"{split_name}_metrics.json"
    correlation_file = metric_path / f"{split_name}_correlation_metrics.json"

    predictions.to_csv(prediction_file, index=False)
    _write_json(metrics_file, classification_metrics)
    _write_json(correlation_file, correlation_metrics)
    return {
        "predictions": str(prediction_file),
        "classification_metrics": str(metrics_file),
        "correlation_metrics": str(correlation_file),
    }


def average_seed_predictions(
    prediction_paths: Sequence[str | Path],
    run_seeds: Sequence[int],
    settings: EvaluationSettings,
) -> pd.DataFrame:
    """Average seed-level softmax probabilities into a paper-style ensemble file.

    Input:
        Multiple seed prediction CSVs with the same rows in the same order.

    Output:
        DataFrame with seed probability columns, `mean_prob_up`, `std_prob_up`,
        averaged `pred_class`, and correctness.
    """

    if len(prediction_paths) != len(run_seeds):
        raise ValueError("prediction_paths and run_seeds must have the same length.")
    if not prediction_paths:
        raise ValueError("At least one seed prediction file is required.")

    # Each seed file contains probabilities from a separately trained model.
    seed_frames = [pd.read_csv(path) for path in prediction_paths]
    base = seed_frames[0].copy()
    _require_columns(base, [*PREDICTION_ID_COLUMNS, "label", "target_return", "prob_up", "prob_down"])

    output = base[
        [
            *PREDICTION_ID_COLUMNS,
            "target_return",
            "label",
            "MarketCap",
            "EWMA_vol",
            "Ret_5d",
            "Ret_20d",
            "Ret_60d",
            "Ret_month",
        ]
    ].copy()

    prob_up_columns: list[str] = []
    prob_down_columns: list[str] = []
    for seed, frame in zip(run_seeds, seed_frames, strict=True):
        _require_columns(frame, [*PREDICTION_ID_COLUMNS, "prob_up", "prob_down"])
        if not _same_prediction_identity(base, frame):
            raise ValueError(f"Prediction identity columns do not match for seed {seed}.")
        prob_up_column = f"prob_up_seed_{seed}"
        prob_down_column = f"prob_down_seed_{seed}"
        output[prob_up_column] = frame["prob_up"].astype(float).to_numpy()
        output[prob_down_column] = frame["prob_down"].astype(float).to_numpy()
        prob_up_columns.append(prob_up_column)
        prob_down_columns.append(prob_down_column)

    # Paper-style averaging happens after softmax. We do not average logits
    # because logits are not calibrated across independently initialized models.
    output["mean_prob_up"] = output[prob_up_columns].mean(axis=1)
    output["std_prob_up"] = output[prob_up_columns].std(axis=1, ddof=0)
    output["mean_prob_down"] = output[prob_down_columns].mean(axis=1)
    output["std_prob_down"] = output[prob_down_columns].std(axis=1, ddof=0)
    output["pred_class"] = _predict_from_prob_up(
        output["mean_prob_up"].to_numpy(dtype=float),
        threshold=settings.threshold,
        tie_break_class=settings.tie_break_class,
    )
    output["correct"] = (output["pred_class"].astype(int) == output["label"].astype(int)).astype(int)
    return output


def _metadata_batch_to_records(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert PyTorch default-collated metadata dict into row dictionaries.

    DataLoader turns a list of metadata dictionaries into one dictionary whose
    values are batches. This function reverses that so prediction CSV writing can
    create one row per original sample.
    """

    if not isinstance(metadata, Mapping):
        raise TypeError("Batch metadata must be a mapping.")
    first_value = next(iter(metadata.values()))
    batch_size = _collated_length(first_value)
    records: list[dict[str, Any]] = []
    for row_index in range(batch_size):
        record: dict[str, Any] = {}
        for key, value in metadata.items():
            record[key] = _collated_value_at(value, row_index)
        records.append(record)
    return records


def _collated_length(value: Any) -> int:
    """Return batch length for a collated metadata value."""

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return len(value)
    return 1


def _collated_value_at(value: Any, index: int) -> Any:
    """Extract one scalar metadata value from a collated object."""

    if isinstance(value, torch.Tensor):
        element = value[index] if value.ndim > 0 else value
        return _python_scalar(element)
    if isinstance(value, np.ndarray):
        return _python_scalar(value[index])
    if isinstance(value, pd.Series):
        return _python_scalar(value.iloc[index])
    if isinstance(value, (list, tuple)):
        return _python_scalar(value[index])
    return _python_scalar(value)


def _python_scalar(value: Any) -> Any:
    """Convert numpy/torch scalar values into plain Python values."""

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _predict_from_prob_up(
    prob_up: np.ndarray,
    threshold: float,
    tie_break_class: int,
) -> np.ndarray:
    """Convert Up probability into class labels using the configured tie rule.

    Input:
        `prob_up` array with shape `(num_samples,)`.

    Output:
        Integer array with shape `(num_samples,)`, values 0 or 1.
    """

    if tie_break_class == 1:
        return (prob_up >= threshold).astype(np.int8)
    if tie_break_class == 0:
        return (prob_up > threshold).astype(np.int8)
    raise ValueError(f"tie_break_class must be 0 or 1, got: {tie_break_class}")


def _safe_ratio(
    numerator: int,
    denominator: int,
    metric_name: str,
    warnings: dict[str, str],
) -> float | None:
    """Return numerator/denominator or record an undefined metric."""

    if denominator == 0:
        warnings[metric_name] = "Undefined because denominator is zero."
        return None
    return float(numerator / denominator)


def _binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute binary log loss with clipping for numerical stability."""

    clipped = np.clip(probabilities.astype(float), 1.0e-15, 1.0 - 1.0e-15)
    labels = labels.astype(float)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def _sklearn_metric(
    metric_name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    warnings: dict[str, str],
) -> float | None:
    """Compute optional sklearn metrics while keeping sklearn an optional dependency."""

    if len(np.unique(labels)) < 2:
        warnings[metric_name] = "Undefined because y_true has one class."
        return None
    try:
        if metric_name == "roc_auc":
            from sklearn.metrics import roc_auc_score  # pylint: disable=import-outside-toplevel

            return float(roc_auc_score(labels, probabilities))
        if metric_name == "average_precision":
            from sklearn.metrics import average_precision_score  # pylint: disable=import-outside-toplevel

            return float(average_precision_score(labels, probabilities))
    except Exception as exc:  # pragma: no cover - defensive dependency boundary
        warnings[metric_name] = f"sklearn metric failed: {exc}"
        return None
    raise ValueError(f"Unsupported sklearn metric: {metric_name}")


def _series_corr(series_a: pd.Series, series_b: pd.Series, method: str) -> float | None:
    """Return Pearson/Spearman correlation or None if undefined."""

    if len(series_a) < 2 or series_a.nunique(dropna=True) < 2 or series_b.nunique(dropna=True) < 2:
        return None
    value = series_a.corr(series_b, method=method)
    if pd.isna(value):
        return None
    return float(value)


def _array_stat(values: Sequence[float], stat_name: str) -> float | None:
    """Return an aggregate statistic for a possibly empty sequence."""

    if not values:
        return None
    array = np.asarray(values, dtype=float)
    if stat_name == "mean":
        return float(np.mean(array))
    if stat_name == "std":
        return float(np.std(array))
    if stat_name == "median":
        return float(np.median(array))
    raise ValueError(f"Unsupported stat: {stat_name}")


def _same_prediction_identity(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Check whether two prediction frames describe rows in the same order."""

    if len(left) != len(right):
        return False
    return bool(left.loc[:, PREDICTION_ID_COLUMNS].equals(right.loc[:, PREDICTION_ID_COLUMNS]))


def _order_columns(frame: pd.DataFrame, preferred_columns: Sequence[str]) -> pd.DataFrame:
    """Order known columns first while retaining any extra metadata columns."""

    for column in preferred_columns:
        if column not in frame.columns:
            frame[column] = None
    extra_columns = [column for column in frame.columns if column not in preferred_columns]
    return frame.loc[:, [*preferred_columns, *extra_columns]]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error if a frame is missing required columns."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with numpy/pandas values converted to plain Python values."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(payload), file, indent=2, sort_keys=True)
        file.write("\n")


def _json_ready(value: Any) -> Any:
    """Convert common scientific Python values into JSON-serializable values."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _torch_load_checkpoint(path: Path, map_location: torch.device) -> dict[str, Any]:
    """Load a torch checkpoint across PyTorch versions."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
