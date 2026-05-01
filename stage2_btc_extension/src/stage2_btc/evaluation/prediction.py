"""Stage 2 prediction export and classification metrics.

역할:
    학습된 BTC CNN checkpoint를 사람이 읽을 수 있는 prediction CSV와 metric JSON으로
    바꾼다.

Leakage guard:
    `future_return`과 `label`은 출력/평가 metadata로만 사용한다. model input에는
    image tensor만 전달된다.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from stage2_btc.config import get_config_section


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """checkpoint의 model_state_dict를 model에 load한다."""

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint


def predict_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    config: Mapping[str, Any],
    device: torch.device | str,
    run_context: Mapping[str, Any],
    checkpoint_path: str | Path,
    split_name: str,
) -> pd.DataFrame:
    """DataLoader 전체에 대해 prediction row DataFrame을 만든다."""

    evaluation_config = get_config_section(config, "evaluation")
    threshold = float(evaluation_config.get("threshold", 0.5))
    tie_class = int(evaluation_config.get("threshold_tie_class", 1))
    device = torch.device(device)
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.long)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            logit_array = logits.detach().cpu().numpy()
            prob_array = probabilities.detach().cpu().numpy()
            label_array = labels.detach().cpu().numpy()
            pred_array = _pred_from_prob_up(prob_array[:, 1], threshold, tie_class)
            metadata_rows = _metadata_batch_to_records(batch["metadata"])

            for index, metadata in enumerate(metadata_rows):
                row = dict(metadata)
                row.update(
                    {
                        "split": split_name,
                        "experiment_name": run_context["experiment_name"],
                        "run_seed": int(run_context["run_seed"]),
                        "checkpoint_path": str(checkpoint_path),
                        "logit_down": float(logit_array[index, 0]),
                        "logit_up": float(logit_array[index, 1]),
                        "prob_down": float(prob_array[index, 0]),
                        "prob_up": float(prob_array[index, 1]),
                        "pred_class": int(pred_array[index]),
                        "label": int(label_array[index]),
                        "correct": int(pred_array[index] == label_array[index]),
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows)


def compute_classification_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    """BTC binary prediction table에서 classification metric을 계산한다."""

    labels = predictions["label"].astype(int).to_numpy()
    pred = predictions["pred_class"].astype(int).to_numpy()
    prob = predictions["prob_up"].astype(float).to_numpy()
    future_return = predictions["future_return"].astype(float).to_numpy()
    n = int(len(labels))
    if n == 0:
        raise ValueError("Cannot compute metrics for zero predictions.")

    tp = int(((labels == 1) & (pred == 1)).sum())
    tn = int(((labels == 0) & (pred == 0)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    warnings: dict[str, str] = {}
    metrics = {
        "num_samples": n,
        "accuracy": float((labels == pred).mean()),
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
        "positive_rate": float(labels.mean()),
        "predicted_positive_count": int(pred.sum()),
        "predicted_negative_count": int((pred == 0).sum()),
        "predicted_positive_rate": float(pred.mean()),
        "majority_class_accuracy": float(max(labels.mean(), 1.0 - labels.mean())),
        "precision": _safe_ratio(tp, tp + fp, "precision", warnings),
        "recall": _safe_ratio(tp, tp + fn, "recall", warnings),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "brier_score": float(np.mean((prob - labels) ** 2)),
        "log_loss": _binary_log_loss(labels, prob),
        "roc_auc": _sk_metric("roc_auc_score", labels, prob, warnings),
        "average_precision": _sk_metric("average_precision_score", labels, prob, warnings),
        "probability_return_pearson": _corr(prob, future_return),
        "metric_warnings": warnings,
    }
    precision = metrics["precision"]
    recall = metrics["recall"]
    metrics["f1"] = (
        0.0
        if precision + recall == 0
        else float(2.0 * precision * recall / (precision + recall))
    )
    metrics["calibration_10_bins"] = _calibration_bins(labels, prob)
    return metrics


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """JSON artifact를 저장한다."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _pred_from_prob_up(prob_up: np.ndarray, threshold: float, tie_class: int) -> np.ndarray:
    """prob_up과 threshold로 class prediction을 만든다."""

    if tie_class == 1:
        return (prob_up >= threshold).astype(int)
    return (prob_up > threshold).astype(int)


def _metadata_batch_to_records(metadata_batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    """PyTorch default collate가 묶은 metadata dict를 row dict list로 되돌린다."""

    keys = list(metadata_batch.keys())
    if not keys:
        return []
    first = metadata_batch[keys[0]]
    batch_size = len(first)
    records: list[dict[str, Any]] = []
    for row_index in range(batch_size):
        record: dict[str, Any] = {}
        for key in keys:
            value = metadata_batch[key]
            item = value[row_index]
            if hasattr(item, "item"):
                item = item.item()
            record[key] = item
        records.append(record)
    return records


def _safe_ratio(numerator: int, denominator: int, name: str, warnings: dict[str, str]) -> float:
    """0으로 나눌 수 있는 metric을 안전하게 계산한다."""

    if denominator == 0:
        warnings[name] = "denominator was zero"
        return 0.0
    return float(numerator / denominator)


def _binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """binary log loss를 계산한다."""

    clipped = np.clip(probabilities, 1.0e-15, 1.0 - 1.0e-15)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def _sk_metric(
    metric_name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    warnings: dict[str, str],
) -> float | None:
    """scikit-learn metric이 있으면 사용하고 없으면 None을 반환한다."""

    try:
        import sklearn.metrics as sk_metrics
    except ImportError:
        warnings[metric_name] = "scikit-learn is not installed"
        return None
    try:
        metric = getattr(sk_metrics, metric_name)
        return float(metric(labels, probabilities))
    except Exception as exc:  # noqa: BLE001 - metric failure should be recorded.
        warnings[metric_name] = str(exc)
        return None


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    """Pearson correlation을 계산한다."""

    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _calibration_bins(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    """prob_up 10-bin calibration table을 만든다."""

    bins = np.linspace(0.0, 1.0, 11)
    result: list[dict[str, Any]] = []
    for index in range(10):
        low = bins[index]
        high = bins[index + 1]
        if index == 9:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        count = int(mask.sum())
        result.append(
            {
                "bin": int(index),
                "prob_low": float(low),
                "prob_high": float(high),
                "count": count,
                "mean_prob_up": None if count == 0 else float(probabilities[mask].mean()),
                "empirical_positive_rate": None if count == 0 else float(labels[mask].mean()),
            }
        )
    return result
