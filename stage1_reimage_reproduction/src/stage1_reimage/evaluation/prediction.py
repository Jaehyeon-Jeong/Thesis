"""1단계 prediction CSV와 metric utility.

근거 맥락:
    1단계 plan은 두 개 logits를 반환하는 GitHub-style CNN을 고정한다. 논문 해석은
    softmax Up probability를 사용하므로, 이 module은 training 이후에만
    `softmax(logits, dim=1)`를 적용한다. `prob_up >= 0.5` tie rule은 config와
    docs에 기록한 implementation convention이며, 논문에 별도로 보고된 detail은 아니다.

읽는 법:
    training은 checkpoint를 만든다. 이 파일은 checkpoint를 사람이 읽을 수 있는
    prediction row와 metric JSON으로 바꾼다. logits가 probability로 변환되는 곳이다.

Leakage rule:
    future return은 label/evaluation metadata로만 사용한다. model input으로 절대
    전달하지 않는다.
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
    """config에서 parsing한 1단계 evaluation setting.

    이 setting은 logits를 predicted class로 바꾸는 방식과 correlation diagnostic
    계산 방식을 제어한다.
    """

    threshold: float
    tie_break_class: int
    probability_source: str
    average_probabilities_across_seeds: bool
    batch_size: int
    min_correlation_group_size: int


def evaluation_settings_from_config(config: Mapping[str, Any]) -> EvaluationSettings:
    """config의 `evaluation` section에서 `EvaluationSettings`를 만든다.

    출력:
        `predict_loader()`와 averaging 함수에 전달되는 settings 객체.
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
    """1단계 checkpoint를 `model`에 load하고 checkpoint metadata를 반환한다.

    입력:
        빈 `StockCNNI20` model과 `best.pt` path.

    효과:
        model 객체가 in-place로 수정되어 parameter가 저장된 checkpoint와 같아진다.
        이후 `model(images)`는 학습된 weight를 사용한다.
    """

    checkpoint_file = Path(checkpoint_path).expanduser()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    checkpoint = _torch_load_checkpoint(checkpoint_file, map_location=torch.device(device))
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_file}")
    # `model_state_dict`에는 모든 Conv/BatchNorm/Linear parameter tensor가 들어 있다.
    # 이것을 load하면 방금 초기화된 model weight가 학습된 weight로 교체된다.
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
    """model prediction을 실행하고 prediction-output DataFrame을 반환한다.

    입력:
        batch image tensor: `(batch_size, 1, 64, 60)`.
        model output: logits with shape `(batch_size, 2)`.

    출력:
        image 하나당 row 하나. metadata, logits, softmax probability, predicted class,
        correctness를 포함한다.
    """

    if settings.probability_source != "softmax_logits":
        raise ValueError(f"Unsupported probability_source: {settings.probability_source}")

    device = torch.device(device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in data_loader:
            # evaluation은 training과 같은 image tensor shape를 사용한다:
            # images `(B, 1, 64, 60)`, labels `(B,)`.
            # metadata는 따로 보존하고 model에는 전달하지 않는다.
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.long)

            # 순전파:
            #   images `(B, 1, 64, 60)` -> logits `(B, 2)`.
            logits = model(images)

            # logits를 interpretation/output용 probability로만 변환한다.
            # column 0은 Down/non-positive, column 1은 Up이다.
            probabilities = torch.softmax(logits, dim=1)

            logit_array = logits.detach().cpu().numpy()
            probability_array = probabilities.detach().cpu().numpy()
            label_array = labels.detach().cpu().numpy()
            # `pred_array` shape는 `(B,)`이고 값은 0 또는 1이다.
            pred_array = _predict_from_prob_up(
                probability_array[:, 1],
                threshold=settings.threshold,
                tie_break_class=settings.tie_break_class,
            )
            # DataLoader는 metadata를 batched value dict로 collate한다. prediction CSV
            # row를 만들기 위해 sample 하나당 metadata dict 하나로 다시 바꾼다.
            metadata_rows = _metadata_batch_to_records(batch["metadata"])

            for row_index, metadata in enumerate(metadata_rows):
                # stock-date image 하나에 대한 CSV row 하나를 만든다. 이 row는 원본
                # metadata, model score, probability, correctness를 합친다.
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
    """prediction row에서 binary classification metric을 계산한다.

    입력:
        `label`, probability column, `pred_class`가 있는 prediction DataFrame.

    출력:
        `<split>_metrics.json`으로 저장 가능한 JSON-ready dictionary.
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

    # 혼동행렬 항목:
    #   TP: Up으로 예측했고 실제도 Up
    #   TN: Down/non-positive로 예측했고 실제도 Down/non-positive
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

    # probability diagnostic은 hard class prediction이 아니라 `prob_up`을 사용한다.
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
    """prediction-return correlation diagnostic을 계산한다.

    더 큰 Up probability가 더 큰 realized future return과 연결되는지 확인한다.
    classification accuracy와는 다른 진단이다.
    """

    _require_columns(predictions, [probability_column, return_column, date_column])
    frame = predictions[[date_column, probability_column, return_column]].dropna().copy()
    frame[probability_column] = frame[probability_column].astype(float)
    frame[return_column] = frame[return_column].astype(float)

    # global correlation은 모든 stock-date row를 한꺼번에 pooling한다.
    global_pearson = _series_corr(frame[probability_column], frame[return_column], "pearson")
    global_spearman = _series_corr(frame[probability_column], frame[return_column], "spearman")
    datewise_pearson: list[float] = []
    datewise_spearman: list[float] = []
    skipped_dates = 0

    # date-wise correlation은 같은 날짜에서 predicted Up probability가 높은 stock이
    # realized return도 더 높은지 묻는다.
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
    """하나의 split에 대한 prediction CSV와 metric JSON을 저장한다.

    저장 파일:
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
    """seed별 softmax probability를 평균해서 paper-style ensemble file을 만든다.

    입력:
        같은 row를 같은 순서로 가진 여러 seed prediction CSV.

    출력:
        seed probability column, `mean_prob_up`, `std_prob_up`, averaged
        `pred_class`, correctness가 포함된 DataFrame.
    """

    if len(prediction_paths) != len(run_seeds):
        raise ValueError("prediction_paths and run_seeds must have the same length.")
    if not prediction_paths:
        raise ValueError("At least one seed prediction file is required.")

    # 각 seed file에는 독립적으로 학습된 model의 probability가 들어 있다.
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

    # paper-style averaging은 softmax 이후 probability에서 수행한다. independent
    # initialization model 사이에서 logits는 calibration이 맞지 않을 수 있으므로 평균하지 않는다.
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
    """PyTorch default-collated metadata dict를 row dictionary로 되돌린다.

    DataLoader는 metadata dictionary list를 value가 batch인 dictionary 하나로 바꾼다.
    이 함수는 그 과정을 되돌려 prediction CSV가 원본 sample 하나당 row 하나를 만들게 한다.
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
    """collated metadata value의 batch length를 반환한다."""

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return len(value)
    return 1


def _collated_value_at(value: Any, index: int) -> Any:
    """collated object에서 scalar metadata value 하나를 꺼낸다."""

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
    """numpy/torch scalar 값을 plain Python 값으로 바꾼다."""

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
    """설정된 tie rule로 Up probability를 class label로 바꾼다.

    입력:
        `prob_up` array with shape `(num_samples,)`.

    출력:
        shape `(num_samples,)`의 integer array. 값은 0 또는 1.
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
    """numerator/denominator를 반환하거나 undefined metric을 기록한다."""

    if denominator == 0:
        warnings[metric_name] = "Undefined because denominator is zero."
        return None
    return float(numerator / denominator)


def _binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """수치 안정성을 위해 clipping을 적용해 binary log loss를 계산한다."""

    clipped = np.clip(probabilities.astype(float), 1.0e-15, 1.0 - 1.0e-15)
    labels = labels.astype(float)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def _sklearn_metric(
    metric_name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    warnings: dict[str, str],
) -> float | None:
    """sklearn을 optional dependency로 유지하면서 optional sklearn metric을 계산한다."""

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
    """Pearson/Spearman correlation을 반환하고, undefined이면 None을 반환한다."""

    if len(series_a) < 2 or series_a.nunique(dropna=True) < 2 or series_b.nunique(dropna=True) < 2:
        return None
    value = series_a.corr(series_b, method=method)
    if pd.isna(value):
        return None
    return float(value)


def _array_stat(values: Sequence[float], stat_name: str) -> float | None:
    """비어 있을 수 있는 sequence에 대해 aggregate statistic을 반환한다."""

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
    """두 prediction frame이 같은 row를 같은 순서로 설명하는지 확인한다."""

    if len(left) != len(right):
        return False
    return bool(left.loc[:, PREDICTION_ID_COLUMNS].equals(right.loc[:, PREDICTION_ID_COLUMNS]))


def _order_columns(frame: pd.DataFrame, preferred_columns: Sequence[str]) -> pd.DataFrame:
    """알려진 column을 먼저 배치하고 extra metadata column도 보존한다."""

    for column in preferred_columns:
        if column not in frame.columns:
            frame[column] = None
    extra_columns = [column for column in frame.columns if column not in preferred_columns]
    return frame.loc[:, [*preferred_columns, *extra_columns]]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """필수 column이 없으면 명확한 error를 낸다."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """numpy/pandas 값을 plain Python 값으로 바꾼 뒤 JSON을 저장한다."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(payload), file, indent=2, sort_keys=True)
        file.write("\n")


def _json_ready(value: Any) -> Any:
    """일반적인 scientific Python 값을 JSON-serializable 값으로 바꾼다."""

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
    """PyTorch version 차이를 고려해서 torch checkpoint를 load한다."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
