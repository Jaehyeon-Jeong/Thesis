"""Stage 4 context-conditioned experiment runner.

The runner deliberately reuses Stage 2 BTC sample construction, chart image
generation, time split, and train-only pixel normalization. Stage 4 adds only:

    BTC/F&G context table -> train-only context scaler -> batch["context"]
    context model variant -> Stage 4 training loop
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stage2_btc.models.stock_cnn import count_parameters
from stage2_btc.runners.btc_baseline import (
    PreparedBtcData,
    build_dataloaders,
    prepare_btc_experiment_data,
    split_summary,
)
from stage4_film.config import (
    get_context_config,
    stage4_run_context_base,
    validate_context_method,
)
from stage4_film.context import (
    find_fear_greed_source,
    fit_transform_context_features,
    load_fear_greed_index,
    make_context_output_name,
)
from stage4_film.context.features import build_context_feature_audit, build_context_feature_table
from stage4_film.context.normalization import normalized_feature_columns
from stage4_film.models import (
    build_concat_context_stock_cnn_for_window,
    build_film_context_stock_cnn_for_window,
    build_gated_context_stock_cnn_for_window,
)
from stage4_film.paths import Stage4Paths, experiment_output_roots
from stage4_film.training import fit_context_model


@dataclass(frozen=True)
class PreparedStage4ContextData:
    """Stage 4 context-conditioned experiment 하나에 필요한 data 객체 모음."""

    source_file: str
    fear_greed_source_file: str
    btc_data: PreparedBtcData
    context_name: str
    context_features_path: str
    context_scaler_path: str
    context_feature_audit_path: str
    context_feature_summary_path: str
    context_table: pd.DataFrame
    context_scaler: Any
    context_audit: dict[str, Any]
    datasets: dict[str, "ContextBtcImageDataset"]


class ContextBtcImageDataset(Dataset):
    """Stage 2 BTC image dataset에 normalized context vector를 붙이는 wrapper."""

    def __init__(
        self,
        base_dataset: Dataset,
        context_table: pd.DataFrame,
        feature_columns: list[str],
    ) -> None:
        self.base_dataset = base_dataset
        self.feature_columns = list(feature_columns)
        if "sample_index" not in context_table.columns:
            raise KeyError("context_table must contain sample_index.")
        missing = [column for column in self.feature_columns if column not in context_table.columns]
        if missing:
            raise KeyError("context_table missing normalized feature column(s): " + ", ".join(missing))

        values = context_table.loc[:, ["sample_index", *self.feature_columns]].copy()
        values["sample_index"] = pd.to_numeric(values["sample_index"], errors="raise").astype(int)
        if values["sample_index"].duplicated().any():
            duplicates = values.loc[values["sample_index"].duplicated(), "sample_index"].head()
            raise ValueError(f"context_table contains duplicate sample_index values: {duplicates.tolist()}")
        sample_indices = values["sample_index"].to_numpy(dtype="int64")
        context_values = values.loc[:, self.feature_columns].to_numpy(dtype="float32")
        self._context_by_sample_index = {
            int(sample_index): row_values
            for sample_index, row_values in zip(sample_indices, context_values, strict=True)
        }
        self._validate_alignment()

    def __len__(self) -> int:
        """Dataset row count."""

        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return Stage 2 sample plus `context` tensor."""

        item = dict(self.base_dataset[index])
        metadata = dict(item["metadata"])
        sample_index = int(metadata["sample_index"])
        try:
            context_values = self._context_by_sample_index[sample_index]
        except KeyError as exc:
            raise KeyError(f"Missing context row for sample_index={sample_index}") from exc
        item["metadata"] = metadata
        item["context"] = torch.tensor(context_values, dtype=torch.float32)
        return item

    def _validate_alignment(self) -> None:
        """Fail early if any image sample lacks a context vector."""

        missing: list[int] = []
        samples = getattr(self.base_dataset, "samples", None)
        if samples is not None and "sample_index" in samples.columns:
            sample_indices = pd.to_numeric(samples["sample_index"], errors="raise").astype(int)
        else:
            sample_indices = [
                int(self.base_dataset[index]["metadata"]["sample_index"])
                for index in range(len(self.base_dataset))
            ]
        for sample_index in sample_indices:
            if sample_index not in self._context_by_sample_index:
                missing.append(int(sample_index))
                if len(missing) >= 5:
                    break
        if missing:
            raise KeyError(f"Missing context rows for sample_index values: {missing}")


def prepare_stage4_context_experiment_data(
    *,
    config: Mapping[str, Any],
    paths: Stage4Paths,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    run_seed: int,
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
    max_test_rows: int | None = None,
) -> PreparedStage4ContextData:
    """Stage 2 BTC data와 Stage 4 context table/dataset을 함께 준비한다."""

    btc_data = prepare_btc_experiment_data(
        config,
        paths,  # Stage4Paths is intentionally compatible with Stage2 path access.
        image_window=int(image_window),
        image_spec=str(image_spec),
        return_horizon=int(return_horizon),
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
        max_test_rows=max_test_rows,
    )

    context_config = get_context_config(config)
    context_window = int(context_config["context_window"])
    primary_features = [str(feature) for feature in context_config["primary_features"]]
    feature_columns = normalized_feature_columns(primary_features)

    fg_source = find_fear_greed_source(config, paths)
    fg_config = context_config.get("fear_greed", {})
    fear_greed = load_fear_greed_index(
        fg_source,
        date_column=str(fg_config.get("date_column", "date")),
        value_column=str(fg_config.get("value_column", "value")),
        classification_column=str(fg_config.get("classification_column", "classification")),
    )
    raw_context_table = build_context_feature_table(
        ohlcv=btc_data.ohlcv,
        samples_by_split=btc_data.splits,
        fear_greed=fear_greed,
        config=config,
    )
    context_table, context_scaler = fit_transform_context_features(raw_context_table, config)
    context_audit = build_context_feature_audit(context_table, primary_features)
    context_summary = _build_feature_summary(context_table, primary_features)

    context_name = make_context_output_name(
        image_window=int(image_window),
        image_spec=str(image_spec),
        return_horizon=int(return_horizon),
        context_window=context_window,
    )
    context_dir = paths.context_root / context_name / f"seed_{int(run_seed)}"
    context_dir.mkdir(parents=True, exist_ok=True)
    context_features_path = context_dir / "context_features.csv"
    context_scaler_path = context_dir / "context_scaler.json"
    context_feature_audit_path = context_dir / "context_feature_audit.json"
    context_feature_summary_path = context_dir / "context_feature_summary.csv"

    context_table.to_csv(context_features_path, index=False)
    context_scaler_path.write_text(
        json.dumps(_jsonable(context_scaler.as_dict()), indent=2),
        encoding="utf-8",
    )
    context_feature_audit_path.write_text(
        json.dumps(_jsonable(context_audit), indent=2),
        encoding="utf-8",
    )
    context_summary.to_csv(context_feature_summary_path, index=False)

    datasets = {
        split_name: ContextBtcImageDataset(
            base_dataset=base_dataset,
            context_table=context_table.loc[context_table["split"].astype(str).eq(split_name)],
            feature_columns=feature_columns,
        )
        for split_name, base_dataset in btc_data.datasets.items()
    }
    return PreparedStage4ContextData(
        source_file=btc_data.source_file,
        fear_greed_source_file=str(fg_source),
        btc_data=btc_data,
        context_name=context_name,
        context_features_path=str(context_features_path),
        context_scaler_path=str(context_scaler_path),
        context_feature_audit_path=str(context_feature_audit_path),
        context_feature_summary_path=str(context_feature_summary_path),
        context_table=context_table,
        context_scaler=context_scaler,
        context_audit=context_audit,
        datasets=datasets,
    )


def build_stage4_context_dataloaders(
    datasets: Mapping[str, ContextBtcImageDataset],
    config: Mapping[str, Any],
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    """Stage 4 context Dataset용 train/validation/test DataLoader를 만든다."""

    return build_dataloaders(datasets, config, shuffle_train=shuffle_train)


def build_stage4_context_model(
    *,
    config: Mapping[str, Any],
    image_window: int,
    context_method: str,
) -> nn.Module:
    """context_method에 맞는 Stage 4 model을 만든다."""

    method = validate_context_method(context_method)
    if method == "concat":
        return build_concat_context_stock_cnn_for_window(config, image_window=image_window)
    if method == "gating":
        return build_gated_context_stock_cnn_for_window(config, image_window=image_window)
    if method in {"film_gamma", "film_full"}:
        return build_film_context_stock_cnn_for_window(
            config,
            image_window=image_window,
            mode=method,
        )
    raise ValueError(f"Unsupported Stage 4 context method: {method}")


def run_stage4_context_training(
    *,
    config: Mapping[str, Any],
    paths: Stage4Paths,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    context_method: str,
    run_seed: int,
    device: torch.device | str,
    seed_info: Mapping[str, Any],
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
    max_test_rows: int | None = None,
) -> dict[str, Any]:
    """Prepare data, train one Stage 4 context model, and write a manifest."""

    method = validate_context_method(context_method)
    prepared = prepare_stage4_context_experiment_data(
        config=config,
        paths=paths,
        image_window=int(image_window),
        image_spec=str(image_spec),
        return_horizon=int(return_horizon),
        run_seed=int(run_seed),
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
        max_test_rows=max_test_rows,
    )
    dataloaders = build_stage4_context_dataloaders(prepared.datasets, config)
    model = build_stage4_context_model(
        config=config,
        image_window=int(image_window),
        context_method=method,
    )

    run_context = stage4_run_context_base(
        config,
        image_window=int(image_window),
        image_spec=str(image_spec),
        return_horizon=int(return_horizon),
        context_method=method,
        run_seed=int(run_seed),
    )
    experiment_name = str(run_context["stage4_experiment_name"])
    output_roots = experiment_output_roots(paths, experiment_name, int(run_seed))
    for directory in output_roots.values():
        directory.mkdir(parents=True, exist_ok=True)

    run_context.update(
        {
            "device": str(device),
            "source_file": prepared.source_file,
            "fear_greed_source_file": prepared.fear_greed_source_file,
            "num_parameters": count_parameters(model),
            "split_summary": split_summary(prepared.btc_data.splits),
            "context_name": prepared.context_name,
            "context_features_path": prepared.context_features_path,
            "context_scaler_path": prepared.context_scaler_path,
            "context_feature_audit_path": prepared.context_feature_audit_path,
            "context_feature_summary_path": prepared.context_feature_summary_path,
            "context_audit": prepared.context_audit,
            "seed_info": dict(seed_info),
        }
    )
    context_metadata = {
        "context_name": prepared.context_name,
        "feature_order": prepared.context_scaler.feature_order,
        "normalized_feature_columns": normalized_feature_columns(prepared.context_scaler.feature_order),
        "scaler": prepared.context_scaler.as_dict(),
        "audit": prepared.context_audit,
        "artifacts": {
            "context_features": prepared.context_features_path,
            "context_scaler": prepared.context_scaler_path,
            "context_feature_audit": prepared.context_feature_audit_path,
            "context_feature_summary": prepared.context_feature_summary_path,
        },
    }
    result = fit_context_model(
        model,
        dataloaders["train"],
        dataloaders["validation"],
        config,
        device=device,
        checkpoint_dir=output_roots["checkpoint"],
        metrics_dir=output_roots["metrics"],
        run_context=run_context,
        normalization_metadata=prepared.btc_data.normalization.as_dict(),
        context_metadata=context_metadata,
    )

    manifest_dir = paths.run_manifest_root / experiment_name / f"seed_{int(run_seed)}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "run_manifest.json"
    manifest = {
        "status": "ok",
        "run_context": run_context,
        "result": result.as_dict(),
        "normalization": prepared.btc_data.normalization.as_dict(),
        "context": context_metadata,
        "output_roots": {key: str(value) for key, value in output_roots.items()},
        "run_manifest": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return manifest


def _build_feature_summary(context_table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Build a compact split/feature summary table."""

    rows: list[dict[str, Any]] = []
    for split, split_frame in context_table.groupby("split", sort=True):
        for feature in features:
            raw = pd.to_numeric(split_frame[feature], errors="coerce")
            normalized = pd.to_numeric(split_frame[f"{feature}_normalized"], errors="coerce")
            rows.append(
                {
                    "split": str(split),
                    "feature": feature,
                    "num_rows": int(len(split_frame)),
                    "raw_missing_rate": float(raw.isna().mean()),
                    "raw_mean": _safe_float(raw.mean()),
                    "raw_std": _safe_float(raw.std(ddof=0)),
                    "raw_min": _safe_float(raw.min()),
                    "raw_max": _safe_float(raw.max()),
                    "normalized_mean": _safe_float(normalized.mean()),
                    "normalized_std": _safe_float(normalized.std(ddof=0)),
                    "normalized_min": _safe_float(normalized.min()),
                    "normalized_max": _safe_float(normalized.max()),
                }
            )
    return pd.DataFrame(rows)


def _safe_float(value: Any) -> float | None:
    """Return None for NaN-like values, otherwise float."""

    if pd.isna(value):
        return None
    return float(value)


def _jsonable(value: Any) -> Any:
    """Convert common pandas/numpy scalar values to JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value
