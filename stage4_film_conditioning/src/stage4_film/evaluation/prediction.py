"""Stage 4 prediction export helpers.

Stage 2 prediction utilities assume `model(image)`. Stage 4 models require
`model(image, context)`, so this module keeps the same output schema while
adding the context tensor path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from stage2_btc.config import get_config_section


def load_stage4_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """Load a Stage 4 checkpoint into a model and switch it to eval mode."""

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))
    state_dict = checkpoint["model_state_dict"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(_strip_module_prefix(state_dict))
    model.to(device)
    model.eval()
    return checkpoint


def predict_context_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    config: Mapping[str, Any],
    device: torch.device | str,
    run_context: Mapping[str, Any],
    checkpoint_path: str | Path,
    split_name: str,
    context_feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build prediction rows for a Stage 4 context-conditioned model."""

    evaluation_config = get_config_section(config, "evaluation")
    threshold = float(evaluation_config.get("threshold", 0.5))
    tie_class = int(evaluation_config.get("threshold_tie_class", 1))
    device = torch.device(device)
    rows: list[dict[str, Any]] = []
    feature_names = [str(name) for name in context_feature_names or []]

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            contexts = batch["context"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.long)
            logits = model(images, contexts)
            probabilities = torch.softmax(logits, dim=1)

            logit_array = logits.detach().cpu().numpy()
            prob_array = probabilities.detach().cpu().numpy()
            label_array = labels.detach().cpu().numpy()
            context_array = contexts.detach().cpu().numpy()
            pred_array = _pred_from_prob_up(prob_array[:, 1], threshold, tie_class)
            metadata_rows = _metadata_batch_to_records(batch["metadata"])

            for index, metadata in enumerate(metadata_rows):
                row = dict(metadata)
                row.update(
                    {
                        "split": split_name,
                        "experiment_name": run_context["experiment_name"],
                        "stage4_experiment_name": run_context.get(
                            "stage4_experiment_name",
                            run_context["experiment_name"],
                        ),
                        "context_method": run_context["context_method"],
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
                for feature_index, feature_name in enumerate(feature_names):
                    row[f"context_{feature_name}"] = float(context_array[index, feature_index])
                rows.append(row)

    return pd.DataFrame(rows)


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Handle checkpoints saved from `nn.DataParallel`."""

    prefix = "module."
    if not any(str(key).startswith(prefix) for key in state_dict):
        return dict(state_dict)
    return {
        str(key)[len(prefix) :] if str(key).startswith(prefix) else str(key): value
        for key, value in state_dict.items()
    }


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
