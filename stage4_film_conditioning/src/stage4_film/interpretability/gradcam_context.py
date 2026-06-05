"""Stage 4 Grad-CAM plus context/modulation export utilities.

Stage 2 Grad-CAM explains `model(image)`. Stage 4 explains
`model(image, context)` and additionally records the conditioning path:

- concat: normalized context values and context embedding summary.
- gating: context values plus raw gate and gate summaries/full arrays.
- FiLM: context values plus block-wise gamma/beta summaries/full arrays.

The Grad-CAM target remains the predicted-class pre-softmax logit.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn

from stage2_btc.interpretability.gradcam import select_gradcam_samples


def compute_stage4_gradcam_for_image(
    model: nn.Module,
    image: torch.Tensor,
    context: torch.Tensor,
    target_class: int,
    target_layers: Mapping[str, nn.Module],
    output_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Compute layer-wise Grad-CAM heatmaps for one Stage 4 sample."""

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if context.ndim == 1:
        context = context.unsqueeze(0)

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    handles = []

    def make_forward(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            # Tensor-level hooks avoid the `register_full_backward_hook` +
            # in-place activation conflict that can happen when hooking the
            # post-FiLM tensor immediately before an in-place LeakyReLU.
            activations[name] = output.detach().clone()
            output.register_hook(
                lambda grad, layer_name=name: gradients.__setitem__(
                    layer_name,
                    grad.detach().clone(),
                )
            )

        return hook

    for name, layer in target_layers.items():
        handles.append(layer.register_forward_hook(make_forward(name)))

    try:
        model.zero_grad(set_to_none=True)
        logits = model(image, context)
        score = logits[:, int(target_class)].sum()
        score.backward()

        heatmaps: dict[str, np.ndarray] = {}
        for name in target_layers:
            weights = gradients[name].mean(dim=(2, 3), keepdim=True)
            heatmap = torch.relu((weights * activations[name]).sum(dim=1, keepdim=True))
            heatmap = functional.interpolate(
                heatmap,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            array = heatmap[0, 0].detach().cpu().numpy()
            heatmaps[name] = _normalize(array)
        return heatmaps
    finally:
        for handle in handles:
            handle.remove()


def generate_stage4_gradcam_context_figure(
    model: nn.Module,
    dataset: Any,
    predictions: pd.DataFrame,
    output_path: str | Path,
    samples_per_class: int,
    device: torch.device | str,
    context_method: str,
    context_feature_names: Sequence[str],
    selected_samples: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Write Grad-CAM figure, selected samples, and modulation metadata.

    Returns a dictionary with paths and the selected/modulation dataframes so
    the caller can copy report artifacts without re-reading files.
    """

    device = torch.device(device)
    method = str(context_method)
    model.to(device).eval()

    selected = _prepare_selected_samples(
        selected_samples
        if selected_samples is not None
        else select_gradcam_samples(predictions, samples_per_class=samples_per_class)
    )
    sample_lookup = _sample_index_lookup(dataset)
    target_layers, target_layer_kind = _target_layers_for_method(model, method)
    layer_names = list(target_layers.keys())
    rows_per_panel = 1 + len(layer_names)
    panel_names = _ordered_panel_names(selected)
    cols = max(
        int(samples_per_class),
        int(selected.groupby("gradcam_panel").size().max()),
        1,
    )
    fig, axes = plt.subplots(
        rows_per_panel * len(panel_names),
        cols,
        figsize=(cols * 1.9, rows_per_panel * max(len(panel_names), 1) * 2.05),
        squeeze=False,
    )

    selected_records: list[dict[str, Any]] = []
    modulation_summary_rows: list[dict[str, Any]] = []
    modulation_value_records: list[dict[str, Any]] = []

    for panel_index, panel_name in enumerate(panel_names):
        panel_rows = selected[selected["gradcam_panel"].eq(panel_name)].reset_index(drop=True)
        for col in range(cols):
            if col >= len(panel_rows):
                for row_offset in range(rows_per_panel):
                    axes[panel_index * rows_per_panel + row_offset, col].axis("off")
                continue

            prediction_row = panel_rows.iloc[col]
            dataset_index = sample_lookup[int(prediction_row["sample_index"])]
            item = dataset[dataset_index]
            image = item["image"].to(device=device, dtype=torch.float32).unsqueeze(0)
            context = item["context"].to(device=device, dtype=torch.float32).unsqueeze(0)
            heatmaps = compute_stage4_gradcam_for_image(
                model,
                image,
                context,
                int(prediction_row["target_class"]),
                target_layers,
                output_size=tuple(image.shape[-2:]),
            )
            modulation = _extract_context_and_modulation(
                model,
                method,
                image,
                context,
                context_feature_names,
            )

            raw_image = image[0, 0].detach().cpu().numpy()
            base_row = panel_index * rows_per_panel
            axes[base_row, col].imshow(raw_image, cmap="gray")
            axes[base_row, col].axis("off")
            axes[base_row, col].set_title(_sample_title(prediction_row), fontsize=7)
            if col == 0:
                axes[base_row, col].set_ylabel(
                    f"{_panel_label_from_rows(panel_name, panel_rows)}\nOriginal",
                    fontsize=8,
                    rotation=0,
                    labelpad=38,
                    va="center",
                )

            for layer_offset, layer_name in enumerate(layer_names, start=1):
                axes[base_row + layer_offset, col].imshow(heatmaps[layer_name], cmap="jet")
                axes[base_row + layer_offset, col].axis("off")
                if col == 0:
                    axes[base_row + layer_offset, col].set_ylabel(
                        layer_name,
                        fontsize=8,
                        rotation=0,
                        labelpad=38,
                        va="center",
                    )

            selected_record = prediction_row.to_dict()
            selected_record.update(
                {
                    "context_method": method,
                    "gradcam_target_layer_kind": target_layer_kind,
                }
            )
            selected_record.update(modulation["summary"])
            selected_records.append(selected_record)

            modulation_summary_rows.append(
                {
                    "sample_index": int(prediction_row["sample_index"]),
                    "Date": str(prediction_row.get("Date", "")),
                    "gradcam_panel": str(prediction_row["gradcam_panel"]),
                    "target_class": int(prediction_row["target_class"]),
                    "context_method": method,
                    "gradcam_target_layer_kind": target_layer_kind,
                    **modulation["summary"],
                }
            )
            modulation_value_records.append(
                {
                    "sample_index": int(prediction_row["sample_index"]),
                    "Date": str(prediction_row.get("Date", "")),
                    "gradcam_panel": str(prediction_row["gradcam_panel"]),
                    "target_class": int(prediction_row["target_class"]),
                    "context_method": method,
                    "gradcam_target_layer_kind": target_layer_kind,
                    **modulation["values"],
                }
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"Stage 4 {method}: Grad-CAM with context/modulation export", fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    selected_frame = pd.DataFrame(selected_records)
    modulation_summary = pd.DataFrame(modulation_summary_rows)
    samples_path = output.parent / "samples.csv"
    modulation_summary_path = output.parent / "modulation_summary.csv"
    modulation_values_path = output.parent / "modulation_values.json"
    selected_frame.to_csv(samples_path, index=False)
    modulation_summary.to_csv(modulation_summary_path, index=False)
    modulation_values_path.write_text(
        json.dumps(modulation_value_records, indent=2),
        encoding="utf-8",
    )
    return {
        "figure": output,
        "samples": samples_path,
        "modulation_summary": modulation_summary_path,
        "modulation_values": modulation_values_path,
        "selected": selected_frame,
        "modulation_summary_frame": modulation_summary,
    }


def _prepare_selected_samples(selected: pd.DataFrame) -> pd.DataFrame:
    """Normalize targeted/default Grad-CAM selection columns."""

    frame = selected.copy()
    if frame.empty:
        raise ValueError("No Grad-CAM samples selected.")
    if "sample_index" not in frame.columns:
        raise KeyError("selected samples must include sample_index")
    if "target_class" not in frame.columns:
        if "pred_class" not in frame.columns:
            raise KeyError("selected samples need target_class or pred_class")
        frame["target_class"] = pd.to_numeric(frame["pred_class"], errors="raise").astype(int)
    if "target_class_name" not in frame.columns:
        frame["target_class_name"] = frame["target_class"].map(_class_name)
    if "gradcam_panel" not in frame.columns:
        frame["gradcam_panel"] = frame["target_class"].map(lambda value: "up" if int(value) == 1 else "down")
    if "gradcam_panel_label" not in frame.columns:
        frame["gradcam_panel_label"] = frame["gradcam_panel"].map(_panel_label)
    if "gradcam_sample_fallback" not in frame.columns:
        frame["gradcam_sample_fallback"] = False
    frame["sample_index"] = pd.to_numeric(frame["sample_index"], errors="raise").astype(int)
    frame["target_class"] = pd.to_numeric(frame["target_class"], errors="raise").astype(int)
    frame["gradcam_panel"] = frame["gradcam_panel"].astype(str)
    return frame


def _ordered_panel_names(selected: pd.DataFrame) -> list[str]:
    """Return panel names in first-seen order."""

    return list(dict.fromkeys(selected["gradcam_panel"].astype(str).tolist()))


def _target_layers_for_method(
    model: nn.Module,
    context_method: str,
) -> tuple[dict[str, nn.Module], str]:
    """Select Grad-CAM hook layers for the Stage 4 method."""

    if context_method in {
        "film_gamma",
        "film_full",
        "film_full_bounded_last_block",
        "film_full_uncertainty_gated_last_block",
        "film_full_confidence_gated_last_block",
    } and hasattr(model, "film_target_layers"):
        return dict(model.film_target_layers()), "post_film"
    return dict(model.gradcam_target_layers()), "conv"


def _sample_index_lookup(dataset: Any) -> dict[int, int]:
    """Map global sample_index to dataset row index."""

    base_dataset = getattr(dataset, "base_dataset", dataset)
    samples = getattr(base_dataset, "samples", None)
    if samples is None or "sample_index" not in samples.columns:
        return {
            int(dataset[index]["metadata"]["sample_index"]): index
            for index in range(len(dataset))
        }
    return {
        int(row["sample_index"]): idx
        for idx, row in samples.reset_index(drop=True).iterrows()
    }


def _extract_context_and_modulation(
    model: nn.Module,
    context_method: str,
    image: torch.Tensor,
    context: torch.Tensor,
    context_feature_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Return compact summary columns plus full modulation arrays."""

    with torch.no_grad():
        if context_method == "gating" and hasattr(model, "forward_with_gate_values"):
            output = model.forward_with_gate_values(image, context)
            context_embedding = output["context_embedding"]
            raw_gate = output["raw_gate"]
            gate = output["gate"]
            summary = {
                **_context_summary(context, context_feature_names),
                **_tensor_stats("context_embedding", context_embedding),
                **_tensor_stats("raw_gate", raw_gate),
                **_tensor_stats("gate", gate),
            }
            values = {
                "context": _named_context_values(context, context_feature_names),
                "context_embedding": _tensor_to_list(context_embedding),
                "raw_gate": _tensor_to_list(raw_gate),
                "gate": _tensor_to_list(gate),
                "gate_top_channels": _top_channels(gate),
                "gate_bottom_channels": _bottom_channels(gate),
            }
            return {"summary": summary, "values": values}

        if context_method in {
            "film_gamma",
            "film_full",
            "film_full_bounded_last_block",
            "film_full_uncertainty_gated_last_block",
            "film_full_confidence_gated_last_block",
        } and hasattr(
            model,
            "forward_with_modulation_values",
        ):
            output = model.forward_with_modulation_values(
                image,
                context,
                keep_feature_maps=False,
            )
            context_embedding = output["context_embedding"]
            summary = {
                **_context_summary(context, context_feature_names),
                **_tensor_stats("context_embedding", context_embedding),
            }
            block_values = []
            for block in output["film_parameters"]:
                block_id = int(block["block"])
                gamma = block["gamma"]
                beta = block["beta"]
                delta_gamma = block["delta_gamma"]
                summary.update(_tensor_stats(f"block{block_id}_gamma", gamma))
                summary.update(_tensor_stats(f"block{block_id}_delta_gamma", delta_gamma))
                block_record: dict[str, Any] = {
                    "block": block_id,
                    "gamma": _tensor_to_list(gamma),
                    "delta_gamma": _tensor_to_list(delta_gamma),
                    "gamma_top_channels": _top_channels(gamma),
                    "gamma_bottom_channels": _bottom_channels(gamma),
                }
                if beta is not None:
                    summary.update(_tensor_stats(f"block{block_id}_beta", beta))
                    block_record["beta"] = _tensor_to_list(beta)
                    block_record["beta_top_channels"] = _top_channels(beta)
                    block_record["beta_bottom_channels"] = _bottom_channels(beta)
                raw_gamma = block.get("raw_gamma")
                raw_beta = block.get("raw_beta")
                if raw_gamma is not None:
                    summary.update(_tensor_stats(f"block{block_id}_raw_gamma", raw_gamma))
                    block_record["raw_gamma"] = _tensor_to_list(raw_gamma)
                if raw_beta is not None:
                    summary.update(_tensor_stats(f"block{block_id}_raw_beta", raw_beta))
                    block_record["raw_beta"] = _tensor_to_list(raw_beta)
                if "modulation_scale" in block:
                    block_record["modulation_scale"] = float(block["modulation_scale"])
                modulation_gate = block.get("modulation_gate")
                stage2_prob_up = block.get("stage2_prob_up")
                if modulation_gate is not None:
                    summary.update(_tensor_stats(f"block{block_id}_modulation_gate", modulation_gate))
                    block_record["modulation_gate"] = _tensor_to_list(modulation_gate)
                if stage2_prob_up is not None:
                    summary.update(_tensor_stats(f"block{block_id}_stage2_prob_up", stage2_prob_up))
                    block_record["stage2_prob_up"] = _tensor_to_list(stage2_prob_up)
                block_values.append(block_record)
            values = {
                "context": _named_context_values(context, context_feature_names),
                "context_embedding": _tensor_to_list(context_embedding),
                "film_blocks": block_values,
            }
            return {"summary": summary, "values": values}

        context_embedding = model.context_encoder(context)
        summary = {
            **_context_summary(context, context_feature_names),
            **_tensor_stats("context_embedding", context_embedding),
        }
        values = {
            "context": _named_context_values(context, context_feature_names),
            "context_embedding": _tensor_to_list(context_embedding),
        }
        return {"summary": summary, "values": values}


def _context_summary(
    context: torch.Tensor,
    context_feature_names: Sequence[str],
) -> dict[str, float]:
    """Return one CSV column per normalized context feature."""

    values = _one_dim_numpy(context)
    names = [str(name) for name in context_feature_names]
    if len(names) != len(values):
        names = [f"feature_{index}" for index in range(len(values))]
    return {f"context_{name}": float(value) for name, value in zip(names, values, strict=True)}


def _named_context_values(
    context: torch.Tensor,
    context_feature_names: Sequence[str],
) -> dict[str, float]:
    """Return context values for JSON export."""

    values = _one_dim_numpy(context)
    names = [str(name) for name in context_feature_names]
    if len(names) != len(values):
        names = [f"feature_{index}" for index in range(len(values))]
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _tensor_stats(prefix: str, tensor: torch.Tensor | None) -> dict[str, float | int | None]:
    """Compact tensor statistics for CSV export."""

    if tensor is None:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_l2": None,
        }
    values = _one_dim_numpy(tensor)
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=0)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_l2": float(np.linalg.norm(values)),
    }


def _top_channels(tensor: torch.Tensor, top_k: int = 5) -> list[dict[str, float | int]]:
    """Return top-k channel values from a `(1, C)` tensor."""

    values = _one_dim_numpy(tensor)
    order = np.argsort(values)[::-1][: int(top_k)]
    return [{"channel": int(index), "value": float(values[index])} for index in order]


def _bottom_channels(tensor: torch.Tensor, top_k: int = 5) -> list[dict[str, float | int]]:
    """Return bottom-k channel values from a `(1, C)` tensor."""

    values = _one_dim_numpy(tensor)
    order = np.argsort(values)[: int(top_k)]
    return [{"channel": int(index), "value": float(values[index])} for index in order]


def _tensor_to_list(tensor: torch.Tensor | None) -> list[float] | None:
    """Convert a single-sample tensor to a JSON-safe list."""

    if tensor is None:
        return None
    return [float(value) for value in _one_dim_numpy(tensor)]


def _one_dim_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Flatten a single-sample tensor to one numpy vector."""

    return tensor.detach().cpu().reshape(-1).numpy().astype(float)


def _normalize(array: np.ndarray) -> np.ndarray:
    """Normalize heatmap values to `[0, 1]`."""

    array = np.asarray(array, dtype=np.float32)
    max_value = float(np.max(array))
    min_value = float(np.min(array))
    if max_value == min_value:
        return np.zeros_like(array)
    return (array - min_value) / (max_value - min_value)


def _class_name(class_id: int) -> str:
    """Return readable class name."""

    return "Up" if int(class_id) == 1 else "Down"


def _panel_label(panel_name: str) -> str:
    """Return readable Grad-CAM panel name."""

    return "Predicted Up" if panel_name == "up" else "Predicted Down"


def _panel_label_from_rows(panel_name: str, panel_rows: pd.DataFrame) -> str:
    """Return custom panel label when targeted rows provide one."""

    if "gradcam_panel_label" in panel_rows.columns and not panel_rows.empty:
        label = str(panel_rows.iloc[0].get("gradcam_panel_label", "")).strip()
        if label:
            return label
    return _panel_label(panel_name)


def _sample_title(prediction_row: pd.Series) -> str:
    """Compact figure column title."""

    date = str(prediction_row.get("Date", ""))[:10]
    pred = _class_name(int(prediction_row.get("pred_class", prediction_row.get("target_class", 0))))
    actual = _class_name(int(prediction_row.get("label", 0)))
    correct = "T" if int(prediction_row.get("correct", 0)) == 1 else "F"
    prob_up = float(prediction_row.get("prob_up", 0.0))
    return f"{date}\nPred {pred} pU={prob_up:.2f}\nTrue {actual} {correct}"
