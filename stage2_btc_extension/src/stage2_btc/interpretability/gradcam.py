"""Stage 2 BTC Grad-CAM utilities.

중요:
    이 그림은 raw feature map이 아니다. 선택 class logit에 대한 activation과
    gradient로 만든 class-discriminative heatmap이다.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn


def compute_gradcam_for_image(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    target_layers: Mapping[str, nn.Module],
    output_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """image 하나에 대해 layer별 Grad-CAM heatmap을 계산한다."""

    if image.ndim == 3:
        image = image.unsqueeze(0)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    handles = []

    def make_forward(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            activations[name] = output

        return hook

    def make_backward(name: str):
        def hook(
            _module: nn.Module,
            _grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            gradients[name] = grad_output[0]

        return hook

    for name, layer in target_layers.items():
        handles.append(layer.register_forward_hook(make_forward(name)))
        handles.append(layer.register_full_backward_hook(make_backward(name)))

    try:
        model.zero_grad(set_to_none=True)
        logits = model(image)
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


def select_gradcam_samples(
    predictions: pd.DataFrame,
    samples_per_class: int,
) -> pd.DataFrame:
    """prediction CSV에서 confidence 높은 Up/Down sample을 고른다."""

    frames = []
    for panel, target_class, score_column in [
        ("up", 1, "prob_up"),
        ("down", 0, "prob_down"),
    ]:
        panel_frame = predictions[predictions["pred_class"].astype(int).eq(target_class)].copy()
        if panel_frame.empty:
            # Smoke test에서는 작은 sample 때문에 한 class 예측이 아예 없을 수 있다.
            # 이 경우에도 Grad-CAM code path를 확인할 수 있도록 해당 class probability가
            # 높은 sample을 fallback으로 사용한다. Full report에서는 predicted Up/Down
            # sample이 충분히 있는지 별도로 확인한다.
            panel_frame = predictions.copy()
        panel_frame = panel_frame.sort_values(score_column, ascending=False).head(samples_per_class)
        panel_frame["gradcam_panel"] = panel
        panel_frame["target_class"] = target_class
        frames.append(panel_frame)
    selected = pd.concat(frames, ignore_index=True)
    if selected.empty:
        raise ValueError("No Grad-CAM samples selected.")
    return selected


def generate_gradcam_figure(
    model: nn.Module,
    dataset: Any,
    predictions: pd.DataFrame,
    output_path: str | Path,
    samples_per_class: int,
    device: torch.device | str,
) -> tuple[Path, pd.DataFrame]:
    """Figure 13-style BTC Grad-CAM grid를 저장한다."""

    device = torch.device(device)
    model.to(device).eval()
    selected = select_gradcam_samples(predictions, samples_per_class=samples_per_class)
    sample_lookup = {
        int(row["sample_index"]): idx
        for idx, row in dataset.samples.reset_index(drop=True).iterrows()
    }
    target_layers = model.gradcam_target_layers()
    layer_names = list(target_layers.keys())
    rows_per_panel = 1 + len(layer_names)
    cols = max(samples_per_class, 1)
    fig, axes = plt.subplots(
        rows_per_panel * 2,
        cols,
        figsize=(cols * 1.7, rows_per_panel * 2.0),
        squeeze=False,
    )

    records: list[dict[str, Any]] = []
    for panel_index, panel_name in enumerate(["up", "down"]):
        panel_rows = selected[selected["gradcam_panel"].eq(panel_name)].reset_index(drop=True)
        for col in range(cols):
            if col >= len(panel_rows):
                for row_offset in range(rows_per_panel):
                    axes[panel_index * rows_per_panel + row_offset, col].axis("off")
                continue
            prediction_row = panel_rows.iloc[col]
            dataset_index = sample_lookup[int(prediction_row["sample_index"])]
            item = dataset[dataset_index]
            image = item["image"].to(device).unsqueeze(0)
            heatmaps = compute_gradcam_for_image(
                model,
                image,
                int(prediction_row["target_class"]),
                target_layers,
                output_size=tuple(image.shape[-2:]),
            )
            raw_image = image[0, 0].detach().cpu().numpy()
            base_row = panel_index * rows_per_panel
            axes[base_row, col].imshow(raw_image, cmap="gray")
            axes[base_row, col].axis("off")
            axes[base_row, col].set_title(str(prediction_row["Date"])[:10], fontsize=8)
            for layer_offset, layer_name in enumerate(layer_names, start=1):
                axes[base_row + layer_offset, col].imshow(heatmaps[layer_name], cmap="jet")
                axes[base_row + layer_offset, col].axis("off")
            records.append(prediction_row.to_dict())

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output, pd.DataFrame(records)


def _normalize(array: np.ndarray) -> np.ndarray:
    """heatmap을 0-1 범위로 정규화한다."""

    array = np.asarray(array, dtype=np.float32)
    max_value = float(np.max(array))
    min_value = float(np.min(array))
    if max_value == min_value:
        return np.zeros_like(array)
    return (array - min_value) / (max_value - min_value)
