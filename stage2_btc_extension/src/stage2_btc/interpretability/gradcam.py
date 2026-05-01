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
    """prediction CSV에서 confidence 높은 predicted Up/Down sample을 고른다.

    Grad-CAM은 학습/추론이 끝난 뒤 계산하는 post-hoc explanation이다. 여기서는
    Re-image Figure 13 형식에 맞춰 실제 label 기준이 아니라 모델 예측 기준으로
    `Predicted Up`과 `Predicted Down` panel을 나눈다. 실제 label과 correct 여부는
    `samples.csv`와 figure title에 함께 남겨서, 맞춘 예측과 틀린 예측의 heatmap을
    나중에 비교할 수 있게 한다.
    """

    frames = []
    for panel, panel_label, target_class, score_column in [
        ("up", "Predicted Up", 1, "prob_up"),
        ("down", "Predicted Down", 0, "prob_down"),
    ]:
        panel_frame = predictions[predictions["pred_class"].astype(int).eq(target_class)].copy()
        used_fallback = False
        if panel_frame.empty:
            # Smoke test에서는 작은 sample 때문에 한 class 예측이 아예 없을 수 있다.
            # 이 경우에도 Grad-CAM code path를 확인할 수 있도록 해당 class probability가
            # 높은 sample을 fallback으로 사용한다. Full report에서는 predicted Up/Down
            # sample이 충분히 있는지 별도로 확인한다.
            panel_frame = predictions.copy()
            used_fallback = True
        panel_frame = panel_frame.sort_values(score_column, ascending=False).head(samples_per_class)
        panel_frame["gradcam_panel"] = panel
        panel_frame["gradcam_panel_label"] = panel_label
        panel_frame["target_class"] = target_class
        panel_frame["target_class_name"] = _class_name(target_class)
        panel_frame["gradcam_sample_fallback"] = used_fallback
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
    """Figure 13-style BTC Grad-CAM grid를 저장한다.

    Figure layout:
        위쪽 panel은 `Predicted Up`, 아래쪽 panel은 `Predicted Down`이다.
        각 panel의 첫 row는 원본 chart image이고, 이어지는 row들은 CNN layer별
        Grad-CAM heatmap이다. heatmap은 target predicted class의 pre-softmax logit에
        대한 activation-gradient 조합으로 계산한다.
    """

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
            axes[base_row, col].set_title(_sample_title(prediction_row), fontsize=7)
            if col == 0:
                axes[base_row, col].set_ylabel(
                    f"{_panel_label(panel_name)}\nOriginal",
                    fontsize=8,
                    rotation=0,
                    labelpad=36,
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
                        labelpad=36,
                        va="center",
                    )
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


def _class_name(class_id: int) -> str:
    """class id를 사람이 읽는 이름으로 바꾼다."""

    return "Up" if int(class_id) == 1 else "Down"


def _panel_label(panel_name: str) -> str:
    """figure panel label을 반환한다."""

    return "Predicted Up" if panel_name == "up" else "Predicted Down"


def _sample_title(prediction_row: pd.Series) -> str:
    """Grad-CAM column title에 들어갈 sample 설명을 만든다.

    제목은 좁은 chart 위에 올라가기 때문에 길이를 제한한다. 더 자세한 정보는 같은
    directory의 `samples.csv`에 저장된다.
    """

    date = str(prediction_row.get("Date", ""))[:10]
    pred = _class_name(int(prediction_row.get("pred_class", prediction_row.get("target_class", 0))))
    actual = _class_name(int(prediction_row.get("label", 0)))
    correct = "T" if int(prediction_row.get("correct", 0)) == 1 else "F"
    prob_up = float(prediction_row.get("prob_up", 0.0))
    return f"{date}\nPred {pred} pU={prob_up:.2f}\nTrue {actual} {correct}"
