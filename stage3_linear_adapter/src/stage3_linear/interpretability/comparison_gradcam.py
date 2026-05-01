"""Baseline-vs-Linear Grad-CAM figure helper.

역할:
    Stage 3 Linear model의 Grad-CAM만 따로 저장하면 Stage 2 baseline과 어떤 점이
    달라졌는지 보기 어렵다. 이 helper는 Linear model이 선택한 같은 BTC sample에 대해
    Stage 2 baseline과 Stage 3 Linear의 마지막 convolution Grad-CAM을 나란히 저장한다.

해석 기준:
    sample 선택은 Stage 3 Linear prediction 기준이다. 각 sample에서 target class는
    Linear model의 predicted class logit이다. Baseline heatmap도 같은 target class에
    대해 계산해서 "같은 질문에 대해 두 모델이 어디를 봤는지" 비교한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from stage2_btc.interpretability.gradcam import (
    compute_gradcam_for_image,
    select_gradcam_samples,
)


def generate_baseline_linear_comparison_figure(
    baseline_model: nn.Module,
    linear_model: nn.Module,
    dataset: Any,
    linear_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    output_path: str | Path,
    samples_per_class: int,
    device: torch.device | str,
) -> tuple[Path, pd.DataFrame]:
    """같은 sample에 대해 baseline과 Linear 마지막 layer Grad-CAM을 비교한다."""

    device = torch.device(device)
    baseline_model.to(device).eval()
    linear_model.to(device).eval()

    selected = select_gradcam_samples(linear_predictions, samples_per_class=samples_per_class)
    baseline_lookup = (
        baseline_predictions.copy()
        .assign(sample_index=lambda frame: frame["sample_index"].astype(int))
        .set_index("sample_index")
    )
    sample_lookup = {
        int(row["sample_index"]): idx
        for idx, row in dataset.samples.reset_index(drop=True).iterrows()
    }

    baseline_layers = baseline_model.gradcam_target_layers()
    linear_layers = linear_model.gradcam_target_layers()
    baseline_last_name = list(baseline_layers.keys())[-1]
    linear_last_name = list(linear_layers.keys())[-1]

    rows_per_panel = 3
    cols = max(samples_per_class, 1)
    fig, axes = plt.subplots(rows_per_panel * 2, cols, figsize=(cols * 2.2, 7.0), squeeze=False)
    records: list[dict[str, Any]] = []

    for panel_index, panel_name in enumerate(["up", "down"]):
        panel_rows = selected[selected["gradcam_panel"].eq(panel_name)].reset_index(drop=True)
        for col in range(cols):
            base_row = panel_index * rows_per_panel
            if col >= len(panel_rows):
                for row_offset in range(rows_per_panel):
                    axes[base_row + row_offset, col].axis("off")
                continue

            linear_row = panel_rows.iloc[col]
            sample_index = int(linear_row["sample_index"])
            dataset_index = sample_lookup[sample_index]
            item = dataset[dataset_index]
            image = item["image"].to(device).unsqueeze(0)
            target_class = int(linear_row["target_class"])

            baseline_heatmaps = compute_gradcam_for_image(
                baseline_model,
                image,
                target_class,
                baseline_layers,
                output_size=tuple(image.shape[-2:]),
            )
            linear_heatmaps = compute_gradcam_for_image(
                linear_model,
                image,
                target_class,
                linear_layers,
                output_size=tuple(image.shape[-2:]),
            )

            baseline_row = baseline_lookup.loc[sample_index] if sample_index in baseline_lookup.index else None
            raw_image = image[0, 0].detach().cpu().numpy()
            axes[base_row, col].imshow(raw_image, cmap="gray")
            axes[base_row, col].axis("off")
            axes[base_row, col].set_title(_comparison_title(linear_row, baseline_row), fontsize=7)
            axes[base_row + 1, col].imshow(baseline_heatmaps[baseline_last_name], cmap="jet")
            axes[base_row + 1, col].axis("off")
            axes[base_row + 2, col].imshow(linear_heatmaps[linear_last_name], cmap="jet")
            axes[base_row + 2, col].axis("off")

            if col == 0:
                axes[base_row, col].set_ylabel(_panel_label(panel_name), fontsize=8, rotation=0, labelpad=34, va="center")
                axes[base_row + 1, col].set_ylabel(f"Stage2\n{baseline_last_name}", fontsize=8, rotation=0, labelpad=34, va="center")
                axes[base_row + 2, col].set_ylabel(f"Stage3\n{linear_last_name}", fontsize=8, rotation=0, labelpad=34, va="center")

            record = linear_row.to_dict()
            if baseline_row is not None:
                record.update(
                    {
                        "baseline_pred_class": int(baseline_row["pred_class"]),
                        "baseline_prob_up": float(baseline_row["prob_up"]),
                        "baseline_correct": int(baseline_row["correct"]),
                    }
                )
            records.append(record)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output, pd.DataFrame(records)


def _panel_label(panel_name: str) -> str:
    """panel 이름을 figure label로 바꾼다."""

    return "Predicted Up" if panel_name == "up" else "Predicted Down"


def _class_name(class_id: int) -> str:
    """class id를 사람이 읽는 이름으로 바꾼다."""

    return "Up" if int(class_id) == 1 else "Down"


def _comparison_title(linear_row: pd.Series, baseline_row: pd.Series | None) -> str:
    """좁은 chart 위에 들어갈 비교 title을 만든다."""

    date = str(linear_row.get("Date", ""))[:10]
    true_name = _class_name(int(linear_row.get("label", 0)))
    linear_pred = _class_name(int(linear_row.get("pred_class", 0)))
    linear_prob = float(linear_row.get("prob_up", 0.0))
    if baseline_row is None:
        return f"{date}\nLinear {linear_pred} pU={linear_prob:.2f}\nTrue {true_name}"
    baseline_pred = _class_name(int(baseline_row.get("pred_class", 0)))
    baseline_prob = float(baseline_row.get("prob_up", 0.0))
    return (
        f"{date}\nL {linear_pred} pU={linear_prob:.2f} | "
        f"B {baseline_pred} pU={baseline_prob:.2f}\nTrue {true_name}"
    )

