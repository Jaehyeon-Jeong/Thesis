"""Stage 1 Re-image baseline용 Grad-CAM 구현.

근거:
    - Grad-CAM 원전: Selvaraju et al., Grad-CAM, method section, Eq. (1)-(2).
      로컬 요약 기준 방법 설명은 pp.4-6에 해당한다.
    - Re-image 목표 그림: Jiang, Kelly, and Xiu, Re-Imagining Price Trends,
      Figure 13. 로컬 요약은 해석 섹션을 pp.41-49로 mapping한다.
    - Stage 1 model source: `lich99/Stock_CNN/models/baseline.py`,
      commit `415e2acf2a5013afca67e383acd3edc61fced840`.

중요한 구분:
    이 파일이 만드는 그림은 raw feature map이 아니다. 선택한 class logit에 대한
    feature activation과 gradient로 만든 class-discriminative heatmap이다.

Tensor 규칙:
    입력 image: `(1, 1, 64, 60)` 또는 `(1, 64, 60)`.
    target layer activation: `(1, channels, layer_height, layer_width)`.
    출력 heatmap: `(64, 60)` numpy array, `[0, 1]` 범위.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn


DEFAULT_LAYER_ORDER: tuple[str, ...] = ("layer1_conv", "layer2_conv", "layer3_conv")
INPUT_IMAGE_SIZE: tuple[int, int] = (64, 60)


@dataclass(frozen=True)
class GradCamSample:
    """Grad-CAM을 계산할 stock-date sample 하나.

    `prediction_row`는 prediction CSV에서 온 row다. 여기에는 Date, StockID,
    local_row, shard_index, label, probability, predicted class가 들어 있다.
    `target_class`는 Grad-CAM이 설명할 class logit이다. Figure 13 방식에서는
    predicted class를 설명하므로 Up panel은 1, Down panel은 0이다.
    """

    panel: str
    rank_in_panel: int
    target_class: int
    confidence: float
    prediction_row: dict[str, Any]

    @property
    def sample_id(self) -> str:
        """파일 이름에 쓸 수 있는 안정적인 sample id를 만든다."""

        date = str(self.prediction_row.get("Date", "unknown_date")).split("T")[0]
        stock_id = str(self.prediction_row.get("StockID", "unknown_stock"))
        local_row = str(self.prediction_row.get("local_row", "unknown_row"))
        raw_id = f"{self.panel}_{self.rank_in_panel:02d}_{date}_{stock_id}_{local_row}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id)


@dataclass(frozen=True)
class GradCamResult:
    """sample 하나에 대해 계산된 원본 이미지와 layer별 Grad-CAM heatmap."""

    sample: GradCamSample
    original_image: np.ndarray
    heatmaps: dict[str, np.ndarray]
    heatmap_warnings: dict[str, str]


class _GradCamHookStore:
    """선택한 convolution layer들의 activation과 gradient를 보관한다.

    PyTorch hook은 forward 때 activation을 저장하고 backward 때 gradient를 저장한다.
    Grad-CAM 수식의 `A^k`는 activation, `dy^c/dA^k`는 gradient에 해당한다.
    """

    def __init__(self, target_layers: Mapping[str, nn.Module]) -> None:
        self.activations: dict[str, torch.Tensor] = {}
        self.gradients: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        for layer_name, module in target_layers.items():
            self._handles.append(module.register_forward_hook(self._make_forward_hook(layer_name)))
            self._handles.append(
                module.register_full_backward_hook(self._make_backward_hook(layer_name))
            )

    def close(self) -> None:
        """등록한 hook을 제거한다.

        hook을 제거하지 않으면 다음 Grad-CAM 계산에서도 이전 hook이 살아 있어
        activation/gradient가 중복 저장될 수 있다.
        """

        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_forward_hook(self, layer_name: str) -> Any:
        """forward hook function을 만든다."""

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            # detach하지 않는다. backward 때 이 activation과 gradient가 연결돼 있어야
            # Grad-CAM 가중치 `alpha_k^c`를 계산할 수 있다.
            self.activations[layer_name] = output

        return hook

    def _make_backward_hook(self, layer_name: str) -> Any:
        """backward hook function을 만든다."""

        def hook(
            _module: nn.Module,
            _grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            # Conv2d module의 output에 대한 gradient가 Grad-CAM 수식의
            # `dy^c / dA^k`다.
            self.gradients[layer_name] = grad_output[0]

        return hook


def compute_gradcam_for_image(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    target_layers: Mapping[str, nn.Module],
    output_size: tuple[int, int] = INPUT_IMAGE_SIZE,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """image 하나에 대해 layer별 Grad-CAM heatmap을 계산한다.

    입력:
        model: 학습된 `StockCNNI20`. 이미 eval mode와 device 이동이 끝났다고 가정한다.
        image: normalized model input tensor. shape는 `(1, 1, 64, 60)`이어야 한다.
        target_class: 설명할 logit index. Down/non-positive는 0, Up은 1.
        target_layers: 이름 -> Conv2d module mapping.

    출력:
        layer name -> `(64, 60)` heatmap array.
        layer name -> warning message. ReLU 이후 값이 모두 0인 경우 기록한다.
    """

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Grad-CAM image must have 3 or 4 dimensions, got: {tuple(image.shape)}")

    model.zero_grad(set_to_none=True)
    hook_store = _GradCamHookStore(target_layers)
    try:
        # Grad-CAM은 softmax probability가 아니라 pre-softmax class logit `y^c`를
        # target score로 사용한다. 이는 Grad-CAM 원전 방법 섹션의 기본 정의다.
        logits = model(image)
        target_score = logits[:, int(target_class)].sum()
        target_score.backward()

        heatmaps: dict[str, np.ndarray] = {}
        warnings: dict[str, str] = {}
        for layer_name in target_layers:
            if layer_name not in hook_store.activations:
                raise KeyError(f"Missing activation for Grad-CAM layer: {layer_name}")
            if layer_name not in hook_store.gradients:
                raise KeyError(f"Missing gradient for Grad-CAM layer: {layer_name}")

            activation = hook_store.activations[layer_name]
            gradient = hook_store.gradients[layer_name]

            # Grad-CAM Eq. (1):
            #   alpha_k^c = spatial average of dy^c / dA^k.
            weights = gradient.mean(dim=(2, 3), keepdim=True)

            # Grad-CAM Eq. (2):
            #   L^c = ReLU(sum_k alpha_k^c A^k).
            heatmap = torch.relu((weights * activation).sum(dim=1, keepdim=True))

            # Re-image Figure 13은 입력 image 크기로 비교하는 grid다. layer별
            # activation 크기가 다르므로 모두 `(64, 60)`으로 bilinear upsampling한다.
            heatmap = functional.interpolate(
                heatmap,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            heatmap_2d = heatmap[0, 0].detach().cpu().numpy()
            heatmaps[layer_name], warning = _normalize_heatmap(heatmap_2d)
            if warning is not None:
                warnings[layer_name] = warning
        return heatmaps, warnings
    finally:
        hook_store.close()


def select_gradcam_samples(
    predictions: pd.DataFrame,
    year: int,
    samples_per_class: int,
    allow_fallback_any_year: bool = False,
) -> list[GradCamSample]:
    """prediction CSV에서 Figure 13-style Up/Down sample을 고른다.

    기본 규칙:
        1. `year == 2019` 같은 목표 연도만 남긴다.
        2. `pred_class == 1`은 Up panel, `pred_class == 0`은 Down panel로 나눈다.
        3. 각 panel에서 confidence가 높은 순서대로 고른다.

    논문은 Figure 13 sample 선택 기준을 자세히 보고하지 않으므로, confidence 기준
    deterministic selection은 재현 가능한 구현 선택이다.
    """

    required_columns = ["year", "pred_class", "prob_up", "prob_down", "shard_index", "local_row"]
    missing = [column for column in required_columns if column not in predictions.columns]
    if missing:
        raise KeyError(f"Prediction file missing required column(s): {', '.join(missing)}")

    candidates = predictions[predictions["year"].astype(int).eq(int(year))].copy()
    if candidates.empty and allow_fallback_any_year:
        candidates = predictions.copy()
    if candidates.empty:
        raise ValueError(f"No prediction rows available for Grad-CAM year={year}.")

    selected: list[GradCamSample] = []
    selection_specs = [
        ("up", 1, _confidence_column(candidates, target_class=1)),
        ("down", 0, _confidence_column(candidates, target_class=0)),
    ]
    for panel_name, target_class, confidence_column in selection_specs:
        panel_frame = candidates[candidates["pred_class"].astype(int).eq(target_class)].copy()
        if panel_frame.empty and allow_fallback_any_year:
            panel_frame = predictions[predictions["pred_class"].astype(int).eq(target_class)].copy()
        if panel_frame.empty:
            raise ValueError(f"No {panel_name} predictions available for Grad-CAM selection.")

        panel_frame["gradcam_confidence"] = panel_frame[confidence_column].astype(float)
        panel_frame = panel_frame.sort_values(
            ["gradcam_confidence", "Date", "StockID"],
            ascending=[False, True, True],
        ).head(samples_per_class)
        for rank, (_, row) in enumerate(panel_frame.iterrows(), start=1):
            selected.append(
                GradCamSample(
                    panel=panel_name,
                    rank_in_panel=rank,
                    target_class=target_class,
                    confidence=float(row["gradcam_confidence"]),
                    prediction_row=row.to_dict(),
                )
            )
    return selected


def make_figure13_style_grid(
    results: Sequence[GradCamResult],
    layer_order: Sequence[str] = DEFAULT_LAYER_ORDER,
    title: str | None = None,
) -> Any:
    """Re-image Figure 13과 같은 원본 이미지 + layer별 Grad-CAM grid를 만든다.

    출력:
        matplotlib Figure 객체. caller가 `.savefig()`로 저장한다.
    """

    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from matplotlib.colors import LinearSegmentedColormap  # pylint: disable=import-outside-toplevel

    up_results = [result for result in results if result.sample.panel == "up"]
    down_results = [result for result in results if result.sample.panel == "down"]
    panels = [("Images Receiving Up Classification", up_results), ("Images Receiving Down Classification", down_results)]
    max_columns = max(1, max((len(panel_results) for _, panel_results in panels), default=1))
    rows_per_panel = 1 + len(layer_order)
    total_rows = rows_per_panel * len(panels)

    # sample이 1개뿐인 local smoke test에서도 title이 겹치지 않도록 최소 폭을 둔다.
    figure_width = max(7.0, max_columns * 1.25)
    figure, axes = plt.subplots(
        total_rows,
        max_columns,
        figsize=(figure_width, total_rows * 1.05),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.04,
        right=0.99,
        bottom=0.02,
        top=0.91 if title else 0.95,
        hspace=0.18,
        wspace=0.04,
    )
    heatmap_cmap = LinearSegmentedColormap.from_list(
        "black_blue_cyan",
        [(0.0, "#000000"), (0.45, "#00106b"), (1.0, "#00e5ff")],
    )

    for panel_index, (panel_title, panel_results) in enumerate(panels):
        row_offset = panel_index * rows_per_panel
        for column_index in range(max_columns):
            for row_index in range(rows_per_panel):
                axes[row_offset + row_index, column_index].axis("off")

        if not panel_results:
            axes[row_offset, 0].text(0.5, 0.5, f"No samples: {panel_title}", ha="center", va="center")
            continue

        row_labels = ["Original", *layer_order]
        for label_index, row_label in enumerate(row_labels):
            axes[row_offset + label_index, 0].text(
                -0.08,
                0.5,
                row_label,
                transform=axes[row_offset + label_index, 0].transAxes,
                ha="right",
                va="center",
                fontsize=8,
            )

        for column_index, result in enumerate(panel_results):
            axes[row_offset, column_index].imshow(result.original_image, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_offset, column_index].set_title(
                f"{result.sample.rank_in_panel}",
                fontsize=8,
                pad=2,
            )
            for layer_position, layer_name in enumerate(layer_order, start=1):
                heatmap = result.heatmaps.get(layer_name)
                if heatmap is None:
                    continue
                axes[row_offset + layer_position, column_index].imshow(
                    heatmap,
                    cmap=heatmap_cmap,
                    vmin=0.0,
                    vmax=1.0,
                )

        # panel 제목은 실제 axes 위치를 기준으로 얹는다. 이렇게 하면 column 수가 적은
        # smoke figure에서도 큰 제목과 panel 제목이 서로 겹치지 않는다.
        panel_box = axes[row_offset, 0].get_position()
        y_position = panel_box.y1 + 0.012
        figure.text(0.5, y_position, panel_title, ha="center", va="center", fontsize=11)

    if title:
        figure.suptitle(title, fontsize=12, y=0.985)
    return figure


def write_gradcam_outputs(
    results: Sequence[GradCamResult],
    output_dir: str | Path,
    figure_title: str,
    year: int,
    split_name: str,
    layer_order: Sequence[str] = DEFAULT_LAYER_ORDER,
    report_figure_path: str | Path | None = None,
) -> dict[str, str]:
    """Grad-CAM heatmap, sample CSV, figure PNG, summary JSON을 저장한다.

    저장 구조:
        output_dir/
          figure13_style_<year>_<split>.png
          samples.csv
          summary.json
          heatmaps/<sample_id>_<layer>.npy
    """

    path = Path(output_dir)
    heatmap_dir = path / "heatmaps"
    path.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict[str, Any]] = []
    for result in results:
        row = _sample_to_csv_row(result.sample)
        row["heatmap_warnings"] = json.dumps(result.heatmap_warnings, ensure_ascii=False, sort_keys=True)
        for layer_name in layer_order:
            heatmap = result.heatmaps[layer_name]
            heatmap_path = heatmap_dir / f"{result.sample.sample_id}_{layer_name}.npy"
            np.save(heatmap_path, heatmap)
            row[f"heatmap_{layer_name}"] = str(heatmap_path)
        sample_rows.append(row)

    samples_path = path / "samples.csv"
    pd.DataFrame(sample_rows).to_csv(samples_path, index=False)

    figure = make_figure13_style_grid(results, layer_order=layer_order, title=figure_title)
    figure_path = path / f"figure13_style_{year}_{split_name}.png"
    figure.savefig(figure_path, dpi=180)
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        plt.close(figure)
    except Exception:
        pass

    written = {
        "figure": str(figure_path),
        "samples": str(samples_path),
        "heatmaps_dir": str(heatmap_dir),
    }
    if report_figure_path is not None:
        report_path = Path(report_figure_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure_path, report_path)
        written["report_figure"] = str(report_path)

    summary_path = path / "summary.json"
    summary = {
        "num_samples": len(results),
        "year": int(year),
        "split": split_name,
        "layer_order": list(layer_order),
        "figure": str(figure_path),
        "samples": str(samples_path),
        "source_note": "Grad-CAM 원전 Eq. (1)-(2), Re-image Figure 13-style layout.",
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    written["summary"] = str(summary_path)
    return written


def _normalize_heatmap(heatmap: np.ndarray) -> tuple[np.ndarray, str | None]:
    """heatmap을 `[0, 1]`로 min-max normalize한다."""

    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value <= min_value:
        return np.zeros_like(heatmap, dtype=np.float32), "Grad-CAM heatmap is all zero after ReLU."
    return ((heatmap - min_value) / (max_value - min_value)).astype(np.float32), None


def _confidence_column(frame: pd.DataFrame, target_class: int) -> str:
    """prediction frame에서 target class confidence column 이름을 고른다."""

    if target_class == 1:
        return "mean_prob_up" if "mean_prob_up" in frame.columns else "prob_up"
    return "mean_prob_down" if "mean_prob_down" in frame.columns else "prob_down"


def _sample_to_csv_row(sample: GradCamSample) -> dict[str, Any]:
    """`GradCamSample`을 `samples.csv` row로 바꾼다."""

    prediction = sample.prediction_row
    return {
        "panel": sample.panel,
        "rank_in_panel": sample.rank_in_panel,
        "sample_id": sample.sample_id,
        "experiment_name": prediction.get("experiment_name"),
        "year": prediction.get("year"),
        "Date": prediction.get("Date"),
        "StockID": prediction.get("StockID"),
        "shard_index": prediction.get("shard_index"),
        "local_row": prediction.get("local_row"),
        "global_index": prediction.get("global_index"),
        "target_return_name": prediction.get("target_return_name"),
        "target_return": prediction.get("target_return"),
        "label": prediction.get("label"),
        "pred_class": prediction.get("pred_class"),
        "target_class_for_gradcam": sample.target_class,
        "prob_down": prediction.get("prob_down"),
        "prob_up": prediction.get("prob_up"),
        "mean_prob_down": prediction.get("mean_prob_down"),
        "mean_prob_up": prediction.get("mean_prob_up"),
        "confidence": sample.confidence,
        "correct": prediction.get("correct"),
        "checkpoint_path": prediction.get("checkpoint_path"),
    }
