"""Stage 2 Stock_CNN-style model variants for BTC I5/I20/I60.

참고 구현:
    Stage 1 I20 model은 `lich99/Stock_CNN/models/baseline.py`의 I20 구현을 따른다.

논문/요약 근거:
    `자료조사/Re-image 요약.md` line 47:
    I5/I20/I60 model depth, channel sequence, flatten dimension, parameter count.

중요:
    I20은 Stage 1/GitHub식 core와 같은 shape/parameter count를 유지한다.
    I5/I60은 논문에 보고된 flatten dimension과 parameter count를 맞추는
    Stage-1/Stock_CNN-style variant다.

Tensor 규칙:
    입력: `(batch_size, 1, height, width)`.
    출력: `(batch_size, 2)` logits. class 0은 Down/non-positive, class 1은 Up.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from stage2_btc.config import get_config_section, get_model_config


@dataclass(frozen=True)
class CnnVariantSpec:
    """image window별 CNN 구조를 명시적으로 들고 있는 객체."""

    name: str
    input_shape: tuple[int, int, int]
    channels: tuple[int, ...]
    conv_stride: tuple[int, int]
    conv_dilation: tuple[int, int]
    conv_padding: tuple[int, int]
    pool_size: tuple[int, int]
    expected_flatten_dim: int
    expected_num_params: int


VARIANT_SPECS: dict[int, CnnVariantSpec] = {
    # I5는 보고된 flatten dim 15360을 맞추려면 height가 32 -> 16 -> 8로 내려가야 한다.
    # 따라서 I20 GitHub식 vertical conv stride 3을 그대로 쓰지 않고 stride 1을 쓴다.
    5: CnnVariantSpec(
        name="stock_cnn_i5",
        input_shape=(1, 32, 15),
        channels=(64, 128),
        conv_stride=(1, 1),
        conv_dilation=(1, 1),
        conv_padding=(2, 1),
        pool_size=(2, 1),
        expected_flatten_dim=15_360,
        expected_num_params=155_138,
    ),
    # I20은 Stage 1/GitHub implementation과 동일하게 세 conv layer 모두
    # dilation=(2,1), padding=(12,1), stride=(3,1)을 사용한다.
    20: CnnVariantSpec(
        name="stock_cnn_i20",
        input_shape=(1, 64, 60),
        channels=(64, 128, 256),
        conv_stride=(3, 1),
        conv_dilation=(2, 1),
        conv_padding=(12, 1),
        pool_size=(2, 1),
        expected_flatten_dim=46_080,
        expected_num_params=708_866,
    ),
    # I60은 I20의 Stock_CNN-style pattern을 확장하되, 보고된 flatten dim 184320을
    # 맞추기 위해 dilation=(3,1), padding=(12,1)을 모든 block에 적용한다.
    60: CnnVariantSpec(
        name="stock_cnn_i60",
        input_shape=(1, 96, 180),
        channels=(64, 128, 256, 512),
        conv_stride=(3, 1),
        conv_dilation=(3, 1),
        conv_padding=(12, 1),
        pool_size=(2, 1),
        expected_flatten_dim=184_320,
        expected_num_params=2_952_962,
    ),
}


class StockCNN(nn.Module):
    """BTC I5/I20/I60에 공통으로 쓰는 Stock_CNN-style classifier."""

    def __init__(self, spec: CnnVariantSpec, dropout: float = 0.5) -> None:
        super().__init__()
        self.spec = spec
        input_channels = spec.input_shape[0]
        blocks: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in spec.channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(5, 3),
                        stride=spec.conv_stride,
                        dilation=spec.conv_dilation,
                        padding=spec.conv_padding,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                    nn.MaxPool2d(spec.pool_size, stride=spec.pool_size),
                )
            )
            in_channels = out_channels
        self.layers = nn.ModuleList(blocks)
        self.fc1 = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(spec.expected_flatten_dim, 2),
        )
        self.softmax = nn.Softmax(dim=1)

        actual_flatten_dim = self._infer_flatten_dim()
        if actual_flatten_dim != spec.expected_flatten_dim:
            raise ValueError(
                f"{spec.name} flatten mismatch: actual={actual_flatten_dim}, "
                f"expected={spec.expected_flatten_dim}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 chart image batch를 class logits로 변환한다."""

        x = x.reshape(-1, *self.spec.input_shape)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc1(x)

    def forward_with_shapes(self, x: torch.Tensor) -> dict[str, tuple[int, ...]]:
        """중간 feature map shape를 확인하기 위한 debug helper."""

        self.eval()
        with torch.no_grad():
            hidden = x.reshape(-1, *self.spec.input_shape)
            shapes: dict[str, tuple[int, ...]] = {"input": tuple(hidden.shape)}
            for index, layer in enumerate(self.layers, start=1):
                hidden = layer(hidden)
                shapes[f"layer{index}"] = tuple(hidden.shape)
            hidden = hidden.reshape(hidden.shape[0], -1)
            shapes["flatten"] = tuple(hidden.shape)
            logits = self.fc1(hidden)
            shapes["logits"] = tuple(logits.shape)
        return shapes

    def gradcam_target_layers(self) -> dict[str, nn.Module]:
        """Grad-CAM hook을 걸 Conv2d layer들을 반환한다."""

        return {
            f"layer{index}_conv": layer[0]
            for index, layer in enumerate(self.layers, start=1)
        }

    def _infer_flatten_dim(self) -> int:
        """dummy tensor로 classifier 입력 길이를 확인한다."""

        with torch.no_grad():
            dummy = torch.zeros(1, *self.spec.input_shape)
            hidden = dummy
            for layer in self.layers:
                hidden = layer(hidden)
            return int(hidden.reshape(1, -1).shape[1])


def build_stock_cnn_for_window(
    config: Mapping[str, Any],
    image_window: int,
) -> StockCNN:
    """config와 image window로 Stage 2 CNN variant를 만든다."""

    window = int(image_window)
    if window not in VARIANT_SPECS:
        raise KeyError(f"Unsupported image_window for StockCNN: {window}")

    model_config = get_config_section(config, "model")
    selected_model_config = get_model_config(config, window)
    spec = VARIANT_SPECS[window]
    if str(selected_model_config.get("name")) != spec.name:
        raise ValueError(
            f"Config/model mismatch for I{window}: "
            f"config={selected_model_config.get('name')}, implementation={spec.name}"
        )

    model = StockCNN(spec=spec, dropout=float(model_config.get("dropout", 0.5)))
    parameter_count = count_parameters(model)
    if parameter_count != spec.expected_num_params:
        raise ValueError(
            f"{spec.name} parameter mismatch: actual={parameter_count}, "
            f"expected={spec.expected_num_params}"
        )
    return model


def initialize_model_weights(model: nn.Module) -> None:
    """Conv/Linear/BatchNorm parameter를 Stage 1과 같은 방식으로 초기화한다."""

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """model parameter 수를 반환한다."""

    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return int(sum(parameter.numel() for parameter in parameters))
