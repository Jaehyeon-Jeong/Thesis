"""Stage 3 Stock_CNN + Linear adapter model.

핵심 위치:
    Stage 3 Linear는 CNN block 안에 들어가지 않는다. Stage 2와 동일한
    `Conv -> BatchNorm -> LeakyReLU -> MaxPool` block들을 모두 통과한 뒤,
    feature map을 flatten한 vector에 붙는다.

Tensor 흐름:
    image: `(batch_size, 1, height, width)`
    CNN feature map: image window별 `(batch_size, channels, h, w)`
    flatten: `(batch_size, flatten_dim)`
    adapter: `(batch_size, adapter_dim)`
    logits: `(batch_size, 2)`
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from stage2_btc.config import get_config_section, get_model_config
from stage2_btc.models.stock_cnn import (
    CnnVariantSpec,
    StockCNN,
    VARIANT_SPECS,
    count_parameters,
)
from stage3_linear.config import (
    get_linear_adapter_config,
    stage3_expected_parameter_count,
)


class LinearAdapterStockCNN(nn.Module):
    """Stage 2 Stock_CNN core 뒤에 Linear adapter를 추가한 model.

    구조:
        1. Stage 2 `StockCNN`과 같은 CNN blocks를 만든다.
        2. Stage 2의 기존 `fc1 = Dropout + Linear(flatten_dim, 2)`는 사용하지 않는다.
        3. 대신 `Dropout -> Linear(flatten_dim, adapter_dim, bias=False)
           -> Linear(adapter_dim, 2, bias=False)`를 사용한다.

    왜 block 안이 아닌가:
        block 내부에 들어가면 channel-wise modulation에 가까워지고 Stage 4 FiLM과
        섞인다. Stage 3는 단순 dense adapter 비교이므로 flatten 뒤에 둔다.
    """

    def __init__(
        self,
        spec: CnnVariantSpec,
        adapter_dim: int = 128,
        dropout: float = 0.5,
        adapter_bias: bool = False,
        classifier_bias: bool = False,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.adapter_dim = int(adapter_dim)

        # Stage 2 model을 한 번 만든 뒤, 그 CNN blocks만 재사용한다. 이렇게 하면
        # Conv/BN/ReLU/Pool core가 Stage 2 구현과 어긋나는 위험을 줄일 수 있다.
        base_model = StockCNN(spec=spec, dropout=dropout)
        self.layers = base_model.layers

        # Stage 3의 유일한 model change다. `bias=False`는 additive shift를 제거하기
        # 위한 첫 비교 설정이며, FiLM의 gamma/beta conditioning과는 다르다.
        self.fc1 = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(spec.expected_flatten_dim, self.adapter_dim, bias=bool(adapter_bias)),
            nn.Linear(self.adapter_dim, 2, bias=bool(classifier_bias)),
        )
        self.softmax = nn.Softmax(dim=1)

        actual_flatten_dim = self._infer_flatten_dim()
        if actual_flatten_dim != spec.expected_flatten_dim:
            raise ValueError(
                f"{spec.name} Stage 3 flatten mismatch: actual={actual_flatten_dim}, "
                f"expected={spec.expected_flatten_dim}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 chart image batch를 `[down_logit, up_logit]`으로 변환한다."""

        hidden = x.reshape(-1, *self.spec.input_shape)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = hidden.reshape(hidden.shape[0], -1)
        logits = self.fc1(hidden)
        return logits

    def forward_with_shapes(self, x: torch.Tensor) -> dict[str, tuple[int, ...]]:
        """중간 tensor shape를 확인하기 위한 debug helper."""

        self.eval()
        with torch.no_grad():
            hidden = x.reshape(-1, *self.spec.input_shape)
            shapes: dict[str, tuple[int, ...]] = {"input": tuple(hidden.shape)}
            for index, layer in enumerate(self.layers, start=1):
                hidden = layer(hidden)
                shapes[f"layer{index}"] = tuple(hidden.shape)
            hidden = hidden.reshape(hidden.shape[0], -1)
            shapes["flatten"] = tuple(hidden.shape)
            adapter_output = self.fc1[1](self.fc1[0](hidden))
            shapes["linear_adapter"] = tuple(adapter_output.shape)
            logits = self.fc1[2](adapter_output)
            shapes["logits"] = tuple(logits.shape)
        return shapes

    def gradcam_target_layers(self) -> dict[str, nn.Module]:
        """Grad-CAM hook을 걸 Conv2d layer들을 반환한다."""

        return {
            f"layer{index}_conv": layer[0]
            for index, layer in enumerate(self.layers, start=1)
        }

    def _infer_flatten_dim(self) -> int:
        """dummy tensor로 adapter 입력 길이를 확인한다."""

        with torch.no_grad():
            dummy = torch.zeros(1, *self.spec.input_shape)
            hidden = dummy
            for layer in self.layers:
                hidden = layer(hidden)
            return int(hidden.reshape(1, -1).shape[1])


def build_linear_stock_cnn_for_window(
    config: Mapping[str, Any],
    image_window: int,
) -> LinearAdapterStockCNN:
    """config와 image window로 Stage 3 Linear adapter model을 만든다."""

    window = int(image_window)
    if window not in VARIANT_SPECS:
        raise KeyError(f"Unsupported image_window for Stage 3 Linear adapter: {window}")

    model_config = get_config_section(config, "model")
    selected_model_config = get_model_config(config, window)
    adapter_config = get_linear_adapter_config(config)
    spec = VARIANT_SPECS[window]
    if str(selected_model_config.get("name")) != spec.name:
        raise ValueError(
            f"Config/model mismatch for I{window}: "
            f"config={selected_model_config.get('name')}, implementation={spec.name}"
        )

    model = LinearAdapterStockCNN(
        spec=spec,
        adapter_dim=int(adapter_config.get("adapter_dim", 128)),
        dropout=float(model_config.get("dropout", 0.5)),
        adapter_bias=bool(adapter_config.get("adapter_bias", False)),
        classifier_bias=bool(adapter_config.get("classifier_bias", False)),
    )
    parameter_count = count_parameters(model)
    expected = stage3_expected_parameter_count(config, window)
    if parameter_count != expected:
        raise ValueError(
            f"{spec.name} Stage 3 parameter mismatch: actual={parameter_count}, "
            f"expected={expected}"
        )
    return model

