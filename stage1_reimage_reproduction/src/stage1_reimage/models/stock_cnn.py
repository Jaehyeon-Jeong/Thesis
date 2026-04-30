"""1단계 Re-image 재현용 Stock_CNN-style I20 baseline.

참고 구현:
    lich99/Stock_CNN, `models/baseline.py`
    commit: 415e2acf2a5013afca67e383acd3edc61fced840

논문 근거:
    Jiang, Kelly, and Xiu, Re-Imagining Price Trends.
    local summary는 CNN architecture/training detail을 pp. 12-21,
    Figure 7을 p. 18로 mapping한다.

Tensor 규칙:
    images: (batch_size, 1, height=64, width=60)
    logits: (batch_size, 2)

중요한 mismatch:
    local paper summary는 first-layer vertical dilation을 강조하지만, 확인한
    GitHub I20 baseline은 dilation=(2, 1)을 세 conv layer 모두에 적용한다.
    1단계는 의도적으로 GitHub model core를 따른다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from stage1_reimage.config import get_config_section

EXPECTED_I20_PARAMETER_COUNT = 708_866
EXPECTED_I20_SHAPES: dict[str, tuple[int, ...]] = {
    "input": (2, 1, 64, 60),
    "layer1": (2, 64, 13, 60),
    "layer2": (2, 128, 5, 60),
    "layer3": (2, 256, 3, 60),
    "flatten": (2, 46_080),
    "logits": (2, 2),
}


class StockCNNI20(nn.Module):
    """확인한 `lich99/Stock_CNN` 구현에 맞춘 I20 baseline CNN.

    forward pass는 의도적으로 logits를 반환하고 softmax를 적용하지 않는다.
    class probability 변환은 evaluation code에서 한다.

    모델 읽는 법:
        입력 image batch: `(batch_size, 1, 64, 60)`.
        layer1 이후: `(batch_size, 64, 13, 60)`.
        layer2 이후: `(batch_size, 128, 5, 60)`.
        layer3 이후: `(batch_size, 256, 3, 60)`.
        flatten 이후: `(batch_size, 46080)`.
        출력 logits: `(batch_size, 2)`.
    """

    def __init__(self) -> None:
        super().__init__()

        # 1번 block은 grayscale image를 받는다. GitHub I20 baseline은 vertical
        # stride/dilation으로 height 방향을 줄이면서 60-pixel time axis width는 유지한다.
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                1,
                64,
                kernel_size=(5, 3),
                stride=(3, 1),
                dilation=(2, 1),
                padding=(12, 1),
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )

        # 2번 block은 64개 feature channel을 받아 128개 channel을 학습한다. output도
        # width 60을 유지하므로 temporal position alignment가 유지된다.
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=(5, 3),
                stride=(3, 1),
                dilation=(2, 1),
                padding=(12, 1),
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )

        # 3번 block은 128 channel을 받아 256 channel을 만든다. 이 block 이후 spatial
        # tensor는 `(batch, 256, 3, 60)`이다.
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                kernel_size=(5, 3),
                stride=(3, 1),
                dilation=(2, 1),
                padding=(12, 1),
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )

        # classifier는 마지막 feature map의 모든 값을 본다. `256 * 3 * 60 = 46080`
        # 이므로 linear layer input size는 46080이다.
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(46_080, 2),
        )

        # reference compatibility를 위해 남겨둔다. `CrossEntropyLoss`는 raw logits를
        # 기대하므로 `forward()` 안에서는 이 softmax를 호출하지 않는다.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """class logits를 반환한다.

        입력:
            x: `(batch_size, 1, 64, 60)` image tensor, 또는 그 shape로 reshape
               가능한 tensor.

        출력:
            logits: `(batch_size, 2)`. class index 1이 Up class다.
        """

        # batch가 CNN input convention을 갖도록 보장한다:
        # `(batch_size, channel=1, height=64, width=60)`.
        x = x.reshape(-1, 1, 64, 60)

        # 각 CNN block은 pixel을 점점 더 추상적인 chart-pattern feature map으로 바꾼다.
        # tensor는 계속 batch dimension을 첫 번째 축으로 유지한다.
        x = self.layer1(x)  # `(batch_size, 64, 13, 60)`
        x = self.layer2(x)  # `(batch_size, 128, 5, 60)`
        x = self.layer3(x)  # `(batch_size, 256, 3, 60)`

        # channel/height/width의 모든 값을 image 하나당 vector 하나로 flatten한다.
        # 그래야 마지막 linear classifier가 두 class score를 만들 수 있다.
        x = x.reshape(-1, 46_080)
        x = self.fc1(x)  # `(batch_size, 2)` = `[down_logit, up_logit]`
        return x

    def forward_with_shapes(self, x: torch.Tensor) -> dict[str, tuple[int, ...]]:
        """gradient 없이 forward pass를 실행하고 중간 tensor shape를 반환한다."""

        self.eval()
        with torch.no_grad():
            shapes: dict[str, tuple[int, ...]] = {"input": tuple(x.reshape(-1, 1, 64, 60).shape)}
            hidden = x.reshape(-1, 1, 64, 60)
            hidden = self.layer1(hidden)
            shapes["layer1"] = tuple(hidden.shape)
            hidden = self.layer2(hidden)
            shapes["layer2"] = tuple(hidden.shape)
            hidden = self.layer3(hidden)
            shapes["layer3"] = tuple(hidden.shape)
            hidden = hidden.reshape(-1, 46_080)
            shapes["flatten"] = tuple(hidden.shape)
            logits = self.fc1(hidden)
            shapes["logits"] = tuple(logits.shape)
        return shapes

    def gradcam_target_layers(self) -> dict[str, nn.Module]:
        """1단계 Grad-CAM에서 사용할 convolution layer를 이름과 함께 반환한다.

        Grad-CAM은 전체 sequential block이 아니라 이 Conv2d module에 hook을 건다.
        그래야 뒤의 pooling이 spatial map을 바꾸기 전 convolution activation과
        gradient를 수집할 수 있다.
        """

        return {
            "layer1_conv": self.layer1[0],
            "layer2_conv": self.layer2[0],
            "layer3_conv": self.layer3[0],
        }


def build_stock_cnn_i20_from_config(config: Mapping[str, Any]) -> StockCNNI20:
    """model config name을 확인한 뒤 1단계 I20 model을 만든다.

    이 함수는 guardrail이다. config가 다른 model을 요청하면, 잘못된 network를
    조용히 학습하지 말고 명확히 실패해야 한다.
    """

    model_config = get_config_section(config, "model")
    model_name = str(model_config.get("name"))
    if model_name != "stock_cnn_i20":
        raise ValueError(f"Unsupported model.name for Stage 1 I20 baseline: {model_name}")
    return StockCNNI20()


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """model parameter 수를 센다.

    smoke check에서 local model이 기대한 GitHub-style I20 parameter count와
    일치하는지 확인할 때 사용한다.
    """

    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)
