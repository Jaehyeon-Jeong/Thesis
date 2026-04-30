"""Stock_CNN-style I20 baseline for Stage 1 Re-image reproduction.

Reference implementation:
    lich99/Stock_CNN, `models/baseline.py`
    commit: 415e2acf2a5013afca67e383acd3edc61fced840

Paper source:
    Jiang, Kelly, and Xiu, Re-Imagining Price Trends.
    Local summary maps CNN architecture/training details to pp. 12-21 and
    Figure 7 to p. 18.

Tensor convention:
    images: (batch_size, 1, height=64, width=60)
    logits: (batch_size, 2)

Important mismatch:
    The local paper summary emphasizes first-layer vertical dilation, while the
    checked GitHub I20 baseline applies dilation=(2, 1) to all three conv
    layers. Stage 1 follows the GitHub model core by design.
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
    """I20 baseline CNN matching the checked `lich99/Stock_CNN` implementation.

    The forward pass intentionally returns logits and does not apply softmax.
    Evaluation code will convert logits to class probabilities later.
    """

    def __init__(self) -> None:
        super().__init__()
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
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(46_080, 2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits.

        Input:
            x: image tensor with shape `(batch_size, 1, 64, 60)` or any shape
               that can be reshaped to that GitHub-compatible convention.

        Output:
            logits: `(batch_size, 2)`, where class index 1 is the Up class.
        """

        x = x.reshape(-1, 1, 64, 60)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1, 46_080)
        x = self.fc1(x)
        return x

    def forward_with_shapes(self, x: torch.Tensor) -> dict[str, tuple[int, ...]]:
        """Run a no-grad forward pass and return intermediate tensor shapes."""

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
        """Return named convolution layers planned for Stage 1 Grad-CAM."""

        return {
            "layer1_conv": self.layer1[0],
            "layer2_conv": self.layer2[0],
            "layer3_conv": self.layer3[0],
        }


def build_stock_cnn_i20_from_config(config: Mapping[str, Any]) -> StockCNNI20:
    """Build the Stage 1 I20 model after checking the model config name."""

    model_config = get_config_section(config, "model")
    model_name = str(model_config.get("name"))
    if model_name != "stock_cnn_i20":
        raise ValueError(f"Unsupported model.name for Stage 1 I20 baseline: {model_name}")
    return StockCNNI20()


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""

    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)
