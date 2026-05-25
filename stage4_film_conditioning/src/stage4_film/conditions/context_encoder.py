"""Shared numeric market-context encoder for Stage 4.

역할:
    4-I2에서 만든 normalized context vector를 concat/gating/FiLM model이 공통으로
    사용할 condition embedding으로 변환한다.

Tensor 규칙:
    context: `(batch_size, context_dim=8)`
    embedding: `(batch_size, context_embedding_dim=32)`

구조:
    `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`

주의:
    이 encoder는 chart image를 보지 않는다. Chart image는 Stage 2 Stock_CNN
    branch가 처리하고, 이 encoder는 시장 맥락 numeric vector만 처리한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from stage4_film.config import get_stage4_model_config


@dataclass(frozen=True)
class ContextEncoderSpec:
    """Context MLP 구조 설정."""

    context_dim: int
    embedding_dim: int
    hidden_dim: int
    dropout: float

    def as_dict(self) -> dict[str, Any]:
        """JSON 저장용 dictionary로 변환한다."""

        return {
            "context_dim": int(self.context_dim),
            "embedding_dim": int(self.embedding_dim),
            "hidden_dim": int(self.hidden_dim),
            "dropout": float(self.dropout),
            "architecture": [
                f"Linear({self.context_dim}, {self.hidden_dim})",
                "ReLU",
                f"Dropout({self.dropout})",
                f"Linear({self.hidden_dim}, {self.embedding_dim})",
                "ReLU",
            ],
        }


class ContextEncoder(nn.Module):
    """Structured numeric context vector를 condition embedding으로 바꾸는 MLP."""

    def __init__(self, spec: ContextEncoderSpec) -> None:
        super().__init__()
        if spec.context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if spec.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if spec.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if not 0.0 <= float(spec.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.spec = spec
        self.net = nn.Sequential(
            nn.Linear(spec.context_dim, spec.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(spec.dropout)),
            nn.Linear(spec.hidden_dim, spec.embedding_dim),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Context tensor를 condition embedding으로 변환한다."""

        if context.ndim != 2:
            raise ValueError(
                f"context must have shape (batch, {self.spec.context_dim}); "
                f"got {tuple(context.shape)}"
            )
        if context.shape[1] != self.spec.context_dim:
            raise ValueError(
                f"context feature dimension mismatch: got {context.shape[1]}, "
                f"expected {self.spec.context_dim}"
            )
        return self.net(context.float())

    def forward_with_shapes(self, context: torch.Tensor) -> dict[str, tuple[int, ...]]:
        """중간 tensor shape를 반환하는 debug helper."""

        self.eval()
        shapes: dict[str, tuple[int, ...]] = {"context": tuple(context.shape)}
        with torch.no_grad():
            hidden = self.net[0](context.float())
            shapes["linear1"] = tuple(hidden.shape)
            hidden = self.net[1](hidden)
            shapes["relu1"] = tuple(hidden.shape)
            hidden = self.net[2](hidden)
            shapes["dropout"] = tuple(hidden.shape)
            hidden = self.net[3](hidden)
            shapes["linear2"] = tuple(hidden.shape)
            hidden = self.net[4](hidden)
            shapes["embedding"] = tuple(hidden.shape)
        return shapes

    def _init_weights(self) -> None:
        """Linear layer를 Xavier uniform으로 초기화한다."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def build_context_encoder_from_config(config: Mapping[str, Any]) -> ContextEncoder:
    """Stage 4 config로 shared context encoder를 만든다."""

    stage4_model = get_stage4_model_config(config)
    spec = ContextEncoderSpec(
        context_dim=int(stage4_model.get("context_dim", 8)),
        embedding_dim=int(stage4_model.get("context_embedding_dim", 32)),
        hidden_dim=int(stage4_model.get("context_encoder_hidden_dim", 32)),
        dropout=float(stage4_model.get("context_encoder_dropout", 0.10)),
    )
    return ContextEncoder(spec)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Model parameter 수를 센다."""

    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)
