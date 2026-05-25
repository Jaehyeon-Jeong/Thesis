"""Stage 4 context-conditioned wrappers around the Stage 2 Stock_CNN backbone.

4-I4 implements the first Stage 4 fusion baseline:

    chart image -> Stage 2 Stock_CNN convolution blocks -> flatten image feature
    context vector -> Stage 4 ContextEncoder -> context embedding
    concat(image feature, context embedding) -> linear classifier -> logits

This keeps the Stage 2 visual backbone unchanged. The only change is the
classifier input: instead of using the flattened CNN feature alone, the model
appends a learned market-context embedding immediately before the final logits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from stage2_btc.config import get_config_section, get_model_config
from stage2_btc.models.stock_cnn import (
    StockCNN,
    VARIANT_SPECS,
    CnnVariantSpec,
    count_parameters,
)
from stage4_film.conditions import ContextEncoder, build_context_encoder_from_config
from stage4_film.config import get_stage4_model_config


class ConcatContextStockCNN(nn.Module):
    """Stage 2 Stock_CNN backbone plus context embedding concatenation.

    Tensor rules:
        image: `(batch_size, 1, height, width)` for the selected image window.
        context: `(batch_size, context_dim)`.
        context embedding: `(batch_size, context_embedding_dim)`.
        logits: `(batch_size, 2)`.

    For the primary Stage 4 setting `I60/R20/ohlc_ma_vb`, the CNN flatten
    feature is `(B, 184320)` and the context embedding is `(B, 32)`, so the
    final classifier receives `(B, 184352)`.
    """

    def __init__(
        self,
        spec: CnnVariantSpec,
        context_encoder: ContextEncoder,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.context_encoder = context_encoder

        # Reuse the exact Stage 2 convolution/BN/ReLU/pool blocks. The Stage 2
        # classifier is intentionally not registered because Stage 4 replaces
        # it with a concat-aware classifier.
        stage2_backbone = StockCNN(spec=spec, dropout=dropout)
        self.layers = stage2_backbone.layers

        concat_dim = int(spec.expected_flatten_dim) + int(context_encoder.spec.embedding_dim)
        self.fc1 = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(concat_dim, 2),
        )
        self.softmax = nn.Softmax(dim=1)

        actual_flatten_dim = self._infer_flatten_dim()
        if actual_flatten_dim != spec.expected_flatten_dim:
            raise ValueError(
                f"{spec.name} flatten mismatch: actual={actual_flatten_dim}, "
                f"expected={spec.expected_flatten_dim}"
            )
        if self.fc1[1].in_features != concat_dim:
            raise ValueError(
                "Concat classifier input mismatch: "
                f"actual={self.fc1[1].in_features}, expected={concat_dim}"
            )

    @property
    def concat_feature_dim(self) -> int:
        """Final classifier input dimension."""

        return int(self.fc1[1].in_features)

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return class logits for chart image plus market context."""

        image_feature = self.extract_image_feature(image)
        context_embedding = self.context_encoder(context)
        if image_feature.shape[0] != context_embedding.shape[0]:
            raise ValueError(
                "image and context batch sizes must match: "
                f"{image_feature.shape[0]} != {context_embedding.shape[0]}"
            )
        fused = torch.cat([image_feature, context_embedding], dim=1)
        return self.fc1(fused)

    def extract_image_feature(self, image: torch.Tensor) -> torch.Tensor:
        """Run the Stage 2 CNN blocks and return the flattened image feature."""

        hidden = image.reshape(-1, *self.spec.input_shape)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden.reshape(hidden.shape[0], -1)

    def forward_with_shapes(
        self,
        image: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, tuple[int, ...]]:
        """Run a no-grad forward pass and return intermediate tensor shapes."""

        self.eval()
        with torch.no_grad():
            hidden = image.reshape(-1, *self.spec.input_shape)
            shapes: dict[str, tuple[int, ...]] = {
                "image": tuple(hidden.shape),
                "context": tuple(context.shape),
            }
            for index, layer in enumerate(self.layers, start=1):
                hidden = layer(hidden)
                shapes[f"layer{index}"] = tuple(hidden.shape)
            image_feature = hidden.reshape(hidden.shape[0], -1)
            shapes["flatten"] = tuple(image_feature.shape)
            context_embedding = self.context_encoder(context)
            shapes["context_embedding"] = tuple(context_embedding.shape)
            fused = torch.cat([image_feature, context_embedding], dim=1)
            shapes["concat_feature"] = tuple(fused.shape)
            logits = self.fc1(fused)
            shapes["logits"] = tuple(logits.shape)
        return shapes

    def gradcam_target_layers(self) -> dict[str, nn.Module]:
        """Return Conv2d layers for Grad-CAM hooks."""

        return {
            f"layer{index}_conv": layer[0]
            for index, layer in enumerate(self.layers, start=1)
        }

    def _infer_flatten_dim(self) -> int:
        """Infer CNN flatten dimension with a dummy tensor."""

        with torch.no_grad():
            dummy = torch.zeros(1, *self.spec.input_shape)
            return int(self.extract_image_feature(dummy).shape[1])


def build_concat_context_stock_cnn_for_window(
    config: Mapping[str, Any],
    image_window: int,
) -> ConcatContextStockCNN:
    """Build the Stage 4 concat-context model for an image window."""

    window = int(image_window)
    if window not in VARIANT_SPECS:
        raise KeyError(f"Unsupported image_window for Stage 4 StockCNN: {window}")

    model_config = get_config_section(config, "model")
    selected_model_config = get_model_config(config, window)
    spec = VARIANT_SPECS[window]
    if str(selected_model_config.get("name")) != spec.name:
        raise ValueError(
            f"Config/model mismatch for I{window}: "
            f"config={selected_model_config.get('name')}, implementation={spec.name}"
        )

    context_encoder = build_context_encoder_from_config(config)
    model = ConcatContextStockCNN(
        spec=spec,
        context_encoder=context_encoder,
        dropout=float(model_config.get("dropout", 0.5)),
    )
    actual = count_parameters(model)
    expected = expected_concat_context_parameter_count(config, window)
    if actual != expected:
        raise ValueError(
            f"Stage 4 concat parameter mismatch: actual={actual}, expected={expected}"
        )
    return model


def expected_concat_context_parameter_count(
    config: Mapping[str, Any],
    image_window: int,
) -> int:
    """Expected parameter count for the concat context model.

    The Stage 2 classifier is removed, so the count is:
        Stage2 CNN blocks
        + context encoder
        + Linear(flatten_dim + context_embedding_dim, 2)
    """

    window = int(image_window)
    if window not in VARIANT_SPECS:
        raise KeyError(f"Unsupported image_window for Stage 4 StockCNN: {window}")

    spec = VARIANT_SPECS[window]
    stage2_classifier_params = int(spec.expected_flatten_dim) * 2 + 2
    stage2_feature_params = int(spec.expected_num_params) - stage2_classifier_params

    stage4_model = get_stage4_model_config(config)
    embedding_dim = int(stage4_model.get("context_embedding_dim", 32))
    hidden_dim = int(stage4_model.get("context_encoder_hidden_dim", 32))
    context_dim = int(stage4_model.get("context_dim", 8))
    context_encoder_params = (
        context_dim * hidden_dim
        + hidden_dim
        + hidden_dim * embedding_dim
        + embedding_dim
    )
    concat_classifier_params = (int(spec.expected_flatten_dim) + embedding_dim) * 2 + 2
    return int(stage2_feature_params + context_encoder_params + concat_classifier_params)
