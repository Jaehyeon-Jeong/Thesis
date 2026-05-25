"""Stage 4 FiLM-conditioned Stock_CNN models.

4-I7 attaches the FiLM generator from 4-I6 to the Stage 2 Stock_CNN visual
backbone. The visual CNN blocks stay structurally equivalent to Stage 2, but
each block applies condition-generated channel-wise modulation after BatchNorm
and before LeakyReLU:

    Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d

Two modes are supported:
    film_gamma: `F' = gamma * F`
    film_full: `F' = gamma * F + beta`
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
from stage4_film.conditions import (
    ContextEncoder,
    FilmParameterGenerator,
    build_context_encoder_from_config,
    build_film_generator_for_window,
    expected_film_generator_parameter_count,
)
from stage4_film.config import get_stage4_model_config, validate_context_method
from stage4_film.layers import FeatureWiseAffineModulation
from stage4_film.models.context_stock_cnn import _expected_context_encoder_parameter_count


class FilmContextStockCNN(nn.Module):
    """Stage 2 Stock_CNN backbone with block-wise context-generated FiLM.

    Tensor rules:
        image: `(batch_size, 1, height, width)` for the selected image window.
        context: `(batch_size, context_dim)`.
        context embedding: `(batch_size, context_embedding_dim)`.
        gamma/beta: one `(batch_size, channels)` tensor per CNN block.
        logits: `(batch_size, 2)`.

    For the primary Stage 4 setting `I60/R20/ohlc_ma_vb`, the model emits FiLM
    parameters for channels `[64, 128, 256, 512]`.
    """

    def __init__(
        self,
        spec: CnnVariantSpec,
        context_encoder: ContextEncoder,
        film_generator: FilmParameterGenerator,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        mode = validate_context_method(film_generator.spec.mode)
        if mode not in {"film_gamma", "film_full"}:
            raise ValueError(f"Unsupported FiLM context mode: {mode}")
        if tuple(film_generator.spec.block_channels) != tuple(spec.channels):
            raise ValueError(
                "FiLM generator channels must match CNN block channels: "
                f"{film_generator.spec.block_channels} != {spec.channels}"
            )
        if int(film_generator.spec.context_embedding_dim) != int(context_encoder.spec.embedding_dim):
            raise ValueError(
                "FiLM generator embedding dim must match context encoder output dim: "
                f"{film_generator.spec.context_embedding_dim} != "
                f"{context_encoder.spec.embedding_dim}"
            )

        self.spec = spec
        self.context_encoder = context_encoder
        self.film_generator = film_generator
        self.mode = mode

        stage2_backbone = StockCNN(spec=spec, dropout=dropout)
        self.layers = stage2_backbone.layers
        self.film_layers = nn.ModuleList(
            FeatureWiseAffineModulation() for _channels in spec.channels
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(spec.expected_flatten_dim), 2),
        )
        self.softmax = nn.Softmax(dim=1)

        actual_flatten_dim = self._infer_flatten_dim()
        if actual_flatten_dim != spec.expected_flatten_dim:
            raise ValueError(
                f"{spec.name} flatten mismatch: actual={actual_flatten_dim}, "
                f"expected={spec.expected_flatten_dim}"
            )

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return class logits after block-wise FiLM conditioning."""

        context_embedding = self.context_encoder(context)
        film_parameters = self.film_generator(context_embedding)
        hidden = self._forward_film_blocks(image, film_parameters)
        return self.fc1(hidden.reshape(hidden.shape[0], -1))

    def forward_with_shapes(
        self,
        image: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, tuple[int, ...]]:
        """Run a no-grad forward pass and return intermediate tensor shapes."""

        self.eval()
        with torch.no_grad():
            context_embedding = self.context_encoder(context)
            film_parameters = self.film_generator(context_embedding)
            hidden = image.reshape(-1, *self.spec.input_shape)
            shapes: dict[str, tuple[int, ...]] = {
                "image": tuple(hidden.shape),
                "context": tuple(context.shape),
                "context_embedding": tuple(context_embedding.shape),
            }
            for index, (layer, film_layer, block_params) in enumerate(
                zip(self.layers, self.film_layers, film_parameters, strict=True),
                start=1,
            ):
                hidden = layer[0](hidden)
                shapes[f"layer{index}_conv"] = tuple(hidden.shape)
                hidden = layer[1](hidden)
                shapes[f"layer{index}_bn"] = tuple(hidden.shape)
                gamma = block_params["gamma"]
                beta = block_params["beta"]
                if gamma is None:
                    raise RuntimeError(f"FiLM block {index} missing gamma")
                shapes[f"gamma{index}"] = tuple(gamma.shape)
                if beta is not None:
                    shapes[f"beta{index}"] = tuple(beta.shape)
                hidden = film_layer(hidden, gamma, beta)
                shapes[f"layer{index}_film"] = tuple(hidden.shape)
                hidden = layer[2](hidden)
                hidden = layer[3](hidden)
                shapes[f"layer{index}"] = tuple(hidden.shape)
            flattened = hidden.reshape(hidden.shape[0], -1)
            shapes["flatten"] = tuple(flattened.shape)
            logits = self.fc1(flattened)
            shapes["logits"] = tuple(logits.shape)
        return shapes

    def forward_with_modulation_values(
        self,
        image: torch.Tensor,
        context: torch.Tensor,
        *,
        keep_feature_maps: bool = False,
    ) -> dict[str, Any]:
        """Return logits plus FiLM parameters for interpretation/export."""

        context_embedding = self.context_encoder(context)
        film_parameters = self.film_generator(context_embedding)
        hidden = image.reshape(-1, *self.spec.input_shape)
        block_summaries: list[dict[str, torch.Tensor | bool | int | None]] = []
        pre_film_maps: list[torch.Tensor] = []
        post_film_maps: list[torch.Tensor] = []
        pooled_maps: list[torch.Tensor] = []
        for index, (layer, film_layer, block_params) in enumerate(
            zip(self.layers, self.film_layers, film_parameters, strict=True),
            start=1,
        ):
            hidden = layer[0](hidden)
            hidden = layer[1](hidden)
            pre_film = hidden
            gamma = block_params["gamma"]
            beta = block_params["beta"]
            if gamma is None:
                raise RuntimeError(f"FiLM block {index} missing gamma")
            post_film = film_layer(pre_film, gamma, beta)
            identity_ok = torch.allclose(post_film, pre_film)
            if keep_feature_maps:
                pre_film_maps.append(pre_film.detach().clone())
                post_film_maps.append(post_film.detach().clone())
            hidden = layer[2](post_film)
            hidden = layer[3](hidden)
            block_summaries.append(
                {
                    "block": index,
                    "gamma": gamma,
                    "beta": beta,
                    "delta_gamma": block_params["delta_gamma"],
                    "identity_at_initialization": identity_ok,
                }
            )
            if keep_feature_maps:
                pooled_maps.append(hidden.detach().clone())
        logits = self.fc1(hidden.reshape(hidden.shape[0], -1))
        result: dict[str, Any] = {
            "logits": logits,
            "context_embedding": context_embedding,
            "film_parameters": block_summaries,
        }
        if keep_feature_maps:
            result["pre_film_feature_maps"] = pre_film_maps
            result["post_film_feature_maps"] = post_film_maps
            result["pooled_feature_maps"] = pooled_maps
        return result

    def gradcam_target_layers(self) -> dict[str, nn.Module]:
        """Return Conv2d layers for Grad-CAM hooks."""

        return {
            f"layer{index}_conv": layer[0]
            for index, layer in enumerate(self.layers, start=1)
        }

    def film_target_layers(self) -> dict[str, nn.Module]:
        """Return block-specific FiLM modules for post-FiLM hooks."""

        return {
            f"layer{index}_film": layer
            for index, layer in enumerate(self.film_layers, start=1)
        }

    def _forward_film_blocks(
        self,
        image: torch.Tensor,
        film_parameters: list[dict[str, torch.Tensor | None]],
    ) -> torch.Tensor:
        """Run CNN blocks with FiLM inserted after BatchNorm."""

        hidden = image.reshape(-1, *self.spec.input_shape)
        if len(film_parameters) != len(self.layers):
            raise RuntimeError(
                f"FiLM block count mismatch: got {len(film_parameters)}, "
                f"expected {len(self.layers)}"
            )
        for layer, film_layer, block_params in zip(
            self.layers,
            self.film_layers,
            film_parameters,
            strict=True,
        ):
            hidden = layer[0](hidden)
            hidden = layer[1](hidden)
            gamma = block_params["gamma"]
            beta = block_params["beta"]
            if gamma is None:
                raise RuntimeError("FiLM generator did not emit gamma.")
            hidden = film_layer(hidden, gamma, beta)
            hidden = layer[2](hidden)
            hidden = layer[3](hidden)
        return hidden

    def _infer_flatten_dim(self) -> int:
        """Infer FiLM CNN flatten dimension with dummy tensors."""

        with torch.no_grad():
            image = torch.zeros(1, *self.spec.input_shape)
            context = torch.zeros(1, int(self.context_encoder.spec.context_dim))
            context_embedding = self.context_encoder(context)
            film_parameters = self.film_generator(context_embedding)
            hidden = self._forward_film_blocks(image, film_parameters)
            return int(hidden.reshape(1, -1).shape[1])


def build_film_context_stock_cnn_for_window(
    config: Mapping[str, Any],
    image_window: int,
    mode: str,
) -> FilmContextStockCNN:
    """Build the Stage 4 FiLM-context model for an image window."""

    window = int(image_window)
    if window not in VARIANT_SPECS:
        raise KeyError(f"Unsupported image_window for Stage 4 FiLM StockCNN: {window}")

    canonical_mode = validate_context_method(mode)
    if canonical_mode not in {"film_gamma", "film_full"}:
        raise ValueError(f"Unsupported FiLM context mode: {canonical_mode}")

    model_config = get_config_section(config, "model")
    selected_model_config = get_model_config(config, window)
    spec = VARIANT_SPECS[window]
    if str(selected_model_config.get("name")) != spec.name:
        raise ValueError(
            f"Config/model mismatch for I{window}: "
            f"config={selected_model_config.get('name')}, implementation={spec.name}"
        )

    context_encoder = build_context_encoder_from_config(config)
    film_generator = build_film_generator_for_window(
        config,
        image_window=window,
        mode=canonical_mode,
    )
    model = FilmContextStockCNN(
        spec=spec,
        context_encoder=context_encoder,
        film_generator=film_generator,
        dropout=float(model_config.get("dropout", 0.5)),
    )
    actual = count_parameters(model)
    expected = expected_film_context_parameter_count(config, window, canonical_mode)
    if actual != expected:
        raise ValueError(
            f"Stage 4 FiLM parameter mismatch: actual={actual}, expected={expected}"
        )
    return model


def expected_film_context_parameter_count(
    config: Mapping[str, Any],
    image_window: int,
    mode: str,
) -> int:
    """Expected parameter count for a FiLM-context Stock_CNN model."""

    window = int(image_window)
    if window not in VARIANT_SPECS:
        raise KeyError(f"Unsupported image_window for Stage 4 FiLM StockCNN: {window}")

    canonical_mode = validate_context_method(mode)
    if canonical_mode not in {"film_gamma", "film_full"}:
        raise ValueError(f"Unsupported FiLM context mode: {canonical_mode}")

    spec = VARIANT_SPECS[window]
    embedding_dim = int(get_stage4_model_config(config).get("context_embedding_dim", 32))
    generator_params = expected_film_generator_parameter_count(
        context_embedding_dim=embedding_dim,
        block_channels=spec.channels,
        mode=canonical_mode,
    )
    return int(
        spec.expected_num_params
        + _expected_context_encoder_parameter_count(config)
        + generator_params
    )
