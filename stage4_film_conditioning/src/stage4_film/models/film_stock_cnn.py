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

BOUNDED_LAST_BLOCK_METHOD = "film_full_bounded_last_block"
UNCERTAINTY_GATED_LAST_BLOCK_METHOD = "film_full_uncertainty_gated_last_block"
CONFIDENCE_GATED_LAST_BLOCK_METHOD = "film_full_confidence_gated_last_block"


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


class BoundedLastBlockFilmContextStockCNN(nn.Module):
    """Stage 2 Stock_CNN with conservative final-block FiLM conditioning.

    This V7 variant keeps the early chart-feature extraction path unchanged and
    applies FiLM only after BatchNorm in the final CNN block:

        Conv2d -> BatchNorm2d -> bounded FiLM -> LeakyReLU -> MaxPool2d

    The modulation is residual/bounded:

        gamma = 1 + scale * tanh(raw_gamma)
        beta  =     scale * tanh(raw_beta)

    With zero-initialized heads, the model starts exactly from the visual
    backbone identity path (`gamma=1`, `beta=0`). The default small scale keeps
    context from overwhelming the strong Stage 2 visual baseline.
    """

    def __init__(
        self,
        spec: CnnVariantSpec,
        context_encoder: ContextEncoder,
        dropout: float = 0.5,
        modulation_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if modulation_scale <= 0:
            raise ValueError("modulation_scale must be positive.")

        self.spec = spec
        self.context_encoder = context_encoder
        self.mode = BOUNDED_LAST_BLOCK_METHOD
        self.modulation_scale = float(modulation_scale)
        self.final_block_index = len(spec.channels)
        self.final_channels = int(spec.channels[-1])

        stage2_backbone = StockCNN(spec=spec, dropout=dropout)
        self.layers = stage2_backbone.layers
        self.film_layer = FeatureWiseAffineModulation()
        embedding_dim = int(context_encoder.spec.embedding_dim)
        self.gamma_head = nn.Linear(embedding_dim, self.final_channels)
        self.beta_head = nn.Linear(embedding_dim, self.final_channels)
        self.fc1 = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(spec.expected_flatten_dim), 2),
        )
        self.softmax = nn.Softmax(dim=1)

        self._init_identity_modulation()
        actual_flatten_dim = self._infer_flatten_dim()
        if actual_flatten_dim != spec.expected_flatten_dim:
            raise ValueError(
                f"{spec.name} flatten mismatch: actual={actual_flatten_dim}, "
                f"expected={spec.expected_flatten_dim}"
            )

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return class logits after bounded final-block FiLM conditioning."""

        context_embedding = self.context_encoder(context)
        hidden, _parameters = self._forward_bounded_blocks(image, context_embedding)
        return self.fc1(hidden.reshape(hidden.shape[0], -1))

    def compute_bounded_parameters(
        self,
        context_embedding: torch.Tensor,
        *,
        modulation_gate: torch.Tensor | None = None,
        stage2_prob_up: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | float | int]:
        """Return raw and bounded FiLM parameters for the final CNN block."""

        if context_embedding.ndim != 2:
            raise ValueError(
                "context_embedding must have shape (batch, embedding_dim); "
                f"got {tuple(context_embedding.shape)}"
            )
        raw_gamma = self.gamma_head(context_embedding)
        raw_beta = self.beta_head(context_embedding)
        scale = self.modulation_scale
        gate = self._normalize_modulation_gate(
            modulation_gate,
            batch_size=context_embedding.shape[0],
            device=context_embedding.device,
            dtype=context_embedding.dtype,
        )
        gamma = 1.0 + gate * scale * torch.tanh(raw_gamma)
        beta = gate * scale * torch.tanh(raw_beta)
        result: dict[str, torch.Tensor | float | int] = {
            "block": self.final_block_index,
            "gamma": gamma,
            "beta": beta,
            "delta_gamma": gamma - 1.0,
            "raw_gamma": raw_gamma,
            "raw_beta": raw_beta,
            "modulation_scale": scale,
            "modulation_gate": gate,
        }
        if stage2_prob_up is not None:
            result["stage2_prob_up"] = self._normalize_modulation_gate(
                stage2_prob_up,
                batch_size=context_embedding.shape[0],
                device=context_embedding.device,
                dtype=context_embedding.dtype,
            )
        return result

    def _normalize_modulation_gate(
        self,
        gate: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a `(batch, 1)` gate tensor used to scale bounded FiLM."""

        if gate is None:
            return torch.ones(int(batch_size), 1, device=device, dtype=dtype)
        if gate.ndim == 1:
            gate = gate.reshape(-1, 1)
        if gate.ndim != 2 or gate.shape[0] != int(batch_size) or gate.shape[1] != 1:
            raise ValueError(
                "modulation_gate must have shape (batch, 1); "
                f"got {tuple(gate.shape)} for batch={batch_size}"
            )
        return gate.to(device=device, dtype=dtype)

    def _forward_bounded_blocks_with_parameters(
        self,
        image: torch.Tensor,
        parameters: dict[str, torch.Tensor | float | int],
    ) -> torch.Tensor:
        """Run CNN blocks with a precomputed final-block FiLM parameter set."""

        hidden = image.reshape(-1, *self.spec.input_shape)
        for index, layer in enumerate(self.layers, start=1):
            hidden = layer[0](hidden)
            hidden = layer[1](hidden)
            if index == self.final_block_index:
                gamma = parameters["gamma"]
                beta = parameters["beta"]
                if not isinstance(gamma, torch.Tensor) or not isinstance(beta, torch.Tensor):
                    raise RuntimeError("Bounded FiLM parameters must be tensors.")
                hidden = self.film_layer(hidden, gamma, beta)
            hidden = layer[2](hidden)
            hidden = layer[3](hidden)
        return hidden

    def _forward_visual_baseline_logits(self, image: torch.Tensor) -> torch.Tensor:
        """Run the unmodulated Stage 2 visual path with the current weights."""

        hidden = image.reshape(-1, *self.spec.input_shape)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.fc1(hidden.reshape(hidden.shape[0], -1))

    def _block_summary_from_parameters(
        self,
        parameters: dict[str, torch.Tensor | float | int],
    ) -> dict[str, torch.Tensor | bool | float | int | None]:
        """Build a compact final-block modulation record for export."""

        gamma = parameters["gamma"]
        beta = parameters["beta"]
        if not isinstance(gamma, torch.Tensor) or not isinstance(beta, torch.Tensor):
            raise RuntimeError("Bounded FiLM parameters must be tensors.")
        return {
            "block": self.final_block_index,
            "gamma": gamma,
            "beta": beta,
            "delta_gamma": parameters["delta_gamma"],
            "raw_gamma": parameters["raw_gamma"],
            "raw_beta": parameters["raw_beta"],
            "modulation_scale": parameters["modulation_scale"],
            "modulation_gate": parameters.get("modulation_gate"),
            "stage2_prob_up": parameters.get("stage2_prob_up"),
            "identity_at_initialization": bool(
                torch.allclose(gamma, torch.ones_like(gamma))
                and torch.allclose(beta, torch.zeros_like(beta))
            ),
        }

    def _shape_entries_for_parameters(
        self,
        shapes: dict[str, tuple[int, ...]],
        parameters: dict[str, torch.Tensor | float | int],
    ) -> None:
        """Add bounded FiLM parameter tensor shapes to a shapes dictionary."""

        block_index = self.final_block_index
        for name in (
            "raw_gamma",
            "raw_beta",
            "gamma",
            "beta",
            "modulation_gate",
            "stage2_prob_up",
        ):
            tensor = parameters.get(name)
            if isinstance(tensor, torch.Tensor):
                shapes[f"{name}{block_index}"] = tuple(tensor.shape)

    def forward_with_shapes(
        self,
        image: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, tuple[int, ...]]:
        """Run a no-grad forward pass and return intermediate tensor shapes."""

        self.eval()
        with torch.no_grad():
            context_embedding = self.context_encoder(context)
            parameters = self.compute_bounded_parameters(context_embedding)
            hidden = image.reshape(-1, *self.spec.input_shape)
            shapes: dict[str, tuple[int, ...]] = {
                "image": tuple(hidden.shape),
                "context": tuple(context.shape),
                "context_embedding": tuple(context_embedding.shape),
            }
            for index, layer in enumerate(self.layers, start=1):
                hidden = layer[0](hidden)
                shapes[f"layer{index}_conv"] = tuple(hidden.shape)
                hidden = layer[1](hidden)
                shapes[f"layer{index}_bn"] = tuple(hidden.shape)
                if index == self.final_block_index:
                    self._shape_entries_for_parameters(shapes, parameters)
                    gamma = parameters["gamma"]
                    beta = parameters["beta"]
                    if not isinstance(gamma, torch.Tensor) or not isinstance(beta, torch.Tensor):
                        raise RuntimeError("Bounded FiLM parameters must be tensors.")
                    hidden = self.film_layer(hidden, gamma, beta)
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
        """Return logits plus bounded FiLM parameters for interpretation/export."""

        context_embedding = self.context_encoder(context)
        hidden = image.reshape(-1, *self.spec.input_shape)
        parameters = self.compute_bounded_parameters(context_embedding)
        pre_film_maps: list[torch.Tensor] = []
        post_film_maps: list[torch.Tensor] = []
        pooled_maps: list[torch.Tensor] = []

        for index, layer in enumerate(self.layers, start=1):
            hidden = layer[0](hidden)
            hidden = layer[1](hidden)
            if index == self.final_block_index:
                pre_film = hidden
                gamma = parameters["gamma"]
                beta = parameters["beta"]
                if not isinstance(gamma, torch.Tensor) or not isinstance(beta, torch.Tensor):
                    raise RuntimeError("Bounded FiLM parameters must be tensors.")
                post_film = self.film_layer(pre_film, gamma, beta)
                if keep_feature_maps:
                    pre_film_maps.append(pre_film.detach().clone())
                    post_film_maps.append(post_film.detach().clone())
                hidden = post_film
            hidden = layer[2](hidden)
            hidden = layer[3](hidden)
            if index == self.final_block_index and keep_feature_maps:
                pooled_maps.append(hidden.detach().clone())

        logits = self.fc1(hidden.reshape(hidden.shape[0], -1))
        block_summary = self._block_summary_from_parameters(parameters)
        result: dict[str, Any] = {
            "logits": logits,
            "context_embedding": context_embedding,
            "film_parameters": [block_summary],
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
        """Return the final-block FiLM module for post-FiLM Grad-CAM hooks."""

        return {f"layer{self.final_block_index}_film": self.film_layer}

    def _forward_bounded_blocks(
        self,
        image: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | int]]:
        """Run CNN blocks with bounded FiLM only in the final block."""

        parameters = self.compute_bounded_parameters(context_embedding)
        hidden = self._forward_bounded_blocks_with_parameters(image, parameters)
        return hidden, parameters

    def _infer_flatten_dim(self) -> int:
        """Infer bounded FiLM CNN flatten dimension with dummy tensors."""

        with torch.no_grad():
            image = torch.zeros(1, *self.spec.input_shape)
            context = torch.zeros(1, int(self.context_encoder.spec.context_dim))
            context_embedding = self.context_encoder(context)
            hidden, _parameters = self._forward_bounded_blocks(image, context_embedding)
            return int(hidden.reshape(1, -1).shape[1])

    def _init_identity_modulation(self) -> None:
        """Zero-init heads so bounded FiLM starts as identity modulation."""

        nn.init.zeros_(self.gamma_head.weight)
        if self.gamma_head.bias is not None:
            nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        if self.beta_head.bias is not None:
            nn.init.zeros_(self.beta_head.bias)


class UncertaintyGatedLastBlockFilmContextStockCNN(BoundedLastBlockFilmContextStockCNN):
    """Bounded last-block FiLM gated by the frozen Stage 2 prediction uncertainty.

    N12-A keeps the Stage 2 visual classifier as the reference path and lets the
    context branch intervene mostly on ambiguous visual cases:

        stage2_prob_up = softmax(stage2_logits)[up]
        uncertainty    = 4 * stage2_prob_up * (1 - stage2_prob_up)
        gamma          = 1 + uncertainty * scale * tanh(raw_gamma)
        beta           =     uncertainty * scale * tanh(raw_beta)

    The gate is 1.0 near 50/50 predictions and approaches 0.0 when the visual
    baseline is confident. This tests whether context-FiLM works better as a
    correction layer than as a uniform modulation layer.
    """

    def __init__(
        self,
        spec: CnnVariantSpec,
        context_encoder: ContextEncoder,
        dropout: float = 0.5,
        modulation_scale: float = 0.10,
    ) -> None:
        super().__init__(
            spec=spec,
            context_encoder=context_encoder,
            dropout=dropout,
            modulation_scale=modulation_scale,
        )
        self.mode = UNCERTAINTY_GATED_LAST_BLOCK_METHOD

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return class logits after uncertainty-gated final-block FiLM."""

        context_embedding = self.context_encoder(context)
        gate_values = self.compute_uncertainty_gate(image)
        parameters = self.compute_bounded_parameters(context_embedding, **gate_values)
        hidden = self._forward_bounded_blocks_with_parameters(image, parameters)
        return self.fc1(hidden.reshape(hidden.shape[0], -1))

    def compute_uncertainty_gate(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return Stage 2 probability and `4p(1-p)` uncertainty gate."""

        with torch.no_grad():
            stage2_logits = self._forward_visual_baseline_logits(image)
            stage2_prob_up = torch.softmax(stage2_logits, dim=1)[:, 1:2].detach()
            modulation_gate = (4.0 * stage2_prob_up * (1.0 - stage2_prob_up)).clamp(
                min=0.0,
                max=1.0,
            )
        return {
            "modulation_gate": modulation_gate,
            "stage2_prob_up": stage2_prob_up,
        }

    def forward_with_shapes(
        self,
        image: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, tuple[int, ...]]:
        """Run a no-grad forward pass and return intermediate tensor shapes."""

        self.eval()
        with torch.no_grad():
            context_embedding = self.context_encoder(context)
            gate_values = self.compute_uncertainty_gate(image)
            parameters = self.compute_bounded_parameters(context_embedding, **gate_values)
            hidden = image.reshape(-1, *self.spec.input_shape)
            shapes: dict[str, tuple[int, ...]] = {
                "image": tuple(hidden.shape),
                "context": tuple(context.shape),
                "context_embedding": tuple(context_embedding.shape),
            }
            for index, layer in enumerate(self.layers, start=1):
                hidden = layer[0](hidden)
                shapes[f"layer{index}_conv"] = tuple(hidden.shape)
                hidden = layer[1](hidden)
                shapes[f"layer{index}_bn"] = tuple(hidden.shape)
                if index == self.final_block_index:
                    self._shape_entries_for_parameters(shapes, parameters)
                    gamma = parameters["gamma"]
                    beta = parameters["beta"]
                    if not isinstance(gamma, torch.Tensor) or not isinstance(beta, torch.Tensor):
                        raise RuntimeError("Uncertainty-gated FiLM parameters must be tensors.")
                    hidden = self.film_layer(hidden, gamma, beta)
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
        """Return logits plus uncertainty gate and FiLM parameters."""

        context_embedding = self.context_encoder(context)
        gate_values = self.compute_uncertainty_gate(image)
        parameters = self.compute_bounded_parameters(context_embedding, **gate_values)
        hidden = image.reshape(-1, *self.spec.input_shape)
        pre_film_maps: list[torch.Tensor] = []
        post_film_maps: list[torch.Tensor] = []
        pooled_maps: list[torch.Tensor] = []

        for index, layer in enumerate(self.layers, start=1):
            hidden = layer[0](hidden)
            hidden = layer[1](hidden)
            if index == self.final_block_index:
                pre_film = hidden
                gamma = parameters["gamma"]
                beta = parameters["beta"]
                if not isinstance(gamma, torch.Tensor) or not isinstance(beta, torch.Tensor):
                    raise RuntimeError("Uncertainty-gated FiLM parameters must be tensors.")
                post_film = self.film_layer(pre_film, gamma, beta)
                if keep_feature_maps:
                    pre_film_maps.append(pre_film.detach().clone())
                    post_film_maps.append(post_film.detach().clone())
                hidden = post_film
            hidden = layer[2](hidden)
            hidden = layer[3](hidden)
            if index == self.final_block_index and keep_feature_maps:
                pooled_maps.append(hidden.detach().clone())

        logits = self.fc1(hidden.reshape(hidden.shape[0], -1))
        result: dict[str, Any] = {
            "logits": logits,
            "context_embedding": context_embedding,
            "film_parameters": [self._block_summary_from_parameters(parameters)],
        }
        if keep_feature_maps:
            result["pre_film_feature_maps"] = pre_film_maps
            result["post_film_feature_maps"] = post_film_maps
            result["pooled_feature_maps"] = pooled_maps
        return result


class ConfidenceGatedLastBlockFilmContextStockCNN(UncertaintyGatedLastBlockFilmContextStockCNN):
    """Bounded last-block FiLM gated by frozen Stage 2 prediction confidence.

    N12-B is the counterpart to N12-A. Instead of intervening mostly near the
    Stage 2 decision boundary, it asks whether news context can strengthen
    high-confidence visual evidence:

        stage2_prob_up = softmax(stage2_logits)[up]
        confidence     = abs(2 * stage2_prob_up - 1)
        gamma          = 1 + confidence * scale * tanh(raw_gamma)
        beta           =     confidence * scale * tanh(raw_beta)

    The gate is 0.0 near 50/50 predictions and approaches 1.0 when the frozen
    visual baseline is confident. This can help if context should reinforce
    strong chart signals, but it can also amplify confidently wrong visual
    decisions, so correction/regression counts must be reviewed.
    """

    def __init__(
        self,
        spec: CnnVariantSpec,
        context_encoder: ContextEncoder,
        dropout: float = 0.5,
        modulation_scale: float = 0.10,
    ) -> None:
        super().__init__(
            spec=spec,
            context_encoder=context_encoder,
            dropout=dropout,
            modulation_scale=modulation_scale,
        )
        self.mode = CONFIDENCE_GATED_LAST_BLOCK_METHOD

    def compute_uncertainty_gate(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return Stage 2 probability and `abs(2p-1)` confidence gate.

        The method name is kept to reuse the N12-A inherited forward path.
        """

        with torch.no_grad():
            stage2_logits = self._forward_visual_baseline_logits(image)
            stage2_prob_up = torch.softmax(stage2_logits, dim=1)[:, 1:2].detach()
            modulation_gate = torch.abs(2.0 * stage2_prob_up - 1.0).clamp(
                min=0.0,
                max=1.0,
            )
        return {
            "modulation_gate": modulation_gate,
            "stage2_prob_up": stage2_prob_up,
        }


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


def build_bounded_last_block_film_context_stock_cnn_for_window(
    config: Mapping[str, Any],
    image_window: int,
) -> BoundedLastBlockFilmContextStockCNN:
    """Build the V7 bounded/residual final-block FiLM-context model."""

    window = int(image_window)
    if window not in VARIANT_SPECS:
        raise KeyError(f"Unsupported image_window for Stage 4 FiLM StockCNN: {window}")

    model_config = get_config_section(config, "model")
    selected_model_config = get_model_config(config, window)
    spec = VARIANT_SPECS[window]
    if str(selected_model_config.get("name")) != spec.name:
        raise ValueError(
            f"Config/model mismatch for I{window}: "
            f"config={selected_model_config.get('name')}, implementation={spec.name}"
        )

    stage4_model = get_stage4_model_config(config)
    bounded_config = stage4_model.get(BOUNDED_LAST_BLOCK_METHOD, {})
    if not isinstance(bounded_config, Mapping):
        raise TypeError(f"stage4_model.{BOUNDED_LAST_BLOCK_METHOD} must be a mapping.")

    context_encoder = build_context_encoder_from_config(config)
    model = BoundedLastBlockFilmContextStockCNN(
        spec=spec,
        context_encoder=context_encoder,
        dropout=float(model_config.get("dropout", 0.5)),
        modulation_scale=float(bounded_config.get("modulation_scale", 0.10)),
    )
    actual = count_parameters(model)
    expected = expected_bounded_last_block_film_context_parameter_count(config, window)
    if actual != expected:
        raise ValueError(
            "Stage 4 bounded last-block FiLM parameter mismatch: "
            f"actual={actual}, expected={expected}"
        )
    return model


def build_uncertainty_gated_last_block_film_context_stock_cnn_for_window(
    config: Mapping[str, Any],
    image_window: int,
) -> UncertaintyGatedLastBlockFilmContextStockCNN:
    """Build the N12-A uncertainty-gated bounded final-block FiLM model."""

    window = int(image_window)
    if window not in VARIANT_SPECS:
        raise KeyError(f"Unsupported image_window for Stage 4 FiLM StockCNN: {window}")

    model_config = get_config_section(config, "model")
    selected_model_config = get_model_config(config, window)
    spec = VARIANT_SPECS[window]
    if str(selected_model_config.get("name")) != spec.name:
        raise ValueError(
            f"Config/model mismatch for I{window}: "
            f"config={selected_model_config.get('name')}, implementation={spec.name}"
        )

    stage4_model = get_stage4_model_config(config)
    gated_config = stage4_model.get(
        UNCERTAINTY_GATED_LAST_BLOCK_METHOD,
        stage4_model.get(BOUNDED_LAST_BLOCK_METHOD, {}),
    )
    if not isinstance(gated_config, Mapping):
        raise TypeError(f"stage4_model.{UNCERTAINTY_GATED_LAST_BLOCK_METHOD} must be a mapping.")

    context_encoder = build_context_encoder_from_config(config)
    model = UncertaintyGatedLastBlockFilmContextStockCNN(
        spec=spec,
        context_encoder=context_encoder,
        dropout=float(model_config.get("dropout", 0.5)),
        modulation_scale=float(gated_config.get("modulation_scale", 0.10)),
    )
    actual = count_parameters(model)
    expected = expected_bounded_last_block_film_context_parameter_count(config, window)
    if actual != expected:
        raise ValueError(
            "Stage 4 uncertainty-gated last-block FiLM parameter mismatch: "
            f"actual={actual}, expected={expected}"
        )
    return model


def build_confidence_gated_last_block_film_context_stock_cnn_for_window(
    config: Mapping[str, Any],
    image_window: int,
) -> ConfidenceGatedLastBlockFilmContextStockCNN:
    """Build the N12-B confidence-gated bounded final-block FiLM model."""

    window = int(image_window)
    if window not in VARIANT_SPECS:
        raise KeyError(f"Unsupported image_window for Stage 4 FiLM StockCNN: {window}")

    model_config = get_config_section(config, "model")
    selected_model_config = get_model_config(config, window)
    spec = VARIANT_SPECS[window]
    if str(selected_model_config.get("name")) != spec.name:
        raise ValueError(
            f"Config/model mismatch for I{window}: "
            f"config={selected_model_config.get('name')}, implementation={spec.name}"
        )

    stage4_model = get_stage4_model_config(config)
    gated_config = stage4_model.get(
        CONFIDENCE_GATED_LAST_BLOCK_METHOD,
        stage4_model.get(BOUNDED_LAST_BLOCK_METHOD, {}),
    )
    if not isinstance(gated_config, Mapping):
        raise TypeError(f"stage4_model.{CONFIDENCE_GATED_LAST_BLOCK_METHOD} must be a mapping.")

    context_encoder = build_context_encoder_from_config(config)
    model = ConfidenceGatedLastBlockFilmContextStockCNN(
        spec=spec,
        context_encoder=context_encoder,
        dropout=float(model_config.get("dropout", 0.5)),
        modulation_scale=float(gated_config.get("modulation_scale", 0.10)),
    )
    actual = count_parameters(model)
    expected = expected_bounded_last_block_film_context_parameter_count(config, window)
    if actual != expected:
        raise ValueError(
            "Stage 4 confidence-gated last-block FiLM parameter mismatch: "
            f"actual={actual}, expected={expected}"
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


def expected_bounded_last_block_film_context_parameter_count(
    config: Mapping[str, Any],
    image_window: int,
) -> int:
    """Expected parameter count for V7 bounded final-block FiLM."""

    window = int(image_window)
    if window not in VARIANT_SPECS:
        raise KeyError(f"Unsupported image_window for Stage 4 FiLM StockCNN: {window}")

    spec = VARIANT_SPECS[window]
    embedding_dim = int(get_stage4_model_config(config).get("context_embedding_dim", 32))
    final_channels = int(spec.channels[-1])
    gamma_head_params = embedding_dim * final_channels + final_channels
    beta_head_params = embedding_dim * final_channels + final_channels
    return int(
        spec.expected_num_params
        + _expected_context_encoder_parameter_count(config)
        + gamma_head_params
        + beta_head_params
    )
