"""FiLM parameter generators for Stage 4 structured market context.

The generator receives the shared context embedding `(B, 32)` and produces
block-wise channel parameters for the Stock_CNN feature maps.

For gamma-only FiLM:
    output gamma for each block, initialized to `1`.

For full FiLM:
    output gamma and beta for each block, initialized to `gamma=1`, `beta=0`.

The implementation follows the FiLM paper's practical initialization idea:
learn a delta and add it to an identity value so the network starts as the
unconditioned visual backbone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from stage2_btc.models.stock_cnn import VARIANT_SPECS
from stage4_film.config import get_stage4_model_config, validate_context_method


@dataclass(frozen=True)
class FilmGeneratorSpec:
    """FiLM generator structure for one Stock_CNN variant."""

    context_embedding_dim: int
    block_channels: tuple[int, ...]
    mode: str

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata."""

        return {
            "context_embedding_dim": int(self.context_embedding_dim),
            "block_channels": [int(value) for value in self.block_channels],
            "mode": self.mode,
            "num_blocks": len(self.block_channels),
            "emits_beta": self.mode == "film_full",
        }


class FilmParameterGenerator(nn.Module):
    """Generate block-wise gamma/beta tensors from context embeddings."""

    def __init__(self, spec: FilmGeneratorSpec) -> None:
        super().__init__()
        if spec.context_embedding_dim <= 0:
            raise ValueError("context_embedding_dim must be positive.")
        if not spec.block_channels:
            raise ValueError("block_channels must not be empty.")
        mode = validate_context_method(spec.mode)
        if mode not in {"film_gamma", "film_full"}:
            raise ValueError(f"FiLM generator mode must be film_gamma or film_full: {mode}")

        self.spec = FilmGeneratorSpec(
            context_embedding_dim=int(spec.context_embedding_dim),
            block_channels=tuple(int(value) for value in spec.block_channels),
            mode=mode,
        )
        self.gamma_heads = nn.ModuleList(
            nn.Linear(self.spec.context_embedding_dim, channels)
            for channels in self.spec.block_channels
        )
        self.beta_heads = nn.ModuleList()
        if self.spec.mode == "film_full":
            self.beta_heads = nn.ModuleList(
                nn.Linear(self.spec.context_embedding_dim, channels)
                for channels in self.spec.block_channels
            )
        self.reset_to_identity()

    def forward(self, context_embedding: torch.Tensor) -> list[dict[str, torch.Tensor | None]]:
        """Return a list of FiLM parameter dictionaries, one per CNN block."""

        if context_embedding.ndim != 2:
            raise ValueError(
                "context_embedding must have shape (batch, embedding_dim); "
                f"got {tuple(context_embedding.shape)}"
            )
        if context_embedding.shape[1] != self.spec.context_embedding_dim:
            raise ValueError(
                "context embedding dimension mismatch: "
                f"got {context_embedding.shape[1]}, expected {self.spec.context_embedding_dim}"
            )

        parameters: list[dict[str, torch.Tensor | None]] = []
        for index, gamma_head in enumerate(self.gamma_heads):
            delta_gamma = gamma_head(context_embedding)
            gamma = 1.0 + delta_gamma
            beta: torch.Tensor | None = None
            if self.spec.mode == "film_full":
                beta = self.beta_heads[index](context_embedding)
            parameters.append(
                {
                    "gamma": gamma,
                    "beta": beta,
                    "delta_gamma": delta_gamma,
                }
            )
        return parameters

    def forward_with_shapes(self, context_embedding: torch.Tensor) -> dict[str, Any]:
        """Return shape metadata for all generated block parameters."""

        parameters = self.forward(context_embedding)
        block_shapes: list[dict[str, list[int] | None]] = []
        for block in parameters:
            beta = block["beta"]
            block_shapes.append(
                {
                    "gamma": list(block["gamma"].shape),
                    "delta_gamma": list(block["delta_gamma"].shape),
                    "beta": list(beta.shape) if beta is not None else None,
                }
            )
        return {
            "context_embedding": list(context_embedding.shape),
            "blocks": block_shapes,
        }

    def _init_identity_outputs(self) -> None:
        """Zero-init heads so gamma starts at 1 and beta starts at 0."""

        for head in self.gamma_heads:
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        for head in self.beta_heads:
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def reset_to_identity(self) -> None:
        """Public identity reset used after generic model initialization."""

        self._init_identity_outputs()


def build_film_generator_for_window(
    config: Mapping[str, Any],
    image_window: int,
    mode: str,
) -> FilmParameterGenerator:
    """Build a FiLM generator for a Stock_CNN image window."""

    window = int(image_window)
    if window not in VARIANT_SPECS:
        raise KeyError(f"Unsupported image_window for FiLM generator: {window}")
    stage4_model = get_stage4_model_config(config)
    spec = FilmGeneratorSpec(
        context_embedding_dim=int(stage4_model.get("context_embedding_dim", 32)),
        block_channels=tuple(VARIANT_SPECS[window].channels),
        mode=mode,
    )
    return FilmParameterGenerator(spec)


def expected_film_generator_parameter_count(
    *,
    context_embedding_dim: int,
    block_channels: Sequence[int],
    mode: str,
) -> int:
    """Expected parameter count for FiLM generator heads only."""

    canonical_mode = validate_context_method(mode)
    if canonical_mode not in {"film_gamma", "film_full"}:
        raise ValueError(
            f"FiLM generator mode must be film_gamma or film_full: {canonical_mode}"
        )
    per_gamma = sum(
        int(context_embedding_dim) * int(channels) + int(channels)
        for channels in block_channels
    )
    if canonical_mode == "film_gamma":
        return int(per_gamma)
    return int(2 * per_gamma)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Return model parameter count."""

    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return int(sum(parameter.numel() for parameter in parameters))
