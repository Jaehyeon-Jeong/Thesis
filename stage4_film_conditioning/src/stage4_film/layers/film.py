"""Feature-wise Linear Modulation layer for Stage 4.

FiLM applies condition-generated channel-wise affine parameters to a CNN
feature map:

    F'[:, c, :, :] = gamma[:, c] * F[:, c, :, :] + beta[:, c]

In Stage 4, `gamma` and `beta` come from structured market context rather than
from a natural-language question encoder. The operation remains the same: each
sample receives its own channel-wise modulation values.
"""

from __future__ import annotations

import torch
from torch import nn


class FeatureWiseAffineModulation(nn.Module):
    """Apply channel-wise FiLM modulation to a 4D CNN feature map.

    Tensor rules:
        feature_map: `(batch_size, channels, height, width)`.
        gamma: `(batch_size, channels)` or `(batch_size, channels, 1, 1)`.
        beta: optional `(batch_size, channels)` or `(batch_size, channels, 1, 1)`.

    If beta is `None`, the layer behaves as gamma-only FiLM:
        `F' = gamma * F`.
    Otherwise it applies full FiLM:
        `F' = gamma * F + beta`.
    """

    def forward(
        self,
        feature_map: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the modulated feature map."""

        if feature_map.ndim != 4:
            raise ValueError(
                "feature_map must have shape (batch, channels, height, width); "
                f"got {tuple(feature_map.shape)}"
            )
        gamma_view = self._reshape_parameter(gamma, feature_map, name="gamma")
        if beta is None:
            return feature_map * gamma_view
        beta_view = self._reshape_parameter(beta, feature_map, name="beta")
        return feature_map * gamma_view + beta_view

    def forward_with_shapes(
        self,
        feature_map: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor | None = None,
    ) -> dict[str, tuple[int, ...]]:
        """Apply FiLM and return shape metadata for smoke checks."""

        output = self.forward(feature_map, gamma, beta)
        return {
            "feature_map": tuple(feature_map.shape),
            "gamma": tuple(gamma.shape),
            "beta": tuple(beta.shape) if beta is not None else (),
            "output": tuple(output.shape),
        }

    @staticmethod
    def _reshape_parameter(
        parameter: torch.Tensor,
        feature_map: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        """Validate and reshape a channel-wise FiLM parameter."""

        batch_size, channels = int(feature_map.shape[0]), int(feature_map.shape[1])
        if parameter.ndim == 2:
            expected = (batch_size, channels)
            if tuple(parameter.shape) != expected:
                raise ValueError(
                    f"{name} shape mismatch: got {tuple(parameter.shape)}, "
                    f"expected {expected}"
                )
            return parameter.reshape(batch_size, channels, 1, 1)
        if parameter.ndim == 4:
            expected = (batch_size, channels, 1, 1)
            if tuple(parameter.shape) != expected:
                raise ValueError(
                    f"{name} shape mismatch: got {tuple(parameter.shape)}, "
                    f"expected {expected}"
                )
            return parameter
        raise ValueError(
            f"{name} must have shape (batch, channels) or "
            f"(batch, channels, 1, 1); got {tuple(parameter.shape)}"
        )
