"""Stage 4 condition encoders and modulation generators."""

from stage4_film.conditions.context_encoder import (
    ContextEncoder,
    ContextEncoderSpec,
    build_context_encoder_from_config,
    count_parameters,
)

__all__ = [
    "ContextEncoder",
    "ContextEncoderSpec",
    "build_context_encoder_from_config",
    "count_parameters",
]
