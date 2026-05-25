"""Stage 4 condition encoders and modulation generators."""

from stage4_film.conditions.context_encoder import (
    ContextEncoder,
    ContextEncoderSpec,
    build_context_encoder_from_config,
    count_parameters,
)
from stage4_film.conditions.film_generator import (
    FilmGeneratorSpec,
    FilmParameterGenerator,
    build_film_generator_for_window,
    expected_film_generator_parameter_count,
)

__all__ = [
    "ContextEncoder",
    "ContextEncoderSpec",
    "FilmGeneratorSpec",
    "FilmParameterGenerator",
    "build_context_encoder_from_config",
    "build_film_generator_for_window",
    "count_parameters",
    "expected_film_generator_parameter_count",
]
