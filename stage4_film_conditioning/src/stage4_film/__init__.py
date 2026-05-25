"""Stage 4 market-context conditioning package.

Stage 4 keeps the Stage 2 BTC chart-image pipeline fixed and adds a separate
market-context branch for concat, gating, gamma-only FiLM, and full FiLM
comparisons.
"""

from stage4_film.config import (
    CONTEXT_METHODS,
    get_context_config,
    get_stage2_dependency_config,
    get_stage4_model_config,
    load_config,
    make_stage4_experiment_name,
)
from stage4_film.conditions import ContextEncoder, build_context_encoder_from_config
from stage4_film.paths import Stage4Paths, build_stage4_paths, ensure_stage4_output_dirs

__all__ = [
    "CONTEXT_METHODS",
    "ContextEncoder",
    "Stage4Paths",
    "build_context_encoder_from_config",
    "build_stage4_paths",
    "ensure_stage4_output_dirs",
    "get_context_config",
    "get_stage2_dependency_config",
    "get_stage4_model_config",
    "load_config",
    "make_stage4_experiment_name",
]
