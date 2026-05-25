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
from stage4_film.conditions import (
    ContextEncoder,
    FilmParameterGenerator,
    build_context_encoder_from_config,
    build_film_generator_for_window,
)
from stage4_film.layers import FeatureWiseAffineModulation
from stage4_film.models import (
    ConcatContextStockCNN,
    GatedContextStockCNN,
    build_concat_context_stock_cnn_for_window,
    build_gated_context_stock_cnn_for_window,
)
from stage4_film.paths import Stage4Paths, build_stage4_paths, ensure_stage4_output_dirs

__all__ = [
    "CONTEXT_METHODS",
    "ConcatContextStockCNN",
    "ContextEncoder",
    "FeatureWiseAffineModulation",
    "FilmParameterGenerator",
    "GatedContextStockCNN",
    "Stage4Paths",
    "build_concat_context_stock_cnn_for_window",
    "build_context_encoder_from_config",
    "build_film_generator_for_window",
    "build_gated_context_stock_cnn_for_window",
    "build_stage4_paths",
    "ensure_stage4_output_dirs",
    "get_context_config",
    "get_stage2_dependency_config",
    "get_stage4_model_config",
    "load_config",
    "make_stage4_experiment_name",
]
