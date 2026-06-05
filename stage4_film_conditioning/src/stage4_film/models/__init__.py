"""Stage 4 context-conditioned Stock_CNN model variants."""

from stage4_film.models.context_stock_cnn import (
    ConcatContextStockCNN,
    GatedContextStockCNN,
    build_concat_context_stock_cnn_for_window,
    build_gated_context_stock_cnn_for_window,
    expected_concat_context_parameter_count,
    expected_gated_context_parameter_count,
)
from stage4_film.models.film_stock_cnn import (
    BoundedLastBlockFilmContextStockCNN,
    FilmContextStockCNN,
    UncertaintyGatedLastBlockFilmContextStockCNN,
    build_bounded_last_block_film_context_stock_cnn_for_window,
    build_film_context_stock_cnn_for_window,
    build_uncertainty_gated_last_block_film_context_stock_cnn_for_window,
    expected_bounded_last_block_film_context_parameter_count,
    expected_film_context_parameter_count,
)

__all__ = [
    "BoundedLastBlockFilmContextStockCNN",
    "ConcatContextStockCNN",
    "FilmContextStockCNN",
    "GatedContextStockCNN",
    "UncertaintyGatedLastBlockFilmContextStockCNN",
    "build_bounded_last_block_film_context_stock_cnn_for_window",
    "build_concat_context_stock_cnn_for_window",
    "build_film_context_stock_cnn_for_window",
    "build_gated_context_stock_cnn_for_window",
    "build_uncertainty_gated_last_block_film_context_stock_cnn_for_window",
    "expected_bounded_last_block_film_context_parameter_count",
    "expected_concat_context_parameter_count",
    "expected_film_context_parameter_count",
    "expected_gated_context_parameter_count",
]
