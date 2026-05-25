"""Stage 4 structured context feature utilities."""

from stage4_film.context.features import (
    build_context_feature_table,
    make_context_output_name,
)
from stage4_film.context.normalization import (
    ContextScaler,
    fit_transform_context_features,
)
from stage4_film.context.sources import (
    audit_context_sources,
    find_fear_greed_source,
    load_fear_greed_index,
)

__all__ = [
    "ContextScaler",
    "audit_context_sources",
    "build_context_feature_table",
    "find_fear_greed_source",
    "fit_transform_context_features",
    "load_fear_greed_index",
    "make_context_output_name",
]
