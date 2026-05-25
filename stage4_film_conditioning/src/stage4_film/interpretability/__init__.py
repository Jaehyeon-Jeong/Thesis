"""Stage 4 interpretability helpers."""

from stage4_film.interpretability.gradcam_context import (
    compute_stage4_gradcam_for_image,
    generate_stage4_gradcam_context_figure,
)

__all__ = [
    "compute_stage4_gradcam_for_image",
    "generate_stage4_gradcam_context_figure",
]
