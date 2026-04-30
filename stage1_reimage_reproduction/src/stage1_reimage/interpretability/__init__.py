"""1단계 interpretability utility.

현재는 Re-image Figure 13 재현을 위한 Grad-CAM 기능을 제공한다.
BTC, Linear, FiLM 단계도 나중에 같은 Grad-CAM 원칙을 재사용한다.
"""

from stage1_reimage.interpretability.gradcam import (
    GradCamResult,
    GradCamSample,
    compute_gradcam_for_image,
    make_figure13_style_grid,
    select_gradcam_samples,
    write_gradcam_outputs,
)

__all__ = [
    "GradCamResult",
    "GradCamSample",
    "compute_gradcam_for_image",
    "make_figure13_style_grid",
    "select_gradcam_samples",
    "write_gradcam_outputs",
]
