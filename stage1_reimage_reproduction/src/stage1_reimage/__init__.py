"""1단계 Re-image 재현에서 공유하는 utility.

이 package는 1-I1 scaffold gate에서는 의도적으로 작게 시작했다. data loading,
model code, training, evaluation, Grad-CAM은 이후 checklist item에서 하나씩
추가해 각 구현 단계가 audit 가능하도록 한다.
"""

from stage1_reimage.config import get_config_section, load_config, require_config_keys
from stage1_reimage.paths import (
    Stage1Paths,
    build_stage1_paths,
    ensure_stage1_output_dirs,
)
from stage1_reimage.runtime import select_device
from stage1_reimage.seed import set_global_seed

__all__ = [
    "Stage1Paths",
    "build_stage1_paths",
    "ensure_stage1_output_dirs",
    "get_config_section",
    "load_config",
    "require_config_keys",
    "select_device",
    "set_global_seed",
]
