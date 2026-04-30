"""Shared utilities for Stage 1 Re-image reproduction.

This package is intentionally small at the 1-I1 scaffold gate. Data loading,
model code, training, evaluation, and Grad-CAM are added in later checklist
items so each implementation step stays auditable.
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
