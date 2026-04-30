"""Config loading helpers for Stage 1.

Source policy:
    Root PLAN.md says local and Kaggle runs must share one codebase, with
    runtime/path differences controlled by config. This module implements only
    that config boundary. It does not define model, label, split, or evaluation
    behavior; those decisions stay in their own checklist gates.

How to read this file:
    The YAML config is the experiment control panel. Code reads values such as
    data paths, batch size, seed list, and evaluation threshold from that file
    instead of hardcoding them inside model/training functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "environment",
    "paths",
    "data",
    "runtime",
    "run",
    "split",
    "normalization",
    "model",
    "training",
    "evaluation",
    "reproducibility",
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a Stage 1 YAML config and validate its top-level sections.

    Parameters
    ----------
    config_path:
        Path to `configs/env_local.yaml`, `configs/env_kaggle.yaml`, or another
        environment config with the same section contract.

    Returns
    -------
    dict[str, Any]
        Parsed YAML config. The returned object is passed into path builders,
        data loaders, model builders, training code, and evaluation code.
    """

    # Convert strings such as "configs/env_local.yaml" into a Path object.
    # `expanduser()` lets paths like "~/..." work if they are used later.
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    # YAML becomes a nested Python dictionary. Example:
    #   config["training"]["batch_size"] -> 128
    #   config["model"]["input_height"] -> 64
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must parse to a dictionary: {path}")

    require_config_keys(config)
    return config


def require_config_keys(
    config: Mapping[str, Any],
    required_keys: Sequence[str] = REQUIRED_TOP_LEVEL_KEYS,
) -> None:
    """Raise a clear error if required top-level config sections are missing.

    This catches mistakes early. For example, evaluation code expects an
    `evaluation` section, so a missing section should fail before training starts.
    """

    missing = [key for key in required_keys if key not in config]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"Missing required config section(s): {missing_list}")


def get_config_section(config: Mapping[str, Any], section_name: str) -> Mapping[str, Any]:
    """Return one config section and ensure it is a dictionary-like object.

    Example:
        `get_config_section(config, "training")` returns the training block,
        which includes batch size, optimizer settings, and early stopping.
    """

    section = config.get(section_name)
    if not isinstance(section, Mapping):
        raise TypeError(f"Config section must be a mapping: {section_name}")
    return section
