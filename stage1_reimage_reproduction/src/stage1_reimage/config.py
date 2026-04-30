"""Config loading helpers for Stage 1.

Source policy:
    Root PLAN.md says local and Kaggle runs must share one codebase, with
    runtime/path differences controlled by config. This module implements only
    that config boundary. It does not define model, label, split, or evaluation
    behavior; those decisions stay in their own checklist gates.
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
        Parsed YAML config.
    """

    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

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
    """Raise a clear error if required top-level config sections are missing."""

    missing = [key for key in required_keys if key not in config]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"Missing required config section(s): {missing_list}")


def get_config_section(config: Mapping[str, Any], section_name: str) -> Mapping[str, Any]:
    """Return a named config section and ensure it is a mapping."""

    section = config.get(section_name)
    if not isinstance(section, Mapping):
        raise TypeError(f"Config section must be a mapping: {section_name}")
    return section
