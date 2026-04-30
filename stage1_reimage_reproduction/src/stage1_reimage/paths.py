"""Path helpers for Stage 1 local/Kaggle execution.

The helpers keep filesystem layout explicit and centralized. They create only
output directories; they never create or modify raw data directories.

How to read this file:
    `Stage1Paths` is a small object that carries all important folders through
    the pipeline. Runners pass it around so code does not repeatedly rebuild
    strings like `outputs/checkpoints/...` in different files.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stage1_reimage.config import get_config_section

PATH_SECTION_KEYS: tuple[str, ...] = (
    "project_root",
    "data_root",
    "output_root",
    "checkpoint_root",
    "metrics_root",
    "predictions_root",
    "figures_root",
    "run_manifest_root",
)


@dataclass(frozen=True)
class Stage1Paths:
    """Resolved filesystem paths used by Stage 1 runners.

    These paths are not tensors or data frames. They are filesystem locations.
    Training writes checkpoints under `checkpoint_root`, metrics under
    `metrics_root`, and prediction CSVs under `predictions_root`.
    """

    project_root: Path
    data_root: Path
    output_root: Path
    checkpoint_root: Path
    metrics_root: Path
    predictions_root: Path
    figures_root: Path
    run_manifest_root: Path

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable path dictionary."""

        return {
            "project_root": str(self.project_root),
            "data_root": str(self.data_root),
            "output_root": str(self.output_root),
            "checkpoint_root": str(self.checkpoint_root),
            "metrics_root": str(self.metrics_root),
            "predictions_root": str(self.predictions_root),
            "figures_root": str(self.figures_root),
            "run_manifest_root": str(self.run_manifest_root),
        }


def build_stage1_paths(config: Mapping[str, Any]) -> Stage1Paths:
    """Build `Stage1Paths` from the `paths` config section.

    Input:
        Parsed config dictionary from `load_config()`.

    Output:
        `Stage1Paths`, which is passed into runner/training/evaluation code.
    """

    paths_section = get_config_section(config, "paths")
    missing = [key for key in PATH_SECTION_KEYS if key not in paths_section]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"Missing required path config key(s): {missing_list}")

    # Each YAML path string becomes a `Path` object so later code can safely do
    # operations such as `paths.checkpoint_root / horizon / seed_dir`.
    path_values = {
        key: Path(str(paths_section[key])).expanduser()
        for key in PATH_SECTION_KEYS
    }
    return Stage1Paths(**path_values)


def ensure_stage1_output_dirs(paths: Stage1Paths) -> list[Path]:
    """Create Stage 1 output directories if needed and return verified paths.

    This prepares folders for files that will be generated later:
    checkpoints, metric JSONs, prediction CSVs, figures, and run manifests.
    Raw data folders are intentionally not created here.
    """

    output_dirs = [
        paths.output_root,
        paths.checkpoint_root,
        paths.metrics_root,
        paths.predictions_root,
        paths.figures_root,
        paths.run_manifest_root,
    ]
    for directory in output_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    return output_dirs
