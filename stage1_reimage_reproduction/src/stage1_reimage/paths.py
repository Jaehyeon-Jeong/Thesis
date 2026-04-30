"""1단계 local/Kaggle 실행용 path helper.

이 helper들은 filesystem layout을 한곳에서 관리한다. output directory만 만들고,
raw data directory는 만들거나 수정하지 않는다.

읽는 법:
    `Stage1Paths`는 중요한 폴더 위치를 들고 다니는 작은 객체다. runner는 이
    객체를 여러 함수로 넘기면서 `outputs/checkpoints/...` 같은 경로 문자열을
    여기저기서 반복해서 만들지 않게 한다.
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
    """1단계 runner가 사용하는 확정된 filesystem path 모음.

    이 값들은 tensor나 DataFrame이 아니라 파일/폴더 위치다. training은
    `checkpoint_root`에 checkpoint를 쓰고, metric은 `metrics_root`, prediction
    CSV는 `predictions_root` 아래에 저장한다.
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
        """JSON으로 저장 가능한 path dictionary를 반환한다."""

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
    """config의 `paths` section으로 `Stage1Paths`를 만든다.

    입력:
        `load_config()`가 반환한 parsing된 config dict.

    출력:
        runner/training/evaluation code로 전달되는 `Stage1Paths`.
    """

    paths_section = get_config_section(config, "paths")
    missing = [key for key in PATH_SECTION_KEYS if key not in paths_section]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"Missing required path config key(s): {missing_list}")

    # YAML의 각 path 문자열을 Path 객체로 바꾼다. 그래야 뒤에서
    # `paths.checkpoint_root / horizon / seed_dir`처럼 안전하게 경로를 붙일 수 있다.
    path_values = {
        key: Path(str(paths_section[key])).expanduser()
        for key in PATH_SECTION_KEYS
    }
    return Stage1Paths(**path_values)


def ensure_stage1_output_dirs(paths: Stage1Paths) -> list[Path]:
    """필요한 1단계 output directory를 만들고 확인된 path 목록을 반환한다.

    나중에 생성될 checkpoint, metric JSON, prediction CSV, figure, run manifest
    폴더를 준비한다. raw data folder는 의도적으로 여기서 만들지 않는다.
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
