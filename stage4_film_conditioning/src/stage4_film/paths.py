"""Stage 4 filesystem path helpers.

역할:
    Stage 4 runner가 raw BTC/F&G input과 checkpoint, metrics, predictions,
    Grad-CAM/context export 위치를 같은 방식으로 사용하도록 path object를 만든다.

주의:
    이 파일은 output directory만 만든다. raw data folder나 Kaggle input directory는
    읽기 전용 입력으로 취급한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stage4_film.config import get_config_section

PATH_SECTION_KEYS: tuple[str, ...] = (
    "project_root",
    "data_root",
    "output_root",
    "checkpoint_root",
    "metrics_root",
    "predictions_root",
    "figures_root",
    "context_root",
    "run_manifest_root",
    "reports_root",
    "tables_root",
)


@dataclass(frozen=True)
class Stage4Paths:
    """Stage 4 runner가 공유해서 쓰는 filesystem path 모음."""

    project_root: Path
    data_root: Path
    source_file: Path | None
    fear_greed_file: Path | None
    output_root: Path
    checkpoint_root: Path
    metrics_root: Path
    predictions_root: Path
    figures_root: Path
    context_root: Path
    run_manifest_root: Path
    reports_root: Path
    tables_root: Path

    def as_dict(self) -> dict[str, str | None]:
        """manifest/debug log에 저장하기 쉬운 path dictionary를 반환한다."""

        return {
            "project_root": str(self.project_root),
            "data_root": str(self.data_root),
            "source_file": str(self.source_file) if self.source_file is not None else None,
            "fear_greed_file": (
                str(self.fear_greed_file) if self.fear_greed_file is not None else None
            ),
            "output_root": str(self.output_root),
            "checkpoint_root": str(self.checkpoint_root),
            "metrics_root": str(self.metrics_root),
            "predictions_root": str(self.predictions_root),
            "figures_root": str(self.figures_root),
            "context_root": str(self.context_root),
            "run_manifest_root": str(self.run_manifest_root),
            "reports_root": str(self.reports_root),
            "tables_root": str(self.tables_root),
        }


def _optional_path(value: Any) -> Path | None:
    """빈 문자열/None은 `None`, 값이 있으면 `Path`로 변환한다."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def build_stage4_paths(config: Mapping[str, Any]) -> Stage4Paths:
    """config의 `paths` section으로 `Stage4Paths`를 만든다."""

    paths_section = get_config_section(config, "paths")
    missing = [key for key in PATH_SECTION_KEYS if key not in paths_section]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"Missing required Stage 4 path config key(s): {missing_list}")

    path_values = {
        key: Path(str(paths_section[key])).expanduser()
        for key in PATH_SECTION_KEYS
    }
    return Stage4Paths(
        source_file=_optional_path(paths_section.get("source_file")),
        fear_greed_file=_optional_path(paths_section.get("fear_greed_file")),
        **path_values,
    )


def ensure_stage4_output_dirs(paths: Stage4Paths) -> list[Path]:
    """Stage 4 output directory들을 만들고 생성/확인한 목록을 반환한다."""

    output_dirs = [
        paths.output_root,
        paths.checkpoint_root,
        paths.metrics_root,
        paths.predictions_root,
        paths.figures_root,
        paths.context_root,
        paths.run_manifest_root,
        paths.reports_root,
        paths.tables_root,
    ]
    for directory in output_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    return output_dirs


def experiment_output_roots(
    paths: Stage4Paths,
    experiment_name: str,
    run_seed: int,
) -> dict[str, Path]:
    """한 Stage 4 experiment/seed 조합의 output root들을 계산한다."""

    seed_name = f"seed_{int(run_seed)}"
    return {
        "checkpoint": paths.checkpoint_root / experiment_name / seed_name,
        "metrics": paths.metrics_root / experiment_name / seed_name,
        "predictions": paths.predictions_root / experiment_name / seed_name,
        "figures": paths.figures_root / experiment_name / seed_name,
        "context": paths.context_root / experiment_name / seed_name,
    }
