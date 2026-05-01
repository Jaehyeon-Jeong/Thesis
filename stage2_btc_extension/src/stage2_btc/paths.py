"""Stage 2 local/Kaggle path helper.

역할:
    raw BTC CSV 위치와 output 위치를 한곳에서 관리한다. runner나 training loop가
    직접 문자열 경로를 조립하기 시작하면 local/Kaggle 차이가 코드 전체로 퍼지기
    쉽다. 이 helper는 config의 `paths` section을 `Stage2Paths` 객체로 바꿔서
    downstream 코드가 같은 방식으로 경로를 사용하게 만든다.

주의:
    이 파일은 output directory만 만든다. raw data directory나 Kaggle input
    directory는 만들거나 수정하지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stage2_btc.config import get_config_section

PATH_SECTION_KEYS: tuple[str, ...] = (
    "project_root",
    "data_root",
    "output_root",
    "checkpoint_root",
    "metrics_root",
    "predictions_root",
    "figures_root",
    "run_manifest_root",
    "reports_root",
    "tables_root",
)


@dataclass(frozen=True)
class Stage2Paths:
    """Stage 2 runner가 공유해서 쓰는 filesystem path 모음.

    주요 field:
        `data_root`: BTC CSV가 있는 root. Kaggle에서는 보통 `/kaggle/input`.
        `source_file`: 특정 BTC CSV 파일. Kaggle auto-detect를 쓸 때는 `None`.
        `output_root`: checkpoint, metrics, predictions, figures를 묶는 root.

    이 객체는 데이터 자체가 아니라 위치 정보만 담는다.
    """

    project_root: Path
    data_root: Path
    source_file: Path | None
    output_root: Path
    checkpoint_root: Path
    metrics_root: Path
    predictions_root: Path
    figures_root: Path
    run_manifest_root: Path
    reports_root: Path
    tables_root: Path

    def as_dict(self) -> dict[str, str | None]:
        """run manifest나 debug log에 저장하기 쉬운 path dictionary를 반환한다."""

        return {
            "project_root": str(self.project_root),
            "data_root": str(self.data_root),
            "source_file": str(self.source_file) if self.source_file is not None else None,
            "output_root": str(self.output_root),
            "checkpoint_root": str(self.checkpoint_root),
            "metrics_root": str(self.metrics_root),
            "predictions_root": str(self.predictions_root),
            "figures_root": str(self.figures_root),
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


def build_stage2_paths(config: Mapping[str, Any]) -> Stage2Paths:
    """config의 `paths` section으로 `Stage2Paths`를 만든다.

    입력:
        `load_config()`가 반환한 전체 config.

    출력:
        runner, loader, trainer, evaluator가 공유하는 path object.
    """

    paths_section = get_config_section(config, "paths")
    missing = [key for key in PATH_SECTION_KEYS if key not in paths_section]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"Missing required path config key(s): {missing_list}")

    path_values = {
        key: Path(str(paths_section[key])).expanduser()
        for key in PATH_SECTION_KEYS
    }
    source_file = _optional_path(paths_section.get("source_file"))
    return Stage2Paths(source_file=source_file, **path_values)


def ensure_stage2_output_dirs(paths: Stage2Paths) -> list[Path]:
    """Stage 2 output directory들을 만들고 생성/확인한 목록을 반환한다.

    생성 대상:
        checkpoint, metric JSON/CSV, prediction CSV, Grad-CAM figure,
        run manifest, report table directory.

    raw BTC data는 읽기 전용 입력이므로 이 함수가 건드리지 않는다.
    """

    output_dirs = [
        paths.output_root,
        paths.checkpoint_root,
        paths.metrics_root,
        paths.predictions_root,
        paths.figures_root,
        paths.run_manifest_root,
        paths.reports_root,
        paths.tables_root,
    ]
    for directory in output_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    return output_dirs


def experiment_output_roots(
    paths: Stage2Paths,
    experiment_name: str,
    run_seed: int,
) -> dict[str, Path]:
    """한 experiment/seed 조합의 output root들을 계산한다.

    입력:
        `experiment_name`: 예를 들어 `stage2_i20_ohlc_ma_vb_r20`.
        `run_seed`: 예를 들어 `42`.

    출력:
        checkpoint, metrics, predictions, figures가 저장될 seed-specific 경로.
        실제 directory 생성은 runner가 필요할 때 수행한다.
    """

    seed_name = f"seed_{int(run_seed)}"
    return {
        "checkpoint": paths.checkpoint_root / experiment_name / seed_name,
        "metrics": paths.metrics_root / experiment_name / seed_name,
        "predictions": paths.predictions_root / experiment_name / seed_name,
        "figures": paths.figures_root / experiment_name / seed_name,
    }
