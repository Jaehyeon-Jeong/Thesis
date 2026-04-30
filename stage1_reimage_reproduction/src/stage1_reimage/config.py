"""1단계 config loading helper.

근거/역할:
    Root PLAN.md는 local과 Kaggle이 같은 codebase를 쓰고, 실행 환경 차이는
    config로 관리한다고 고정한다. 이 파일은 그 config 경계만 담당한다.
    모델 구조, label, split, evaluation 로직은 각 단계 파일에서 따로 정의한다.

읽는 법:
    YAML config는 실험 control panel이다. data path, batch size, seed list,
    evaluation threshold 같은 값은 model/training 함수 안에 박아두지 않고
    config에서 읽어온다.
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
    """1단계 YAML config를 읽고 top-level section을 검증한다.

    입력:
        `configs/env_local.yaml`, `configs/env_kaggle.yaml`처럼 같은 section 구조를
        가진 환경별 config 경로.

    출력:
        parsing된 YAML config. 이 dict는 이후 path builder, data loader,
        model builder, training code, evaluation code로 전달된다.
    """

    # "configs/env_local.yaml" 같은 문자열을 Path 객체로 바꾼다.
    # `expanduser()`는 나중에 "~/..." 같은 경로를 써도 동작하게 해준다.
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    # YAML은 nested Python dictionary가 된다. 예:
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
    """필수 top-level config section이 없으면 명확한 error를 낸다.

    예를 들어 evaluation code는 `evaluation` section을 기대한다.
    이 section이 없으면 training이 시작되기 전에 바로 실패해야 한다.
    """

    missing = [key for key in required_keys if key not in config]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"Missing required config section(s): {missing_list}")


def get_config_section(config: Mapping[str, Any], section_name: str) -> Mapping[str, Any]:
    """config의 특정 section 하나를 반환하고 dictionary 형태인지 확인한다.

    예:
        `get_config_section(config, "training")`은 batch size, optimizer,
        early stopping 정보가 들어 있는 training block을 반환한다.
    """

    section = config.get(section_name)
    if not isinstance(section, Mapping):
        raise TypeError(f"Config section must be a mapping: {section_name}")
    return section
