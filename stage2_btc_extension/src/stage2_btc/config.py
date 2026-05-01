"""Stage 2 config loading helper.

역할:
    Stage 2는 local smoke test와 Kaggle full run이 같은 Python 코드를 사용한다.
    환경마다 달라지는 값은 YAML config에만 두고, 이 파일은 그 YAML을 읽고
    필수 section이 빠지지 않았는지 확인한다.

실험 흐름에서의 위치:
    `scripts/run_stage2_btc_baseline.py` 같은 runner가 가장 먼저 이 helper로
    config를 읽는다. 그 다음 같은 config object가 path builder, BTC data loader,
    image generator, model builder, training loop, evaluation, Grad-CAM으로
    전달된다.
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
    "imaging",
    "normalization",
    "model",
    "training",
    "evaluation",
    "trading",
    "gradcam",
    "reproducibility",
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Stage 2 YAML config를 읽고 top-level schema를 검증한다.

    입력:
        `configs/env_local.yaml` 또는 `configs/env_kaggle.yaml` 같은 YAML 경로.

    출력:
        Python dictionary로 변환된 config. 이후 단계의 함수들은 이 config에서
        필요한 section만 꺼내 사용한다.
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
    """필수 top-level config section이 없으면 실행 시작 전에 실패시킨다.

    예:
        `trading` section이 빠진 상태로 baseline을 돌리면 학습은 끝나도 나중에
        trading metric을 만들 수 없다. 이런 문제를 학습 전에 발견하기 위해
        필수 section을 초기에 검사한다.
    """

    missing = [key for key in required_keys if key not in config]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"Missing required config section(s): {missing_list}")


def get_config_section(config: Mapping[str, Any], section_name: str) -> Mapping[str, Any]:
    """config에서 특정 section 하나를 꺼내고 dictionary 형태인지 확인한다.

    입력:
        `config`: `load_config()`가 반환한 전체 config.
        `section_name`: 예를 들어 `"training"`, `"imaging"`, `"model"`.

    출력:
        요청한 section mapping. 이 값은 downstream 함수가 batch size, image size,
        split date 같은 값을 읽는 데 사용한다.
    """

    section = config.get(section_name)
    if not isinstance(section, Mapping):
        raise TypeError(f"Config section must be a mapping: {section_name}")
    return section


def get_window_config(config: Mapping[str, Any], image_window: int) -> Mapping[str, Any]:
    """image window별 image layout config를 반환한다.

    입력:
        `image_window`: `5`, `20`, `60` 중 하나.

    출력:
        해당 window의 `height`, `width`, `price_height`, `volume_height`,
        `gap_height`, `ma_window` 설정. BTC image generator는 이 값을 사용해
        `(height, width)` 크기의 1-channel chart image를 만든다.
    """

    imaging_config = get_config_section(config, "imaging")
    window_specs = imaging_config.get("window_specs")
    if not isinstance(window_specs, Mapping):
        raise TypeError("Config section imaging.window_specs must be a mapping.")

    key = str(int(image_window))
    window_config = window_specs.get(key)
    if not isinstance(window_config, Mapping):
        available = ", ".join(str(item) for item in window_specs.keys())
        raise KeyError(f"Unknown image_window={image_window}. Available: {available}")
    return window_config


def get_model_config(config: Mapping[str, Any], image_window: int) -> Mapping[str, Any]:
    """image window별 CNN model config를 반환한다.

    입력:
        `image_window`: model family를 고르는 기준. return horizon이 아니라
        입력 이미지 크기인 `I5`, `I20`, `I60`에 따라 모델이 달라진다.

    출력:
        해당 window의 model name, input shape, block 수, channel 수, dilation,
        expected flatten dim, expected parameter count.
    """

    model_config = get_config_section(config, "model")
    window_models = model_config.get("window_models")
    if not isinstance(window_models, Mapping):
        raise TypeError("Config section model.window_models must be a mapping.")

    key = str(int(image_window))
    selected = window_models.get(key)
    if not isinstance(selected, Mapping):
        available = ", ".join(str(item) for item in window_models.keys())
        raise KeyError(f"Unknown model image_window={image_window}. Available: {available}")
    return selected


def make_experiment_name(
    image_window: int,
    image_spec: str,
    return_horizon: int,
) -> str:
    """Stage 2 output folder에 사용할 실험 이름을 만든다.

    예:
        `I20`, `ohlc_ma_vb`, `R20` 조합은
        `stage2_i20_ohlc_ma_vb_r20`이 된다.

    이 이름은 checkpoint, prediction, metric, Grad-CAM output 경로에서 같은
    실험을 한 묶음으로 찾기 위한 key 역할을 한다.
    """

    return f"stage2_i{int(image_window)}_{image_spec}_r{int(return_horizon)}"
