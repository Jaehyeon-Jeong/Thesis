"""Stage 4 config helpers.

역할:
    Stage 4는 Stage 2 BTC image/data/evaluation pipeline을 고정하고, 별도
    market-context branch만 추가한다. 이 파일은 YAML config를 읽고 Stage 4에
    필요한 section이 빠지지 않았는지 초기에 검증한다.

주의:
    Stage 2 helper를 import해야 하는 script는 먼저 `scripts/_stage4_script_utils.py`
    로 Stage 4 `src`와 Stage 2 `src`를 모두 `sys.path`에 추가해야 한다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

CONTEXT_METHODS: tuple[str, ...] = (
    "concat",
    "gating",
    "film_gamma",
    "film_full",
    "film_full_bounded_last_block",
)

REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "environment",
    "paths",
    "stage2_dependency",
    "data",
    "context",
    "runtime",
    "run",
    "split",
    "imaging",
    "normalization",
    "model",
    "stage4_model",
    "training",
    "evaluation",
    "trading",
    "gradcam",
    "reproducibility",
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Stage 4 YAML config를 읽고 top-level schema를 검증한다."""

    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must parse to a dictionary: {path}")

    require_config_keys(config)
    validate_stage4_config(config)
    return config


def require_config_keys(
    config: Mapping[str, Any],
    required_keys: Sequence[str] = REQUIRED_TOP_LEVEL_KEYS,
) -> None:
    """필수 top-level section이 없으면 실행 전에 명확히 실패시킨다."""

    missing = [key for key in required_keys if key not in config]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"Missing required config section(s): {missing_list}")


def get_config_section(config: Mapping[str, Any], section_name: str) -> Mapping[str, Any]:
    """config에서 section 하나를 꺼내고 mapping인지 확인한다."""

    section = config.get(section_name)
    if not isinstance(section, Mapping):
        raise TypeError(f"Config section must be a mapping: {section_name}")
    return section


def get_stage2_dependency_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Stage 2 source/output dependency section을 반환한다."""

    return get_config_section(config, "stage2_dependency")


def get_context_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Stage 4 numeric/news context 설정 section을 반환한다."""

    return get_config_section(config, "context")


def get_stage4_model_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Stage 4 fusion/modulation model 설정 section을 반환한다."""

    return get_config_section(config, "stage4_model")


def get_primary_context_features(config: Mapping[str, Any]) -> list[str]:
    """첫 numeric-context run에 사용할 feature 이름 목록을 반환한다."""

    context_config = get_context_config(config)
    features = context_config.get("primary_features")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        raise TypeError("Config context.primary_features must be a sequence of strings.")
    result = [str(feature) for feature in features]
    if not result:
        raise ValueError("Config context.primary_features must not be empty.")
    return result


def validate_context_method(context_method: str) -> str:
    """지원하는 Stage 4 context method인지 확인하고 canonical string을 반환한다."""

    method = str(context_method)
    if method not in CONTEXT_METHODS:
        available = ", ".join(CONTEXT_METHODS)
        raise ValueError(f"Unsupported context_method={method!r}. Available: {available}")
    return method


def validate_stage4_config(config: Mapping[str, Any]) -> None:
    """Stage 4 전용 config 값의 내부 일관성을 확인한다."""

    stage4_model = get_stage4_model_config(config)
    context_config = get_context_config(config)
    features = get_primary_context_features(config)

    context_dim = int(stage4_model.get("context_dim", len(features)))
    if context_dim != len(features):
        raise ValueError(
            "stage4_model.context_dim must match len(context.primary_features): "
            f"{context_dim} != {len(features)}"
        )

    configured_methods = stage4_model.get("context_methods", CONTEXT_METHODS)
    if not isinstance(configured_methods, Sequence) or isinstance(configured_methods, (str, bytes)):
        raise TypeError("stage4_model.context_methods must be a sequence of strings.")
    for method in configured_methods:
        validate_context_method(str(method))

    context_window = int(context_config.get("context_window", 0))
    if context_window <= 0:
        raise ValueError("context.context_window must be a positive integer.")

    primary_window = int(stage4_model.get("primary_image_window", 0))
    primary_horizon = int(stage4_model.get("primary_return_horizon", 0))
    if primary_window <= 0 or primary_horizon <= 0:
        raise ValueError("stage4_model primary image window and return horizon must be positive.")


def make_stage4_experiment_name(
    image_window: int,
    image_spec: str,
    return_horizon: int,
    context_method: str,
    context_window: int,
    experiment_suffix: str = "",
) -> str:
    """Stage 4 output folder에 사용할 실험 이름을 만든다.

    예:
        `I60`, `R20`, `ohlc_ma_vb`, method `film_full`, context window `60`
        조합은 `stage4_film_full_i60_ohlc_ma_vb_r20_c60`이 된다.
    """

    method = validate_context_method(context_method)
    base_name = (
        f"stage4_{method}_i{int(image_window)}_"
        f"{image_spec}_r{int(return_horizon)}_c{int(context_window)}"
    )
    suffix = sanitize_name_suffix(experiment_suffix)
    return f"{base_name}_{suffix}" if suffix else base_name


def sanitize_name_suffix(value: Any) -> str:
    """Return a filesystem-safe optional suffix for experiment/context names."""

    suffix = str(value or "").strip().lower()
    if not suffix:
        return ""
    cleaned = []
    previous_separator = False
    for char in suffix:
        if char.isalnum():
            cleaned.append(char)
            previous_separator = False
        else:
            if not previous_separator:
                cleaned.append("_")
            previous_separator = True
    return "".join(cleaned).strip("_")


def make_stage2_baseline_experiment_name(
    image_window: int,
    image_spec: str,
    return_horizon: int,
) -> str:
    """Stage 4 비교 기준인 Stage 2 baseline experiment name을 만든다."""

    return f"stage2_i{int(image_window)}_{image_spec}_r{int(return_horizon)}"


def stage4_run_context_base(
    config: Mapping[str, Any],
    image_window: int,
    image_spec: str,
    return_horizon: int,
    context_method: str,
    run_seed: int,
) -> dict[str, Any]:
    """manifest에 반복 저장되는 Stage 4 실험 metadata를 만든다."""

    stage4_model = get_stage4_model_config(config)
    context_config = get_context_config(config)
    context_window = int(context_config.get("context_window", image_window))
    method = validate_context_method(context_method)
    context_feature_set_name = str(context_config.get("feature_set_name", "")).strip()
    experiment_suffix = str(
        stage4_model.get("experiment_suffix", context_feature_set_name)
    ).strip()
    return {
        "stage": "stage4",
        "model_family": "stock_cnn_plus_market_context",
        "baseline_stage": "stage2",
        "image_window": int(image_window),
        "image_spec": str(image_spec),
        "return_horizon": int(return_horizon),
        "context_method": method,
        "context_window": context_window,
        "context_feature_set_name": context_feature_set_name or "primary",
        "context_dim": int(stage4_model.get("context_dim", 8)),
        "context_embedding_dim": int(stage4_model.get("context_embedding_dim", 32)),
        "run_seed": int(run_seed),
        "baseline_experiment_name": make_stage2_baseline_experiment_name(
            image_window,
            image_spec,
            return_horizon,
        ),
        "stage4_experiment_name": make_stage4_experiment_name(
            image_window,
            image_spec,
            return_horizon,
            method,
            context_window,
            experiment_suffix=experiment_suffix,
        ),
        "primary_context_features": get_primary_context_features(config),
    }
