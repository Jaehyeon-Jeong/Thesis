"""Stage 3 Linear adapter config helpers.

역할:
    Stage 3는 Stage 2 config schema를 대부분 그대로 사용한다. 이 파일은 그 위에
    Stage 3 전용 값인 `linear_adapter.adapter_dim`과 experiment name 규칙만 얹는다.

주의:
    데이터 경로, split 날짜, image spec, normalization, training/evaluation 값은
    Stage 2 config key를 그대로 읽는다. Stage 3가 바꾸는 것은 model head뿐이다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stage2_btc.config import get_config_section
from stage2_btc.models.stock_cnn import VARIANT_SPECS


def get_linear_adapter_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Stage 3 전용 `linear_adapter` section을 반환한다."""

    section = config.get("linear_adapter")
    if not isinstance(section, Mapping):
        raise TypeError("Stage 3 config must include a mapping section: linear_adapter")
    return section


def make_stage3_experiment_name(
    image_window: int,
    image_spec: str,
    return_horizon: int,
    adapter_dim: int,
) -> str:
    """Stage 3 output folder에 사용할 실험 이름을 만든다.

    예:
        `I60`, `ohlc_ma_vb`, `R20`, adapter dim `128` 조합은
        `stage3_linear_i60_ohlc_ma_vb_r20_a128`이 된다.
    """

    return (
        f"stage3_linear_i{int(image_window)}_"
        f"{image_spec}_r{int(return_horizon)}_a{int(adapter_dim)}"
    )


def make_stage2_baseline_experiment_name(
    image_window: int,
    image_spec: str,
    return_horizon: int,
) -> str:
    """Stage 3 결과와 비교할 Stage 2 baseline experiment name을 만든다."""

    return f"stage2_i{int(image_window)}_{image_spec}_r{int(return_horizon)}"


def stage3_expected_parameter_count(config: Mapping[str, Any], image_window: int) -> int:
    """Stage 3 Linear adapter model의 예상 parameter 수를 계산한다.

    계산 방식:
        Stage 2 baseline parameter 수에서 기존 final classifier
        `Linear(flatten_dim, 2, bias=True)` parameter를 뺀다. 그 다음 Stage 3의
        `Linear(flatten_dim, adapter_dim, bias=False)`와
        `Linear(adapter_dim, 2, bias=False)` parameter를 더한다.
    """

    adapter_config = get_linear_adapter_config(config)
    adapter_dim = int(adapter_config.get("adapter_dim", 128))
    classifier_bias = bool(adapter_config.get("classifier_bias", False))
    adapter_bias = bool(adapter_config.get("adapter_bias", False))
    spec = VARIANT_SPECS[int(image_window)]

    old_classifier = spec.expected_flatten_dim * 2 + 2
    convolutional_core = spec.expected_num_params - old_classifier
    adapter_params = spec.expected_flatten_dim * adapter_dim
    if adapter_bias:
        adapter_params += adapter_dim
    classifier_params = adapter_dim * 2
    if classifier_bias:
        classifier_params += 2
    return int(convolutional_core + adapter_params + classifier_params)


def get_stage2_dependency_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Stage 2 source/output dependency section을 반환한다."""

    section = config.get("stage2_dependency", {})
    if not isinstance(section, Mapping):
        raise TypeError("Config section stage2_dependency must be a mapping if provided.")
    return section


def stage3_run_context_base(
    config: Mapping[str, Any],
    image_window: int,
    image_spec: str,
    return_horizon: int,
    run_seed: int,
) -> dict[str, Any]:
    """manifest에 반복 저장되는 Stage 3 실험 metadata를 만든다."""

    adapter_config = get_linear_adapter_config(config)
    model_config = get_config_section(config, "model")
    adapter_dim = int(adapter_config.get("adapter_dim", 128))
    return {
        "stage": "stage3",
        "model_family": "stock_cnn_plus_linear_adapter",
        "baseline_stage": "stage2",
        "image_window": int(image_window),
        "image_spec": str(image_spec),
        "return_horizon": int(return_horizon),
        "run_seed": int(run_seed),
        "adapter_dim": adapter_dim,
        "adapter_bias": bool(adapter_config.get("adapter_bias", False)),
        "classifier_bias": bool(adapter_config.get("classifier_bias", False)),
        "baseline_experiment_name": make_stage2_baseline_experiment_name(
            image_window,
            image_spec,
            return_horizon,
        ),
        "stage3_experiment_name": make_stage3_experiment_name(
            image_window,
            image_spec,
            return_horizon,
            adapter_dim,
        ),
        "dropout": float(model_config.get("dropout", 0.5)),
    }

