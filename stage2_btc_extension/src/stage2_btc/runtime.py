"""Stage 2 runtime helper.

역할:
    local과 Kaggle에서 tensor/model을 어느 device에 올릴지 결정한다. 이 파일은
    model을 만들거나 학습하지 않고, `cpu`/`cuda`/`mps` 같은 실행 device 이름만
    확정한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stage2_btc.config import get_config_section


def select_device(config: Mapping[str, Any]) -> str:
    """config의 `runtime.device` 값을 바탕으로 실행 device를 고른다.

    입력:
        전체 config. local config는 보통 `auto`, Kaggle config는 `cuda`를 요청한다.

    출력:
        `"cpu"`, `"cuda"`, `"mps"` 중 하나. 이후 training code가 이 문자열을
        `torch.device(...)`로 바꿔서 model과 batch tensor를 이동시킨다.
    """

    runtime_config = get_config_section(config, "runtime")
    requested = str(runtime_config.get("device", "auto")).lower()
    fail_if_cuda_unavailable = bool(
        runtime_config.get("fail_if_cuda_unavailable_for_full_run", False)
    )

    try:
        import torch
    except ImportError as exc:
        if requested == "cuda" and fail_if_cuda_unavailable:
            raise RuntimeError("CUDA was requested, but PyTorch is not importable.") from exc
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"

    if requested == "cuda" and not cuda_available:
        if fail_if_cuda_unavailable:
            raise RuntimeError("CUDA was requested by config, but no CUDA device is available.")
        return "cpu"

    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported runtime device value: {requested}")

    return requested


def get_runtime_flag(config: Mapping[str, Any], flag_name: str, default: Any = None) -> Any:
    """`runtime` section에서 보조 실행 flag를 읽는다.

    예:
        `num_workers`, `pin_memory`, `preload_train_val_images` 같은 값은 data
        loading과 training 속도에 영향을 주지만 model 구조를 바꾸지는 않는다.
    """

    runtime_config = get_config_section(config, "runtime")
    return runtime_config.get(flag_name, default)
