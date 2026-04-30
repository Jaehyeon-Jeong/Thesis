"""local과 Kaggle 1단계 실행이 공유하는 runtime helper.

이 파일은 training/evaluation 중 tensor를 어디에 올릴지 결정한다:
`cpu`, `cuda`, 또는 지원되는 다른 device. 선택된 값은 이후
`images.to(device)` 같은 PyTorch 호출에 전달된다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stage1_reimage.config import get_config_section


def select_device(config: Mapping[str, Any]) -> str:
    """config에서 요청한 runtime device를 선택한다.

    `torch.device`가 아니라 문자열을 반환한다. scaffold/smoke check에서 확인하기
    쉽도록 하기 위함이다. 이후 training code가 이 값을 `torch.device`로 변환할 수 있다.
    """

    # 환경 config에서 `runtime.device`를 읽는다. local config는 보통 `auto`,
    # Kaggle config는 `cuda`를 요청한다.
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

    # CUDA 사용 가능 여부가 tensor/model을 GPU로 보낼 수 있는지 결정한다.
    # local에서 CUDA가 없으면 smoke test는 CPU로 fallback한다.
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
