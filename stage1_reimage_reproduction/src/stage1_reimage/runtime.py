"""Runtime helpers shared by local and Kaggle Stage 1 runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stage1_reimage.config import get_config_section


def select_device(config: Mapping[str, Any]) -> str:
    """Select the requested runtime device from config.

    Returns a string instead of a `torch.device` so this scaffold remains easy to
    inspect in smoke checks. Later training code can convert the returned value
    to `torch.device`.
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
