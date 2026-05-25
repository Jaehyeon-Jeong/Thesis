"""Stage 4 reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Any


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Python, NumPy, PyTorch random seed를 가능한 범위에서 설정한다."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    applied: dict[str, Any] = {
        "seed": int(seed),
        "python_random": True,
        "numpy": False,
        "torch": False,
        "torch_cuda": False,
        "deterministic": bool(deterministic),
    }

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
        applied["numpy"] = True

    try:
        import torch
    except ImportError:
        return applied

    torch.manual_seed(seed)
    applied["torch"] = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        applied["torch_cuda"] = True

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return applied
