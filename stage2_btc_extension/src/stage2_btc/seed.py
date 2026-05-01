"""Stage 2 reproducibility helper.

역할:
    BTC split shuffling, model weight initialization, PyTorch sampling 같은
    random 동작을 seed 하나로 최대한 재현 가능하게 맞춘다.

주의:
    seed를 고정해도 GPU kernel, library version, DataLoader worker scheduling에
    따라 완전히 bit-level 동일하지 않을 수 있다. 그래서 run manifest에 seed와
    runtime option을 같이 저장한다.
"""

from __future__ import annotations

import os
import random
from typing import Any


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Python, NumPy, PyTorch random seed를 가능한 범위에서 설정한다.

    입력:
        `seed`: 현재 run에 적용할 정수 seed.
        `deterministic`: cuDNN deterministic mode를 켤지 여부.

    출력:
        어떤 library에 seed가 적용되었는지 기록한 dictionary. 이 값은 나중에
        run manifest에 넣어 재현성 기록으로 사용할 수 있다.
    """

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
