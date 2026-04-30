"""1단계 재현성을 위한 seed helper.

이 파일은 random number generator를 제어한다. 모델 구조를 바꾸는 코드는 아니고,
train/validation shuffling, weight initialization, PyTorch sampling 같은 랜덤
작업을 더 재현 가능하게 만든다.
"""

from __future__ import annotations

import os
import random
from typing import Any


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """사용 가능한 경우 Python, NumPy, PyTorch seed를 설정한다.

    정확한 seed list는 run config가 제어한다. 이 helper는 선택된 seed 하나를 현재
    process에 적용하는 역할만 한다.
    """

    # Python hash randomization은 일부 ordering 동작에 영향을 줄 수 있다.
    # 이 환경변수를 고정하는 것도 재현성 설정의 일부다.
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
        # NumPy는 split shuffling과 normalization sampling logic에서 쓰인다.
        np.random.seed(seed)
        applied["numpy"] = True

    try:
        import torch
    except ImportError:
        return applied

    # PyTorch는 이 seed를 weight initialization과 DataLoader generator에 사용한다.
    torch.manual_seed(seed)
    applied["torch"] = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        applied["torch_cuda"] = True

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return applied
