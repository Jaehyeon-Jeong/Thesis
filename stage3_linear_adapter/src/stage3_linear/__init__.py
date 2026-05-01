"""Stage 3 Linear adapter package.

이 package는 Stage 2 BTC pipeline을 그대로 재사용하면서, CNN feature extractor
뒤에 단순 Linear adapter를 추가하는 비교 실험만 담당한다.
"""

from stage3_linear.config import (
    get_linear_adapter_config,
    make_stage2_baseline_experiment_name,
    make_stage3_experiment_name,
    stage3_expected_parameter_count,
)

__all__ = [
    "get_linear_adapter_config",
    "make_stage2_baseline_experiment_name",
    "make_stage3_experiment_name",
    "stage3_expected_parameter_count",
]

