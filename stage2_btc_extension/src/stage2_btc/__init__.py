"""Stage 2 BTC extension package.

이 package는 Stage 1에서 확정한 Re-image/Stock_CNN식 pipeline을 BTC OHLCV에
적용하기 위한 코드를 담는다. 실제 구현은 Stage 2 checklist 순서대로 추가한다.
"""

from stage2_btc.config import (
    get_config_section,
    get_model_config,
    get_window_config,
    load_config,
    make_experiment_name,
)
from stage2_btc.paths import (
    Stage2Paths,
    build_stage2_paths,
    ensure_stage2_output_dirs,
    experiment_output_roots,
)
from stage2_btc.runtime import get_runtime_flag, select_device
from stage2_btc.seed import set_global_seed

__all__ = [
    "Stage2Paths",
    "build_stage2_paths",
    "ensure_stage2_output_dirs",
    "experiment_output_roots",
    "get_config_section",
    "get_model_config",
    "get_runtime_flag",
    "get_window_config",
    "load_config",
    "make_experiment_name",
    "select_device",
    "set_global_seed",
]
