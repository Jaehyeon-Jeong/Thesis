"""Stage 3 CLI script shared helpers.

이 파일은 Kaggle/local script들이 같은 방식으로 Stage 3 src와 Stage 2 src를
`sys.path`에 추가하도록 돕는다. Stage 3는 Stage 2 data/evaluation code를 재사용하기
때문에 두 src path가 모두 필요하다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def add_stage3_and_stage2_src_to_path(config_arg: str = "configs/env_local.yaml") -> Path:
    """Stage 3 project root를 반환하고 Stage 3/Stage 2 src를 import path에 넣는다."""

    stage_root = Path(__file__).resolve().parents[1]
    stage3_src = stage_root / "src"
    if str(stage3_src) not in sys.path:
        sys.path.insert(0, str(stage3_src))

    config_path = stage_root / config_arg
    stage2_src = stage_root.parent / "stage2_btc_extension" / "src"
    if config_path.exists():
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            candidate = config.get("stage2_dependency", {}).get("src_path")
            if candidate:
                stage2_src = Path(str(candidate)).expanduser()
        except Exception:
            # Path setup이 실패하면 실제 config load 단계에서 명확한 오류가 난다.
            # 여기서는 sibling fallback을 유지한다.
            pass
    if str(stage2_src) not in sys.path:
        sys.path.insert(0, str(stage2_src))
    return stage_root


def add_stage3_and_stage2_src_from_argv(argv: list[str]) -> Path:
    """`--config` 인자를 argv에서 찾아 path setup을 수행한다."""

    config_arg = "configs/env_local.yaml"
    if "--config" in argv:
        index = argv.index("--config")
        if index + 1 < len(argv):
            config_arg = argv[index + 1]
    return add_stage3_and_stage2_src_to_path(config_arg)

