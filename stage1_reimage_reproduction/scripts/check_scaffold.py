#!/usr/bin/env python3
"""1단계 shared scaffold를 smoke check한다.

이 script는 1-I1 scaffold만 확인한다: package import, config loading, path 생성,
선택적 output-directory 생성, seed setting, device 선택. 의도적으로 `.dat` image를
읽거나 model을 학습하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_stage1_src_to_path() -> Path:
    """로컬 1단계 `src/` directory를 `sys.path`에 추가한다.

    이 smoke script는 package로 설치되어 있지 않다. 이 helper는 `src/`에서
    `stage1_reimage.config` 같은 local module을 직접 import할 수 있게 한다.
    """

    stage_root = Path(__file__).resolve().parents[1]
    src_root = stage_root / "src"
    sys.path.insert(0, str(src_root))
    return stage_root


def parse_args(stage_root: Path) -> argparse.Namespace:
    """scaffold smoke check용 명령행 인자를 parsing한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=stage_root / "configs" / "env_local.yaml",
        help="1단계 환경 config 경로.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="현재 smoke-check process에 적용할 seed.",
    )
    parser.add_argument(
        "--create-output-dirs",
        action="store_true",
        help="config에 지정된 output directory가 없으면 생성한다.",
    )
    return parser.parse_args()


def main() -> int:
    """scaffold check를 실행하고 작은 JSON summary를 출력한다.

    이 check는 infrastructure만 확인한다:
        config를 load할 수 있는지, path를 resolve할 수 있는지, seed를 설정할 수
        있는지, runtime device를 선택할 수 있는지 확인한다. image tensor는 다루지 않는다.
    """

    stage_root = add_stage1_src_to_path()
    args = parse_args(stage_root)

    from stage1_reimage import (  # pylint: disable=import-outside-toplevel
        build_stage1_paths,
        ensure_stage1_output_dirs,
        load_config,
        select_device,
        set_global_seed,
    )

    # config/path helper는 이후 모든 pipeline step의 첫 dependency다. data/model code를
    # 사용하기 전에 이 helper들이 정상인지 먼저 확인한다.
    config = load_config(args.config)
    paths = build_stage1_paths(config)
    created_or_verified_dirs = []
    if args.create_output_dirs:
        created_or_verified_dirs = [
            str(path) for path in ensure_stage1_output_dirs(paths)
        ]

    # `seed_info`는 현재 환경에서 어떤 random library가 실제로 사용 가능하고 seed가
    # 설정되었는지 기록한다.
    seed_info = set_global_seed(args.seed)
    device = select_device(config)

    summary = {
        "status": "ok",
        "stage_root": str(stage_root),
        "config_path": str(args.config),
        "environment": config["environment"]["name"],
        "selected_device": device,
        "data_root_exists": paths.data_root.exists(),
        "output_root": str(paths.output_root),
        "created_or_verified_dirs": created_or_verified_dirs,
        "seed_info": seed_info,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
