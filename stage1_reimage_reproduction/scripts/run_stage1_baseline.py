#!/usr/bin/env python3
"""local 또는 Kaggle config로 1단계 baseline training을 실행한다.

실행 예시:
    local smoke:
        python scripts/run_stage1_baseline.py \
          --config configs/env_local.yaml \
          --run-mode smoke \
          --horizons stage1_i20_r20 \
          --max-train-rows 8 \
          --max-val-rows 4 \
          --normalization-max-images 128 \
          --max-epochs 1

    Kaggle single-seed:
        python scripts/run_stage1_baseline.py \
          --config configs/env_kaggle.yaml \
          --run-mode full_single_seed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_stage1_src_to_path() -> Path:
    """로컬 1단계 `src/` directory를 `sys.path`에 추가한다.

    필요한 이유:
        이 script는 `scripts/` 폴더에서 실행되지만, 재사용 code는
        `src/stage1_reimage/` 아래에 있다. `src/`를 `sys.path`에 추가하면 package를
        설치하지 않아도 Python이 `stage1_reimage.*`를 import할 수 있다.
    """

    # `__file__`은 현재 script path다. `parents[1]`은
    # `.../stage1_reimage_reproduction/scripts/run_stage1_baseline.py`에서
    # `.../stage1_reimage_reproduction`로 이동한다.
    stage_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage_root / "src"))
    return stage_root


def parse_args(stage_root: Path) -> argparse.Namespace:
    """명령행 인자를 parsing한다.

    출력:
        config path, run mode, horizon, seed list, smoke-test row limit 등이
        들어 있는 Namespace 객체.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=stage_root / "configs" / "env_local.yaml",
        help="1단계 환경 config 경로.",
    )
    parser.add_argument(
        "--run-mode",
        choices=["smoke", "full_single_seed", "full_paper_style"],
        default=None,
        help="실행 mode. 생략하면 config의 run.default_run_mode를 사용한다.",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        default=None,
        help="실행할 horizon 이름. smoke는 기본 stage1_i20_r20, full mode는 전체 horizon.",
    )
    parser.add_argument("--run-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--normalization-max-images", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--allow-local-full",
        action="store_true",
        help="config environment.full_run_target이 false여도 non-smoke mode 실행을 허용한다.",
    )
    return parser.parse_args()


def main() -> int:
    """1단계 baseline runner를 실행하고 JSON summary를 출력한다.

    명령행 진입점:
        CLI args -> config -> paths -> RunSelection -> run_stage1_baseline().
    """

    stage_root = add_stage1_src_to_path()
    args = parse_args(stage_root)

    from stage1_reimage.config import load_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.data import TARGET_COLUMNS  # pylint: disable=import-outside-toplevel
    from stage1_reimage.paths import build_stage1_paths  # pylint: disable=import-outside-toplevel
    from stage1_reimage.runners import (  # pylint: disable=import-outside-toplevel
        RunSelection,
        run_stage1_baseline,
    )

    # YAML config를 읽고 path 문자열을 `Stage1Paths`로 변환한다.
    config = load_config(args.config)
    paths = build_stage1_paths(config)
    run_config = config["run"]

    # 명령행 인자는 config default를 override한다. 사용자가 `--run-mode`를 넘기지
    # 않으면 config가 smoke/full 여부를 결정한다.
    run_mode = args.run_mode or str(run_config["default_run_mode"])
    horizons = tuple(args.horizons or _default_horizons(run_mode, TARGET_COLUMNS))
    run_seeds = tuple(args.run_seeds or _default_run_seeds(run_mode, run_config))

    # `RunSelection`은 runner에 전달되어 어떤 horizon/seed 조합을 학습할지,
    # smoke check용 row cap이 켜져 있는지 알려준다.
    selection = RunSelection(
        run_mode=run_mode,
        horizons=horizons,
        run_seeds=run_seeds,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        normalization_max_images=args.normalization_max_images,
        max_epochs=args.max_epochs,
        allow_local_full=bool(args.allow_local_full),
    )
    # 실제 training은 `run_stage1_baseline()` 안에서 일어난다. 이 script는 argument를
    # 준비하고 반환된 run summary를 출력하는 역할만 한다.
    summary = run_stage1_baseline(config=config, paths=paths, selection=selection)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _default_horizons(run_mode: str, target_columns: dict[str, str]) -> list[str]:
    """선택된 mode에 맞는 안전한 default horizon을 반환한다."""

    if run_mode == "smoke":
        return ["stage1_i20_r20"]
    return list(target_columns)


def _default_run_seeds(run_mode: str, run_config: dict[str, object]) -> list[int]:
    """선택된 mode에 맞는 run seed list를 config에서 가져온다."""

    key = {
        "smoke": "smoke_run_seeds",
        "full_single_seed": "full_single_seed_run_seeds",
        "full_paper_style": "full_paper_style_run_seeds",
    }[run_mode]
    return [int(value) for value in run_config[key]]  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
