#!/usr/bin/env python3
"""1단계 label, split, normalization 구현을 smoke check한다.

이 script는 model training 전에 data preparation step을 검증한다:
future-return label, deterministic split assignment, train-only pixel normalization.
작은 JSON audit output을 `outputs/metrics/` 아래에 저장한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_stage1_src_to_path() -> Path:
    """로컬 1단계 `src/` directory를 `sys.path`에 추가한다."""

    stage_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage_root / "src"))
    return stage_root


def parse_args(stage_root: Path) -> argparse.Namespace:
    """명령행 인자를 parsing한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=stage_root / "configs" / "env_local.yaml",
        help="1단계 환경 config 경로.",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        default=["stage1_i20_r5", "stage1_i20_r20", "stage1_i20_r60"],
        help="확인할 horizon 이름.",
    )
    parser.add_argument(
        "--normalization-max-images",
        type=int,
        default=2048,
        help=(
            "local normalization smoke check에서 사용할 최대 training image 수. "
            "0이면 모든 training image로 계산한다."
        ),
    )
    parser.add_argument(
        "--normalization-chunk-size",
        type=int,
        default=4096,
        help="pixel 통계를 계산할 때 memmap chunk 하나에 넣을 image 수.",
    )
    parser.add_argument(
        "--write-split-index",
        action="store_true",
        help="JSON summary 외에 split_index.csv도 저장한다.",
    )
    return parser.parse_args()


def main() -> int:
    """label/split/normalization smoke check를 실행한다.

    데이터 흐름:
        monthly_20d shards -> base metadata DataFrame -> horizon labels ->
        split frame -> train-only normalization statistics.
    """

    add_stage1_src_to_path()
    args = parse_args(Path(__file__).resolve().parents[1])

    from stage1_reimage.config import load_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.data import (  # pylint: disable=import-outside-toplevel
        TARGET_COLUMNS,
        assign_splits,
        build_base_metadata,
        build_dataset_from_config,
        build_horizon_frame,
        compute_pixel_normalization,
        make_split_summary,
        normalization_settings_from_config,
        split_settings_from_config,
        write_horizon_metadata,
    )
    from stage1_reimage.paths import (  # pylint: disable=import-outside-toplevel
        build_stage1_paths,
        ensure_stage1_output_dirs,
    )

    config = load_config(args.config)
    paths = build_stage1_paths(config)
    ensure_stage1_output_dirs(paths)
    # Dataset은 image를 읽고, base_metadata는 Date/StockID/future return과 row id를
    # 보관한다. 이 script에서는 CNN tensor batch를 만들지 않는다.
    dataset = build_dataset_from_config(config)
    base_metadata = build_base_metadata(dataset.shards)
    split_settings = split_settings_from_config(config)
    normalization_settings = normalization_settings_from_config(config)
    max_images = None if args.normalization_max_images <= 0 else args.normalization_max_images

    horizon_results = {}
    for horizon_name in args.horizons:
        if horizon_name not in TARGET_COLUMNS:
            raise KeyError(f"Unknown horizon: {horizon_name}")

        # target horizon 하나에 대한 label을 만든다. 예: Ret_20d -> label.
        horizon_frame = build_horizon_frame(base_metadata, horizon_name)
        split_frame = assign_splits(horizon_frame, split_settings)
        split_summary = make_split_summary(split_frame, split_settings, horizon_name)

        # image pixel에서 scalar train mean/std를 계산한다. smoke cap은 local 실행을
        # 작게 유지한다.
        normalization_stats = compute_pixel_normalization(
            dataset=dataset,
            split_frame=split_frame,
            settings=normalization_settings,
            target_return_name=TARGET_COLUMNS[horizon_name],
            max_images=max_images,
            chunk_size=args.normalization_chunk_size,
        )

        output_dir = paths.metrics_root / horizon_name
        written_files = write_horizon_metadata(
            output_dir=output_dir,
            split_summary=split_summary,
            normalization_stats=normalization_stats,
            split_frame=split_frame,
            write_split_index=args.write_split_index,
        )
        horizon_results[horizon_name] = {
            "target_return_name": TARGET_COLUMNS[horizon_name],
            "split_summary": split_summary,
            "normalization": normalization_stats.as_dict(),
            "written_files": written_files,
        }

    summary = {
        "status": "ok",
        "config_path": str(args.config),
        "num_base_rows": int(len(base_metadata)),
        "normalization_max_images": max_images,
        "horizon_results": horizon_results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
