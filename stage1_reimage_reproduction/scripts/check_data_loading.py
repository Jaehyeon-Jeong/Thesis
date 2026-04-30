#!/usr/bin/env python3
"""1단계 monthly_20d data loading을 smoke check한다.

이 script는 shard discovery, row alignment, memmap 기반 image loading, sample
tensor shape를 검증한다. horizon label, split, normalization statistics, image tensor
외 model input, training DataLoader는 만들지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def add_stage1_src_to_path() -> Path:
    """로컬 1단계 `src/` directory를 `sys.path`에 추가한다.

    package를 Python environment에 설치하지 않아도 smoke script가 로컬 1단계 module을
    import할 수 있게 한다.
    """

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
        "--sample-indices",
        nargs="*",
        type=int,
        default=[0, -1],
        help="dataset 생성 후 확인할 global row index 목록.",
    )
    return parser.parse_args()


def summarize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """dataset sample 하나를 작고 JSON 저장 가능한 summary로 바꾼다.

    입력 sample 형태:
        `{"image": tensor(1,64,60), "metadata": {...}}`.
    출력 summary는 전체 image tensor를 덤프하지 않고 shape와 min/max pixel 값만 보여준다.
    """

    image = sample["image"]
    metadata = sample["metadata"]
    return {
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "image_min": float(image.min().item()),
        "image_max": float(image.max().item()),
        "year": metadata["year"],
        "local_row": metadata["local_row"],
        "date": metadata["Date"],
        "stock_id": metadata["StockID"],
        "metadata_columns": sorted(metadata.keys()),
    }


def main() -> int:
    """data-loading smoke check를 실행하고 JSON summary를 출력한다."""

    add_stage1_src_to_path()
    args = parse_args(Path(__file__).resolve().parents[1])

    from stage1_reimage.config import load_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.data import (  # pylint: disable=import-outside-toplevel
        build_dataset_from_config,
    )

    # memmap-backed dataset을 만든다. 이 시점에서 image는 여전히 disk에서 lazy하게
    # 읽히고, label metadata만 memory에 올라간다.
    config = load_config(args.config)
    dataset = build_dataset_from_config(config)
    inspected_samples = []
    for index in args.sample_indices:
        # `dataset[index]`에 접근하면 image row 하나를 읽고 tensor `(1,64,60)`와
        # 해당 row metadata를 반환한다.
        inspected_samples.append(
            {
                "requested_index": index,
                "sample": summarize_sample(dataset[index]),
            }
        )

    summary = {
        "status": "ok",
        "config_path": str(args.config),
        "num_shards": len(dataset.shards),
        "num_rows": len(dataset),
        "first_shard": dataset.shards[0].as_dict(),
        "last_shard": dataset.shards[-1].as_dict(),
        "inspected_samples": inspected_samples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
