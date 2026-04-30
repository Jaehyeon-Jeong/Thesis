#!/usr/bin/env python3
"""1-I10 Kaggle single-seed full run 산출물 존재 여부를 확인한다.

이 script는 model을 다시 학습하지 않는다. Kaggle에서 full run이 끝난 뒤
checkpoint, prediction CSV, metric JSON, Grad-CAM figure가 빠짐없이 만들어졌는지
검사하는 receipt checker다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_stage1_src_to_path() -> Path:
    """현재 script 기준으로 Stage 1 `src/`를 import path에 추가한다."""

    stage_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage_root / "src"))
    return stage_root


def parse_args(stage_root: Path) -> argparse.Namespace:
    """산출물 확인에 필요한 CLI 인자를 parsing한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=stage_root / "configs" / "env_kaggle.yaml",
        help="Stage 1 환경 config 경로.",
    )
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--gradcam-year", type=int, default=2019)
    parser.add_argument(
        "--horizons",
        nargs="*",
        default=None,
        help="검사할 horizon 이름. 생략하면 공개 monthly_20d의 I20/R5,R20,R60 전체.",
    )
    return parser.parse_args()


def main() -> int:
    """예상 산출물을 검사하고 JSON summary를 stdout에 출력한다."""

    stage_root = add_stage1_src_to_path()
    args = parse_args(stage_root)

    from stage1_reimage.config import load_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.data import TARGET_COLUMNS  # pylint: disable=import-outside-toplevel
    from stage1_reimage.paths import build_stage1_paths  # pylint: disable=import-outside-toplevel

    config = load_config(args.config)
    paths = build_stage1_paths(config)
    horizons = tuple(args.horizons or TARGET_COLUMNS.keys())

    checked_files: list[dict[str, object]] = []
    missing_files: list[str] = []

    def require_file(label: str, path: Path) -> None:
        """파일 하나를 검사하고 summary list에 기록한다."""

        exists = path.exists() and path.is_file()
        checked_files.append(
            {
                "label": label,
                "path": str(path),
                "exists": bool(exists),
                "size_bytes": int(path.stat().st_size) if exists else 0,
            }
        )
        if not exists:
            missing_files.append(str(path))

    require_file("run_manifest", paths.run_manifest_root / "run_manifest.json")

    for horizon in horizons:
        if horizon not in TARGET_COLUMNS:
            raise KeyError(f"Unknown horizon: {horizon}")

        checkpoint_dir = paths.checkpoint_root / horizon / f"seed_{args.run_seed}"
        metrics_dir = paths.metrics_root / horizon / f"seed_{args.run_seed}"
        predictions_dir = paths.predictions_root / horizon / f"seed_{args.run_seed}"
        gradcam_dir = paths.figures_root / "gradcam" / horizon / f"seed_{args.run_seed}" / args.split

        require_file(f"{horizon}: best checkpoint", checkpoint_dir / "best.pt")
        require_file(f"{horizon}: last checkpoint", checkpoint_dir / "last.pt")
        require_file(f"{horizon}: train history", metrics_dir / "train_history.csv")
        require_file(f"{horizon}: train metadata", metrics_dir / "train_metadata.json")
        require_file(
            f"{horizon}: {args.split} predictions",
            predictions_dir / f"{args.split}_predictions.csv",
        )
        require_file(f"{horizon}: {args.split} metrics", metrics_dir / f"{args.split}_metrics.json")
        require_file(
            f"{horizon}: {args.split} correlation metrics",
            metrics_dir / f"{args.split}_correlation_metrics.json",
        )
        require_file(
            f"{horizon}: Grad-CAM figure",
            gradcam_dir / f"figure13_style_{args.gradcam_year}_{args.split}.png",
        )
        require_file(f"{horizon}: Grad-CAM samples", gradcam_dir / "samples.csv")
        require_file(f"{horizon}: Grad-CAM summary", gradcam_dir / "summary.json")

    summary = {
        "status": "ok" if not missing_files else "missing_outputs",
        "config": str(args.config),
        "run_seed": args.run_seed,
        "split": args.split,
        "gradcam_year": args.gradcam_year,
        "horizons": list(horizons),
        "num_checked_files": len(checked_files),
        "num_missing_files": len(missing_files),
        "missing_files": missing_files,
        "checked_files": checked_files,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not missing_files else 2


if __name__ == "__main__":
    raise SystemExit(main())
