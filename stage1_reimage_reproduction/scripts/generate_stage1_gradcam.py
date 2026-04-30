#!/usr/bin/env python3
"""1단계 Re-image Figure 13-style Grad-CAM figure를 생성한다.

사용 전제:
    1. `run_stage1_baseline.py`로 checkpoint를 만든다.
    2. `evaluate_stage1_predictions.py`로 prediction CSV를 만든다.
    3. 이 script가 prediction CSV에서 Up/Down sample을 고르고 Grad-CAM을 저장한다.

로컬 smoke 예시:
    python scripts/generate_stage1_gradcam.py \
      --config configs/env_local.yaml \
      --horizon stage1_i20_r20 \
      --run-seed 42 \
      --split validation \
      --year 1993 \
      --samples-per-class 1

Kaggle Figure 13-style 예시:
    python scripts/generate_stage1_gradcam.py \
      --config configs/env_kaggle.yaml \
      --horizon stage1_i20_r20 \
      --run-seed 42 \
      --split test \
      --year 2019 \
      --samples-per-class 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def add_stage1_src_to_path() -> Path:
    """로컬 1단계 `src/` directory를 `sys.path`에 추가한다."""

    stage_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage_root / "src"))
    return stage_root


def parse_args(stage_root: Path) -> argparse.Namespace:
    """Grad-CAM 실행용 명령행 인자를 parsing한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=stage_root / "configs" / "env_local.yaml",
        help="1단계 환경 config 경로.",
    )
    parser.add_argument("--horizon", default="stage1_i20_r20")
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--prediction-path", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-fallback-any-year",
        action="store_true",
        help="요청한 year에 sample이 부족하면 전체 prediction에서 sample을 고른다.",
    )
    parser.add_argument(
        "--write-report-copy",
        action="store_true",
        help="outputs figure를 reports/figures/gradcam에도 복사한다.",
    )
    return parser.parse_args()


def main() -> int:
    """prediction CSV와 checkpoint를 사용해 Grad-CAM output을 생성한다."""

    add_stage1_src_to_path()
    args = parse_args(Path(__file__).resolve().parents[1])

    from stage1_reimage.config import load_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.data import build_dataset_from_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.evaluation import load_checkpoint_into_model  # pylint: disable=import-outside-toplevel
    from stage1_reimage.interpretability import (  # pylint: disable=import-outside-toplevel
        GradCamResult,
        compute_gradcam_for_image,
        select_gradcam_samples,
        write_gradcam_outputs,
    )
    from stage1_reimage.models import StockCNNI20  # pylint: disable=import-outside-toplevel
    from stage1_reimage.paths import (  # pylint: disable=import-outside-toplevel
        build_stage1_paths,
        ensure_stage1_output_dirs,
    )
    from stage1_reimage.runtime import select_device  # pylint: disable=import-outside-toplevel

    config = load_config(args.config)
    paths = build_stage1_paths(config)
    ensure_stage1_output_dirs(paths)
    prediction_path = args.prediction_path or (
        paths.predictions_root
        / args.horizon
        / f"seed_{args.run_seed}"
        / f"{args.split}_predictions.csv"
    )
    checkpoint_path = args.checkpoint_path or (
        paths.checkpoint_root / args.horizon / f"seed_{args.run_seed}" / "best.pt"
    )
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction CSV does not exist: {prediction_path}. "
            "Run scripts/evaluate_stage1_predictions.py first."
        )

    predictions = pd.read_csv(prediction_path)
    samples = select_gradcam_samples(
        predictions=predictions,
        year=args.year,
        samples_per_class=args.samples_per_class,
        allow_fallback_any_year=bool(args.allow_fallback_any_year),
    )

    device = select_device(config)
    model = StockCNNI20()
    checkpoint = load_checkpoint_into_model(model, checkpoint_path, device)
    normalization = _normalization_from_checkpoint(checkpoint)
    target_layers = model.gradcam_target_layers()
    base_dataset = build_dataset_from_config(config)

    results: list[GradCamResult] = []
    for sample in samples:
        prediction_row = sample.prediction_row
        shard_index = int(prediction_row["shard_index"])
        local_row = int(prediction_row["local_row"])

        # original_image는 Figure 13의 첫 row에 들어갈 raw rendered image다.
        # model input은 같은 image를 train mean/std로 normalize한 별도 tensor다.
        original_tensor = base_dataset.get_image_tensor(shard_index, local_row)
        normalized_tensor = (original_tensor - normalization["train_pixel_mean"]) / normalization[
            "train_pixel_std"
        ]
        normalized_tensor = normalized_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
        heatmaps, warnings = compute_gradcam_for_image(
            model=model,
            image=normalized_tensor,
            target_class=sample.target_class,
            target_layers=target_layers,
        )
        results.append(
            GradCamResult(
                sample=sample,
                original_image=original_tensor.squeeze(0).detach().cpu().numpy(),
                heatmaps=heatmaps,
                heatmap_warnings=warnings,
            )
        )

    output_dir = args.output_dir or (
        paths.figures_root / "gradcam" / args.horizon / f"seed_{args.run_seed}" / args.split
    )
    report_figure_path = None
    if args.write_report_copy:
        report_figure_path = (
            paths.project_root
            / "reports"
            / "figures"
            / "gradcam"
            / f"{args.horizon}_seed_{args.run_seed}_{args.split}_{args.year}_figure13_style.png"
        )
    written = write_gradcam_outputs(
        results=results,
        output_dir=output_dir,
        figure_title=(
            f"{args.horizon.upper()} Grad-CAM, seed {args.run_seed}, "
            f"{args.split}, year {args.year}"
        ),
        year=args.year,
        split_name=args.split,
        report_figure_path=report_figure_path,
    )
    summary = {
        "status": "ok",
        "horizon": args.horizon,
        "split": args.split,
        "year": args.year,
        "run_seed": args.run_seed,
        "num_samples": len(results),
        "prediction_path": str(prediction_path),
        "checkpoint_path": str(checkpoint_path),
        "written": written,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _normalization_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, float]:
    """checkpoint metadata에서 train-only normalization 값을 꺼낸다."""

    metadata = checkpoint.get("normalization_metadata") or {}
    required = ["train_pixel_mean", "train_pixel_std"]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise KeyError(
            "Checkpoint missing normalization metadata required for Grad-CAM: "
            + ", ".join(missing)
        )
    return {
        "train_pixel_mean": float(metadata["train_pixel_mean"]),
        "train_pixel_std": float(metadata["train_pixel_std"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
