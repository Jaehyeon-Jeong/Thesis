#!/usr/bin/env python3
"""1단계 prediction과 evaluation metric을 export한다.

실행 예시:
    `run_stage1_baseline.py` 이후 local smoke evaluation:
        python scripts/evaluate_stage1_predictions.py \
          --config configs/env_local.yaml \
          --horizon stage1_i20_r20 \
          --run-seed 42 \
          --split validation \
          --max-rows 4

    Kaggle 전체 test evaluation:
        python scripts/evaluate_stage1_predictions.py \
          --config configs/env_kaggle.yaml \
          --horizon stage1_i20_r20 \
          --run-seed 42 \
          --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def add_stage1_src_to_path() -> Path:
    """로컬 1단계 `src/` directory를 `sys.path`에 추가한다.

    script를 직접 실행할 때 local source import가 가능하게 한다:
    `from stage1_reimage.evaluation import predict_loader`.
    """

    stage_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage_root / "src"))
    return stage_root


def parse_args(stage_root: Path) -> argparse.Namespace:
    """명령행 인자를 parsing한다.

    중요한 mode:
        normal mode는 checkpoint 하나를 load하고 seed-level prediction을 저장한다.
        `--average-seed-predictions`는 model loading을 건너뛰고 기존 prediction CSV를
        평균한다.
    """

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
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="smoke evaluation용 optional row 제한. 재현 metric 계산에는 사용하지 않는다.",
    )
    parser.add_argument(
        "--normalization-max-images",
        type=int,
        default=None,
        help="checkpoint metadata가 없을 때만 쓰는 fallback normalization image 제한.",
    )
    parser.add_argument(
        "--average-seed-predictions",
        nargs="*",
        type=int,
        default=None,
        help="선택한 horizon/split의 기존 seed prediction CSV를 평균한다.",
    )
    return parser.parse_args()


def main() -> int:
    """seed-level prediction export 또는 기존 seed prediction 평균을 실행한다.

    Seed-level 흐름:
        checkpoint -> model -> DataLoader -> prediction DataFrame -> CSV/JSON.

    Seed 평균 흐름:
        seed prediction CSVs -> mean probabilities -> averaged CSV/JSON.
    """

    stage_root = add_stage1_src_to_path()
    args = parse_args(stage_root)

    from stage1_reimage.config import load_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.data import (  # pylint: disable=import-outside-toplevel
        HORIZON_SPECS,
        TARGET_COLUMNS,
        HorizonSplitImageDataset,
        PixelNormalizationStats,
        assign_splits,
        build_base_metadata,
        build_dataset_from_config,
        build_horizon_frame,
        compute_pixel_normalization,
        normalization_settings_from_config,
        split_settings_from_config,
    )
    from stage1_reimage.evaluation import (  # pylint: disable=import-outside-toplevel
        average_seed_predictions,
        compute_classification_metrics,
        compute_correlation_metrics,
        evaluation_settings_from_config,
        load_checkpoint_into_model,
        predict_loader,
        write_evaluation_outputs,
    )
    from stage1_reimage.models import StockCNNI20  # pylint: disable=import-outside-toplevel
    from stage1_reimage.paths import (  # pylint: disable=import-outside-toplevel
        build_stage1_paths,
        ensure_stage1_output_dirs,
    )
    from stage1_reimage.runtime import select_device  # pylint: disable=import-outside-toplevel

    # Config는 data path, device, split setting, evaluation rule을 제어한다.
    config = load_config(args.config)
    paths = build_stage1_paths(config)
    ensure_stage1_output_dirs(paths)
    settings = evaluation_settings_from_config(config)

    if args.average_seed_predictions is not None:
        # 이 branch는 CNN을 실행하지 않는다. 기존 seed prediction CSV를 읽고 softmax
        # probability만 평균한다.
        summary = _run_average_predictions(
            paths=paths,
            horizon=args.horizon,
            split_name=args.split,
            run_seeds=args.average_seed_predictions,
            settings=settings,
            compute_classification_metrics=compute_classification_metrics,
            compute_correlation_metrics=compute_correlation_metrics,
            write_evaluation_outputs=write_evaluation_outputs,
            average_seed_predictions=average_seed_predictions,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.horizon not in TARGET_COLUMNS:
        raise KeyError(f"Unknown horizon: {args.horizon}")

    # 기본적으로 evaluation은 `run_stage1_baseline.py`가 만든 best validation-loss
    # checkpoint를 사용한다.
    checkpoint_path = args.checkpoint_path or (
        paths.checkpoint_root / args.horizon / f"seed_{args.run_seed}" / "best.pt"
    )
    device = select_device(config)

    # 먼저 빈 model 객체를 만들고, checkpoint에서 학습된 weight를 load한다.
    model = StockCNNI20()
    checkpoint = load_checkpoint_into_model(model, checkpoint_path, device)

    # training은 image normalization에 사용한 정확한 mean/std를 저장했다. 이것을
    # 재사용해야 evaluation data가 training 중 validation과 같은 방식으로 변환된다.
    normalization_stats = _normalization_stats_from_checkpoint(
        checkpoint=checkpoint,
        target_return_name=TARGET_COLUMNS[args.horizon],
    )

    # training에서 사용한 row index를 다시 만들어 prediction row가 원본 Date/StockID/
    # return metadata와 align되게 한다.
    base_dataset = build_dataset_from_config(config)
    base_metadata = build_base_metadata(base_dataset.shards)
    horizon_frame = build_horizon_frame(base_metadata, args.horizon)
    split_frame = assign_splits(horizon_frame, split_settings_from_config(config))
    if normalization_stats is None:
        normalization_stats = compute_pixel_normalization(
            dataset=base_dataset,
            split_frame=split_frame,
            settings=normalization_settings_from_config(config),
            target_return_name=TARGET_COLUMNS[args.horizon],
            max_images=args.normalization_max_images,
        )

    # evaluation dataset은 normalized image `(1,64,60)`, label, metadata를 반환한다.
    # DataLoader는 이를 `(B,1,64,60)` batch로 stack한다.
    dataset = HorizonSplitImageDataset(
        base_dataset=base_dataset,
        split_frame=split_frame,
        split_name=args.split,
        normalization_stats=normalization_stats,
        max_rows=args.max_rows,
    )
    loader = _build_eval_loader(config=config, dataset=dataset, batch_size=settings.batch_size)
    horizon_spec = HORIZON_SPECS[args.horizon]
    # 선택된 split에 checkpoint를 적용하고 image 하나당 prediction row 하나를 만든다.
    predictions = predict_loader(
        model=model,
        data_loader=loader,
        checkpoint_path=checkpoint_path,
        experiment_name=args.horizon,
        image_window=horizon_spec["image_window"],
        target_horizon=_target_horizon_from_name(args.horizon),
        run_seed=args.run_seed,
        split_name=args.split,
        settings=settings,
        device=device,
    )
    # metric은 model에서 직접 계산하지 않고 prediction DataFrame에서 계산한다. 이렇게
    # 해야 저장된 CSV와 metric을 나중에 audit할 수 있다.
    classification_metrics = compute_classification_metrics(predictions)
    correlation_metrics = compute_correlation_metrics(
        predictions,
        min_group_size=settings.min_correlation_group_size,
    )
    # CSV/JSON file을 outputs/predictions와 outputs/metrics 아래에 저장한다.
    written = write_evaluation_outputs(
        predictions=predictions,
        classification_metrics=classification_metrics,
        correlation_metrics=correlation_metrics,
        predictions_dir=paths.predictions_root / args.horizon / f"seed_{args.run_seed}",
        metrics_dir=paths.metrics_root / args.horizon / f"seed_{args.run_seed}",
        split_name=args.split,
    )
    summary = {
        "status": "ok",
        "horizon": args.horizon,
        "split": args.split,
        "run_seed": args.run_seed,
        "checkpoint_path": str(checkpoint_path),
        "num_predictions": int(len(predictions)),
        "accuracy": classification_metrics["accuracy"],
        "positive_rate": classification_metrics["positive_rate"],
        "predicted_positive_rate": classification_metrics["predicted_positive_rate"],
        "written": written,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _build_eval_loader(
    config: dict[str, Any],
    dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> DataLoader:
    """row 순서가 고정된 evaluation DataLoader를 만든다.

    prediction CSV row가 metadata와 맞아야 하고 나중 seed averaging에도 쓰이므로
    evaluation은 row order를 안정적으로 유지해야 한다.
    """

    from stage1_reimage.config import get_config_section  # pylint: disable=import-outside-toplevel

    runtime_config = get_config_section(config, "runtime")
    num_workers = int(runtime_config.get("num_workers", 0))
    persistent_workers = bool(runtime_config.get("persistent_workers", False)) and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=bool(runtime_config.get("pin_memory", False)),
        persistent_workers=persistent_workers,
    )


def _normalization_stats_from_checkpoint(
    checkpoint: dict[str, Any],
    target_return_name: str,
) -> Any:
    """가능하면 checkpoint metadata에서 normalization stats를 복원한다.

    출력:
        `HorizonSplitImageDataset`이 사용하는 `PixelNormalizationStats`. 오래된
        checkpoint가 normalization metadata를 저장하지 않았다면 `None`.
    """

    from stage1_reimage.data import PixelNormalizationStats  # pylint: disable=import-outside-toplevel

    metadata = checkpoint.get("normalization_metadata") or {}
    required = [
        "train_pixel_mean",
        "train_pixel_std",
        "pixel_scale",
        "epsilon",
        "num_train_images_available",
        "num_train_images_used",
        "num_pixels_used",
        "sampled_for_smoke",
    ]
    if not all(key in metadata for key in required):
        return None
    return PixelNormalizationStats(
        target_return_name=str(metadata.get("target_return_name", target_return_name)),
        train_pixel_mean=float(metadata["train_pixel_mean"]),
        train_pixel_std=float(metadata["train_pixel_std"]),
        pixel_scale=float(metadata["pixel_scale"]),
        epsilon=float(metadata["epsilon"]),
        num_train_images_available=int(metadata["num_train_images_available"]),
        num_train_images_used=int(metadata["num_train_images_used"]),
        num_pixels_used=int(metadata["num_pixels_used"]),
        sampled_for_smoke=bool(metadata["sampled_for_smoke"]),
    )


def _target_horizon_from_name(horizon_name: str) -> str:
    """prediction metadata용으로 `stage1_i20_r20`을 `R20` 표기로 바꾼다."""

    return horizon_name.rsplit("_", maxsplit=1)[-1].upper()


def _run_average_predictions(
    paths: Any,
    horizon: str,
    split_name: str,
    run_seeds: list[int],
    settings: Any,
    compute_classification_metrics: Any,
    compute_correlation_metrics: Any,
    write_evaluation_outputs: Any,
    average_seed_predictions: Any,
) -> dict[str, Any]:
    """이미 저장된 seed prediction file을 평균한다.

    5회 independent training 이후 사용한다. 모든 seed file이 같은 row를 설명하는지
    확인한 뒤 `prob_up`을 평균하고 metric을 다시 계산한다.
    """

    # 기대하는 file pattern:
    # outputs/predictions/<horizon>/seed_<seed>/<split>_predictions.csv
    prediction_paths = [
        paths.predictions_root / horizon / f"seed_{run_seed}" / f"{split_name}_predictions.csv"
        for run_seed in run_seeds
    ]
    missing = [str(path) for path in prediction_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing seed prediction file(s): {missing}")

    averaged = average_seed_predictions(
        prediction_paths=prediction_paths,
        run_seeds=run_seeds,
        settings=settings,
    )
    classification_metrics = compute_classification_metrics(
        averaged,
        probability_column="mean_prob_up",
    )
    correlation_metrics = compute_correlation_metrics(
        averaged,
        probability_column="mean_prob_up",
        min_group_size=settings.min_correlation_group_size,
    )
    written = write_evaluation_outputs(
        predictions=averaged,
        classification_metrics=classification_metrics,
        correlation_metrics=correlation_metrics,
        predictions_dir=paths.predictions_root / horizon / "averaged",
        metrics_dir=paths.metrics_root / horizon / "averaged",
        split_name=split_name,
    )
    return {
        "status": "ok",
        "mode": "average_seed_predictions",
        "horizon": horizon,
        "split": split_name,
        "run_seeds": run_seeds,
        "num_predictions": int(len(averaged)),
        "accuracy": classification_metrics["accuracy"],
        "positive_rate": classification_metrics["positive_rate"],
        "predicted_positive_rate": classification_metrics["predicted_positive_rate"],
        "written": written,
    }


if __name__ == "__main__":
    raise SystemExit(main())
