#!/usr/bin/env python3
"""1단계 training loop와 checkpoint writing을 smoke check한다."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticPriceImageDataset(Dataset):
    """training-loop smoke check용 작은 deterministic image dataset.

    이 dataset은 실제 training batch interface를 흉내 낸다:
        image: `(1,64,60)` tensor
        label: scalar class id 0 or 1
    real `.dat` file을 읽지 않으면서도 training loop를 실제로 통과시킨다.
    """

    def __init__(self, num_samples: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        # batching 이후 real DataLoader sample과 같은 shape가 된다:
        # `(batch_size, 1, 64, 60)`.
        self.images = torch.rand(num_samples, 1, 64, 60, generator=generator)
        # 실제 future-return label을 쓰지 않기 위해 image 평균값으로 deterministic한
        # synthetic label을 만든다. 이 label은 smoke test 구조 확인용이다.
        self.labels = (self.images.mean(dim=(1, 2, 3)) > 0.5).long()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index], "label": self.labels[index]}


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
    parser.add_argument("--experiment-name", default="_smoke_training")
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--val-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    """synthetic training-loop smoke check를 실행한다.

    `fit_model()`이 forward/backward pass를 실행하고, weight를 업데이트하고,
    checkpoint를 저장하고, history/metadata file을 쓸 수 있는지 확인한다.
    """

    stage_root = add_stage1_src_to_path()
    args = parse_args(stage_root)

    from stage1_reimage.config import load_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.models import StockCNNI20  # pylint: disable=import-outside-toplevel
    from stage1_reimage.paths import (  # pylint: disable=import-outside-toplevel
        build_stage1_paths,
        ensure_stage1_output_dirs,
    )
    from stage1_reimage.runtime import select_device  # pylint: disable=import-outside-toplevel
    from stage1_reimage.seed import set_global_seed  # pylint: disable=import-outside-toplevel
    from stage1_reimage.training import (  # pylint: disable=import-outside-toplevel
        fit_model,
        training_settings_from_config,
    )

    config = load_config(args.config)
    paths = build_stage1_paths(config)
    ensure_stage1_output_dirs(paths)
    set_global_seed(args.run_seed)

    base_settings = training_settings_from_config(config)
    settings = replace(
        base_settings,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        log_every_batches=0,
    )
    device = select_device(config)

    # 여기 DataLoader는 실제 training과 같은 batch 구조를 갖지만 image/label은 작고
    # synthetic이다.
    train_loader = DataLoader(
        SyntheticPriceImageDataset(args.train_samples, seed=args.run_seed),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.run_seed),
    )
    val_loader = DataLoader(
        SyntheticPriceImageDataset(args.val_samples, seed=args.run_seed + 1),
        batch_size=args.batch_size,
        shuffle=False,
    )

    run_context = {
        "experiment_name": args.experiment_name,
        "target_return_name": "synthetic",
        "run_mode": "smoke",
        "report_as_reproduction": False,
        "run_seed": args.run_seed,
        "split_seed": None,
    }
    # 작은 model 하나를 학습한다. output path는 비싼 real dataset을 돌리기 전에
    # checkpoint/history writing이 동작한다는 것을 보여준다.
    result = fit_model(
        model=StockCNNI20(),
        train_loader=train_loader,
        val_loader=val_loader,
        settings=settings,
        device=device,
        checkpoint_dir=paths.checkpoint_root / args.experiment_name / f"seed_{args.run_seed}",
        metrics_dir=paths.metrics_root / args.experiment_name / f"seed_{args.run_seed}",
        run_context=run_context,
        config_snapshot=config,
        normalization_metadata={"smoke": True},
        source_reference_metadata={
            "reference_repo": config["model"]["reference_repo"],
            "reference_commit": config["model"]["reference_commit"],
            "paper_source_note": "Re-image summary maps training details to pp. 12-21.",
        },
    )

    summary = {
        "status": "ok",
        "device": device,
        "result": result.as_dict(),
        "best_checkpoint_exists": Path(result.best_checkpoint_path).exists(),
        "last_checkpoint_exists": Path(result.last_checkpoint_path).exists(),
        "history_exists": Path(result.train_history_path).exists(),
        "metadata_exists": Path(result.train_metadata_path).exists(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
