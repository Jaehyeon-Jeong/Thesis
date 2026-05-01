"""1단계 baseline training을 위한 config 기반 local/Kaggle runner.

이 runner는 이미 구현된 조각들을 연결한다:
- monthly_20d data loading
- horizon label과 split
- train-only normalization
- StockCNNI20
- training loop와 checkpoint

Evaluation과 Grad-CAM은 별도 script/gate지만, 이 runner가 그 단계에 필요한
checkpoint와 metadata를 준비한다.

읽는 법:
    이 파일은 orchestration layer다. image를 어떻게 읽는지, CNN이 logits를 어떻게
    계산하는지는 여기서 정의하지 않는다. 대신 전문 module을 올바른 순서로 호출한다:
    data -> label/split -> normalization -> DataLoader -> model -> training -> manifest.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from stage1_reimage.config import get_config_section
from stage1_reimage.data import (
    TARGET_COLUMNS,
    HorizonSplitImageDataset,
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
from stage1_reimage.models import StockCNNI20
from stage1_reimage.paths import Stage1Paths, ensure_stage1_output_dirs
from stage1_reimage.runtime import select_device
from stage1_reimage.seed import set_global_seed
from stage1_reimage.training import fit_model, training_settings_from_config

RUN_MODES = ("smoke", "full_single_seed", "full_paper_style")


@dataclass(frozen=True)
class RunSelection:
    """CLI/config가 선택한 runtime matrix.

    지금 무엇을 실행할지 담는다: 어떤 horizon, 어떤 seed, tiny smoke run 여부,
    local full run 허용 여부.
    """

    run_mode: str
    horizons: tuple[str, ...]
    run_seeds: tuple[int, ...]
    max_train_rows: int | None
    max_val_rows: int | None
    normalization_max_images: int | None
    max_epochs: int | None
    allow_local_full: bool


def run_stage1_baseline(
    config: Mapping[str, Any],
    paths: Stage1Paths,
    selection: RunSelection,
) -> dict[str, Any]:
    """선택된 horizon/seed matrix에 대해 1단계 baseline training을 실행한다.

    출력:
        `scripts/run_stage1_baseline.py`가 출력하는 summary dict. 실제 학습된 model은
        여기서 return되지 않고 checkpoint file로 저장된다.
    """

    if selection.run_mode not in RUN_MODES:
        raise ValueError(f"Unsupported run mode: {selection.run_mode}")

    environment_config = get_config_section(config, "environment")
    is_full_target = bool(environment_config.get("full_run_target", False))
    if selection.run_mode != "smoke" and not is_full_target and not selection.allow_local_full:
        raise RuntimeError(
            "Refusing to run non-smoke mode on a non-full-run environment. "
            "Pass --allow-local-full only if this is intentional."
        )

    print(
        f"[stage1] run_mode={selection.run_mode} "
        f"horizons={list(selection.horizons)} seeds={list(selection.run_seeds)}",
        flush=True,
    )
    ensure_stage1_output_dirs(paths)
    device = select_device(config)
    print(f"[stage1] device={device}", flush=True)

    # `base_dataset`은 raw image를 읽을 수 있다. 아직 어떤 future return horizon을
    # label로 쓸지는 모른다.
    print("[stage1] building monthly_20d dataset", flush=True)
    base_dataset = build_dataset_from_config(config)
    print(
        f"[stage1] dataset ready shards={len(base_dataset.shards)} "
        f"rows={len(base_dataset):,}",
        flush=True,
    )

    # `base_metadata`는 모든 label row와 row id를 담은 하나의 DataFrame이다.
    # image tensor는 포함하지 않는다.
    print("[stage1] building base metadata", flush=True)
    base_metadata = build_base_metadata(base_dataset.shards)
    print(f"[stage1] base metadata rows={len(base_metadata):,}", flush=True)
    split_settings = split_settings_from_config(config)
    normalization_settings = normalization_settings_from_config(config)
    training_settings_base = training_settings_from_config(config)

    run_results: list[dict[str, Any]] = []
    for horizon_name in selection.horizons:
        if horizon_name not in TARGET_COLUMNS:
            raise KeyError(f"Unknown horizon: {horizon_name}")

        print(f"[stage1:{horizon_name}] start", flush=True)
        # Ret_20d 같은 target return column 하나를 binary label로 변환한다.
        horizon_frame = build_horizon_frame(base_metadata, horizon_name)
        print(
            f"[stage1:{horizon_name}] labeled rows={len(horizon_frame):,} "
            f"target={TARGET_COLUMNS[horizon_name]}",
            flush=True,
        )

        # `split` column을 추가한다: train, validation, test 중 하나.
        split_frame = assign_splits(horizon_frame, split_settings)
        split_summary = make_split_summary(split_frame, split_settings, horizon_name)
        split_counts = split_frame["split"].value_counts().to_dict()
        print(f"[stage1:{horizon_name}] split counts={split_counts}", flush=True)

        # 이 horizon의 train row에서만 pixel mean/std를 계산한다.
        # 계산된 stats는 training과 validation data 모두에서 재사용된다.
        normalization_stats = compute_pixel_normalization(
            dataset=base_dataset,
            split_frame=split_frame,
            settings=normalization_settings,
            target_return_name=TARGET_COLUMNS[horizon_name],
            max_images=selection.normalization_max_images,
            progress_label=horizon_name,
        )
        horizon_metrics_dir = paths.metrics_root / horizon_name
        # training 시작 전에 split/normalization audit JSON을 저장한다.
        write_horizon_metadata(
            output_dir=horizon_metrics_dir,
            split_summary=split_summary,
            normalization_stats=normalization_stats,
            split_frame=None,
            write_split_index=False,
        )

        for run_seed in selection.run_seeds:
            # seed는 weight initialization과 DataLoader train shuffling에 영향을 준다.
            print(f"[stage1:{horizon_name}:seed_{run_seed}] training start", flush=True)
            set_global_seed(run_seed)
            training_settings = training_settings_base
            if selection.max_epochs is not None:
                from dataclasses import replace

                training_settings = replace(training_settings, max_epochs=selection.max_epochs)

            # DataLoader는 row-level dataset을 batch로 바꾼다:
            #   images `(B, 1, 64, 60)`
            #   labels `(B,)`
            # training은 train_loader, validation은 val_loader를 사용한다.
            train_loader, val_loader = _build_train_val_loaders(
                config=config,
                base_dataset=base_dataset,
                split_frame=split_frame,
                normalization_stats=normalization_stats,
                run_seed=run_seed,
                max_train_rows=selection.max_train_rows,
                max_val_rows=selection.max_val_rows,
            )
            print(
                f"[stage1:{horizon_name}:seed_{run_seed}] "
                f"train_rows={len(train_loader.dataset):,} "
                f"val_rows={len(val_loader.dataset):,} "
                f"train_batches={len(train_loader):,} "
                f"val_batches={len(val_loader):,}",
                flush=True,
            )
            checkpoint_dir = paths.checkpoint_root / horizon_name / f"seed_{run_seed}"
            metrics_dir = paths.metrics_root / horizon_name / f"seed_{run_seed}"
            run_context = {
                "experiment_name": horizon_name,
                "target_return_name": TARGET_COLUMNS[horizon_name],
                "run_mode": selection.run_mode,
                "report_as_reproduction": selection.run_mode != "smoke",
                "run_seed": run_seed,
                "split_seed": split_settings.split_seed,
            }
            # `fit_model`은 horizon 하나와 seed 하나에 대해 model 하나를 학습한다.
            # `best.pt`, `last.pt`, training history file을 저장한다.
            result = fit_model(
                model=StockCNNI20(),
                train_loader=train_loader,
                val_loader=val_loader,
                settings=training_settings,
                device=device,
                checkpoint_dir=checkpoint_dir,
                metrics_dir=metrics_dir,
                run_context=run_context,
                config_snapshot=config,
                normalization_metadata=normalization_stats.as_dict(),
                source_reference_metadata=_source_reference_metadata(config),
            )
            print(
                f"[stage1:{horizon_name}:seed_{run_seed}] training done "
                f"best_epoch={result.best_epoch} "
                f"best_val_loss={result.best_val_loss:.6f}",
                flush=True,
            )
            run_results.append(
                {
                    "horizon_name": horizon_name,
                    "target_return_name": TARGET_COLUMNS[horizon_name],
                    "run_seed": run_seed,
                    "result": result.as_dict(),
                    "train_rows_used": len(train_loader.dataset),
                    "validation_rows_used": len(val_loader.dataset),
                    "normalization": normalization_stats.as_dict(),
                }
            )

    manifest = write_run_manifest(
        config=config,
        paths=paths,
        selection=selection,
        run_results=run_results,
        device=device,
    )
    print(f"[stage1] run manifest written: {manifest}", flush=True)
    return {
        "status": "ok",
        "run_mode": selection.run_mode,
        "device": device,
        "horizons": list(selection.horizons),
        "run_seeds": list(selection.run_seeds),
        "run_results": run_results,
        "run_manifest": manifest,
    }


def write_run_manifest(
    config: Mapping[str, Any],
    paths: Stage1Paths,
    selection: RunSelection,
    run_results: Sequence[Mapping[str, Any]],
    device: str,
) -> str:
    """`outputs/run_manifests/run_manifest.json`을 저장한다.

    manifest는 run receipt다. config, package version, seed 선택, path, 생성된
    checkpoint/metric 위치를 기록해 나중에 run을 audit할 수 있게 한다.
    """

    manifest_path = paths.run_manifest_root / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_mode": selection.run_mode,
        "horizons": list(selection.horizons),
        "run_seeds": list(selection.run_seeds),
        "max_train_rows": selection.max_train_rows,
        "max_val_rows": selection.max_val_rows,
        "normalization_max_images": selection.normalization_max_images,
        "device": device,
        "environment": dict(get_config_section(config, "environment")),
        "paths": paths.as_dict(),
        "data": dict(get_config_section(config, "data")),
        "model": dict(get_config_section(config, "model")),
        "split": dict(get_config_section(config, "split")),
        "training": dict(get_config_section(config, "training")),
        "python": {
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "packages": _package_versions(["torch", "numpy", "pandas", "pyarrow", "sklearn"]),
        "cuda": _cuda_info(),
        "git_commit": _git_commit(paths.project_root),
        "source_references": _source_reference_metadata(config),
        "run_results": list(run_results),
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    return str(manifest_path)


def _build_train_val_loaders(
    config: Mapping[str, Any],
    base_dataset: Any,
    split_frame: Any,
    normalization_stats: Any,
    run_seed: int,
    max_train_rows: int | None,
    max_val_rows: int | None,
) -> tuple[DataLoader, DataLoader]:
    """split metadata에서 train/validation DataLoader를 만든다.

    출력:
        train_loader는 image `(B,1,64,60)`, label `(B,)`, metadata가 들어 있는
        dictionary batch를 만든다. training loop는 image와 label만 사용한다.
    """

    runtime_config = get_config_section(config, "runtime")
    training_config = get_config_section(config, "training")
    dataloader_config = training_config["dataloader"]
    batch_size = int(training_config.get("batch_size", 128))
    num_workers = int(runtime_config.get("num_workers", 0))
    persistent_workers = bool(runtime_config.get("persistent_workers", False)) and num_workers > 0
    pin_memory = bool(runtime_config.get("pin_memory", False))

    # Dataset은 normalized sample을 하나씩 읽는다. 아래 DataLoader가 그 sample을
    # batch tensor로 stack한다.
    train_dataset = HorizonSplitImageDataset(
        base_dataset=base_dataset,
        split_frame=split_frame,
        split_name="train",
        normalization_stats=normalization_stats,
        max_rows=max_train_rows,
    )
    val_dataset = HorizonSplitImageDataset(
        base_dataset=base_dataset,
        split_frame=split_frame,
        split_name="validation",
        normalization_stats=normalization_stats,
        max_rows=max_val_rows,
    )
    # SGD는 random batch order에서 이점이 있으므로 train loader는 row를 shuffle한다.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool(dataloader_config.get("train_shuffle", True)),
        drop_last=bool(dataloader_config.get("drop_last", False)),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=torch.Generator().manual_seed(run_seed),
    )
    # validation output을 deterministic하게 유지하고 row metadata와 맞추기 쉽도록
    # validation loader는 shuffle하지 않는다.
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=bool(dataloader_config.get("eval_shuffle", False)),
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def _source_reference_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """manifest/checkpoint에 같이 기록할 source metadata를 반환한다."""

    model_config = get_config_section(config, "model")
    return {
        "reimage_summary": "자료조사/Re-image 요약.md",
        "reimage_pdf": "자료조사/Xiu-Re-Imagining-Price-Trends.pdf",
        "stock_cnn_reference_repo": model_config.get("reference_repo"),
        "stock_cnn_reference_commit": model_config.get("reference_commit"),
        "stock_cnn_reference_file": model_config.get("reference_file"),
        "paper_training_source_note": "Local Re-image summary maps CNN/training details to pp. 12-21.",
        "gradcam_future_source_note": "Grad-CAM sources are checked later before 1-I8 implementation.",
    }


def _package_versions(package_names: Sequence[str]) -> dict[str, str | None]:
    """package가 없어도 실패하지 않고 package version을 수집한다."""

    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _cuda_info() -> dict[str, Any]:
    """CUDA/GPU 환경 정보를 반환한다."""

    available = bool(torch.cuda.is_available())
    return {
        "available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "device_name": torch.cuda.get_device_name(0) if available else None,
    }


def _git_commit(project_root: Path) -> str | None:
    """가능하면 `project_root`의 git commit hash를 반환한다."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
