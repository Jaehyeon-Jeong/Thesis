"""Config-driven local/Kaggle runner for Stage 1 baseline training.

This runner connects the already implemented pieces:
- monthly_20d data loading,
- horizon labels and splits,
- train-only normalization,
- StockCNNI20,
- training loop and checkpoints.

Evaluation and Grad-CAM are separate scripts/gates, but this runner prepares
the checkpoints and metadata those later steps need.

How to read this file:
    This is the orchestration layer. It does not define how images are read or
    how the CNN computes logits. Instead, it calls the specialized modules in
    the correct order: data -> labels/splits -> normalization -> DataLoader ->
    model -> training -> manifest.
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
    """Runtime matrix selected by CLI/config.

    This object describes what to run now: which horizon(s), which seed(s),
    whether this is a tiny smoke run, and whether local full runs are allowed.
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
    """Run Stage 1 baseline training for the selected horizon/seed matrix.

    Output:
        A summary dictionary printed by `scripts/run_stage1_baseline.py`. The
        actual learned model is saved to checkpoint files, not returned here.
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

    ensure_stage1_output_dirs(paths)
    device = select_device(config)

    # `base_dataset` can read raw images. It does not yet know which future
    # return horizon is the label.
    base_dataset = build_dataset_from_config(config)

    # `base_metadata` is one DataFrame containing all label rows and row ids.
    # It does not contain image tensors.
    base_metadata = build_base_metadata(base_dataset.shards)
    split_settings = split_settings_from_config(config)
    normalization_settings = normalization_settings_from_config(config)
    training_settings_base = training_settings_from_config(config)

    run_results: list[dict[str, Any]] = []
    for horizon_name in selection.horizons:
        if horizon_name not in TARGET_COLUMNS:
            raise KeyError(f"Unknown horizon: {horizon_name}")

        # Convert one target return column, such as Ret_20d, into binary labels.
        horizon_frame = build_horizon_frame(base_metadata, horizon_name)

        # Add a `split` column: train, validation, or test.
        split_frame = assign_splits(horizon_frame, split_settings)
        split_summary = make_split_summary(split_frame, split_settings, horizon_name)

        # Compute train-only pixel mean/std for this horizon's train rows.
        # The resulting stats are reused by both training and validation data.
        normalization_stats = compute_pixel_normalization(
            dataset=base_dataset,
            split_frame=split_frame,
            settings=normalization_settings,
            target_return_name=TARGET_COLUMNS[horizon_name],
            max_images=selection.normalization_max_images,
        )
        horizon_metrics_dir = paths.metrics_root / horizon_name
        # Save split/normalization audit JSONs before training starts.
        write_horizon_metadata(
            output_dir=horizon_metrics_dir,
            split_summary=split_summary,
            normalization_stats=normalization_stats,
            split_frame=None,
            write_split_index=False,
        )

        for run_seed in selection.run_seeds:
            # Seed affects weight initialization and DataLoader train shuffling.
            set_global_seed(run_seed)
            training_settings = training_settings_base
            if selection.max_epochs is not None:
                from dataclasses import replace

                training_settings = replace(training_settings, max_epochs=selection.max_epochs)

            # DataLoaders convert row-level datasets into batches:
            #   images `(B, 1, 64, 60)`
            #   labels `(B,)`
            # Training consumes train_loader; validation consumes val_loader.
            train_loader, val_loader = _build_train_val_loaders(
                config=config,
                base_dataset=base_dataset,
                split_frame=split_frame,
                normalization_stats=normalization_stats,
                run_seed=run_seed,
                max_train_rows=selection.max_train_rows,
                max_val_rows=selection.max_val_rows,
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
            # `fit_model` trains one model for one horizon and one seed. It
            # writes `best.pt`, `last.pt`, and training history files.
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
    """Write `outputs/run_manifests/run_manifest.json`.

    The manifest is the run receipt. It records config, package versions, seed
    choices, paths, and produced checkpoint/metric locations so a run can be
    audited later.
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
    """Build train/validation dataloaders from split metadata.

    Output:
        train_loader batches dictionaries with image `(B,1,64,60)`, label `(B,)`,
        and metadata. The training loop only uses image and label.
    """

    runtime_config = get_config_section(config, "runtime")
    training_config = get_config_section(config, "training")
    dataloader_config = training_config["dataloader"]
    batch_size = int(training_config.get("batch_size", 128))
    num_workers = int(runtime_config.get("num_workers", 0))
    persistent_workers = bool(runtime_config.get("persistent_workers", False)) and num_workers > 0
    pin_memory = bool(runtime_config.get("pin_memory", False))

    # Dataset reads one normalized sample at a time. DataLoader below stacks
    # those samples into batch tensors.
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
    # Train loader shuffles rows because SGD benefits from random batch order.
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
    # Validation loader does not shuffle so validation outputs are deterministic
    # and easier to align with row metadata.
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
    """Return source metadata that should travel with manifests/checkpoints."""

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
    """Collect package versions without failing when a package is absent."""

    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _cuda_info() -> dict[str, Any]:
    """Return CUDA/GPU environment information."""

    available = bool(torch.cuda.is_available())
    return {
        "available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "device_name": torch.cuda.get_device_name(0) if available else None,
    }


def _git_commit(project_root: Path) -> str | None:
    """Return git commit for `project_root` when available."""

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
