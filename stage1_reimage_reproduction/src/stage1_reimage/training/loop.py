"""Training loop and checkpoint utilities for Stage 1.

Paper/source context:
    The local Re-image summary reports cross-entropy loss, Adam with learning
    rate 1e-5, batch size 128, Xavier initialization, dropout 0.5, and early
    stopping after validation loss fails to improve for 2 epochs. Exact seeds,
    epoch count, Adam betas/eps, weight decay, and mixed precision are not
    reported, so these values are explicit implementation choices recorded in
    config and metadata.

Scope:
    This module implements the generic training loop. It does not implement
    final prediction CSVs, evaluation metrics beyond loss/accuracy, portfolio
    results, or Grad-CAM.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from stage1_reimage.config import get_config_section


@dataclass(frozen=True)
class OptimizerSettings:
    """Adam optimizer settings for Stage 1."""

    name: str
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclass(frozen=True)
class EarlyStoppingSettings:
    """Validation-loss early stopping settings."""

    monitor: str
    mode: str
    patience: int
    min_delta: float
    restore_best: bool


@dataclass(frozen=True)
class TrainingSettings:
    """Stage 1 training settings parsed from config."""

    run_mode: str
    report_as_reproduction: bool
    batch_size: int
    max_epochs: int
    loss: str
    optimizer: OptimizerSettings
    initialization_name: str
    early_stopping: EarlyStoppingSettings
    mixed_precision: bool
    gradient_clipping: float | None
    log_every_batches: int


@dataclass(frozen=True)
class TrainingResult:
    """Summary returned by `fit_model`."""

    best_epoch: int
    best_val_loss: float
    stopped_epoch: int
    stopped_early: bool
    best_checkpoint_path: str
    last_checkpoint_path: str
    train_history_path: str
    train_metadata_path: str

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable result metadata."""

        return asdict(self)


def training_settings_from_config(config: Mapping[str, Any]) -> TrainingSettings:
    """Build `TrainingSettings` from the `training` config section."""

    training_config = get_config_section(config, "training")
    optimizer_config = training_config["optimizer"]
    early_stopping_config = training_config["early_stopping"]
    gradient_clipping = training_config.get("gradient_clipping")
    return TrainingSettings(
        run_mode=str(training_config.get("run_mode", "full")),
        report_as_reproduction=bool(training_config.get("report_as_reproduction", True)),
        batch_size=int(training_config.get("batch_size", 128)),
        max_epochs=int(training_config.get("max_epochs", 100)),
        loss=str(training_config.get("loss", "cross_entropy")),
        optimizer=OptimizerSettings(
            name=str(optimizer_config.get("name", "adam")),
            learning_rate=float(optimizer_config.get("learning_rate", 1.0e-5)),
            betas=(
                float(optimizer_config.get("betas", [0.9, 0.999])[0]),
                float(optimizer_config.get("betas", [0.9, 0.999])[1]),
            ),
            eps=float(optimizer_config.get("eps", 1.0e-8)),
            weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        ),
        initialization_name=str(
            training_config.get("initialization", {}).get("name", "xavier_uniform")
        ),
        early_stopping=EarlyStoppingSettings(
            monitor=str(early_stopping_config.get("monitor", "val_loss")),
            mode=str(early_stopping_config.get("mode", "min")),
            patience=int(early_stopping_config.get("patience", 2)),
            min_delta=float(early_stopping_config.get("min_delta", 0.0)),
            restore_best=bool(early_stopping_config.get("restore_best", True)),
        ),
        mixed_precision=bool(training_config.get("mixed_precision", False)),
        gradient_clipping=None if gradient_clipping is None else float(gradient_clipping),
        log_every_batches=int(training_config.get("log_every_batches", 100)),
    )


def initialize_model_weights(model: nn.Module, variant: str = "xavier_uniform") -> None:
    """Apply Stage 1 Xavier initialization in-place."""

    if variant != "xavier_uniform":
        raise ValueError(f"Unsupported initialization variant: {variant}")
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def build_loss(settings: TrainingSettings) -> nn.Module:
    """Build the configured training loss."""

    if settings.loss != "cross_entropy":
        raise ValueError(f"Unsupported loss: {settings.loss}")
    return nn.CrossEntropyLoss()


def build_optimizer(model: nn.Module, settings: TrainingSettings) -> torch.optim.Optimizer:
    """Build the configured optimizer."""

    if settings.optimizer.name.lower() != "adam":
        raise ValueError(f"Unsupported optimizer: {settings.optimizer.name}")
    return torch.optim.Adam(
        model.parameters(),
        lr=settings.optimizer.learning_rate,
        betas=settings.optimizer.betas,
        eps=settings.optimizer.eps,
        weight_decay=settings.optimizer.weight_decay,
    )


def fit_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    settings: TrainingSettings,
    device: torch.device | str,
    checkpoint_dir: str | Path,
    metrics_dir: str | Path,
    run_context: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    normalization_metadata: Mapping[str, Any] | None = None,
    source_reference_metadata: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train a model and write best/last checkpoints plus history/metadata."""

    device = torch.device(device)
    checkpoint_path = Path(checkpoint_dir)
    metrics_path = Path(metrics_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    initialize_model_weights(model, settings.initialization_name)
    model.to(device)
    loss_fn = build_loss(settings)
    optimizer = build_optimizer(model, settings)

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    start_time = _utc_now()

    for epoch in range(1, settings.max_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = _run_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            train=True,
            settings=settings,
        )
        val_loss, val_accuracy = _run_epoch(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=None,
            device=device,
            train=False,
            settings=settings,
        )

        improved = val_loss < (best_val_loss - settings.early_stopping.min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_path / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                run_context=run_context,
                config_snapshot=config_snapshot,
                normalization_metadata=normalization_metadata,
                source_reference_metadata=source_reference_metadata,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_start,
                "best_so_far": improved,
            }
        )

        if epochs_without_improvement >= settings.early_stopping.patience:
            stopped_early = True
            break

    stopped_epoch = history[-1]["epoch"] if history else 0
    _save_checkpoint(
        checkpoint_path / "last.pt",
        model=model,
        optimizer=optimizer,
        epoch=stopped_epoch,
        best_val_loss=best_val_loss,
        run_context=run_context,
        config_snapshot=config_snapshot,
        normalization_metadata=normalization_metadata,
        source_reference_metadata=source_reference_metadata,
    )

    train_history_path = metrics_path / "train_history.csv"
    train_metadata_path = metrics_path / "train_metadata.json"
    _write_history_csv(train_history_path, history)
    _write_json(
        train_metadata_path,
        {
            "run_context": dict(run_context),
            "settings": _settings_as_dict(settings),
            "normalization_metadata": dict(normalization_metadata or {}),
            "source_reference_metadata": dict(source_reference_metadata or {}),
            "start_time_utc": start_time,
            "end_time_utc": _utc_now(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "stopped_epoch": stopped_epoch,
            "stopped_early": stopped_early,
            "checkpoint_dir": str(checkpoint_path),
            "metrics_dir": str(metrics_path),
        },
    )

    return TrainingResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        stopped_epoch=stopped_epoch,
        stopped_early=stopped_early,
        best_checkpoint_path=str(checkpoint_path / "best.pt"),
        last_checkpoint_path=str(checkpoint_path / "last.pt"),
        train_history_path=str(train_history_path),
        train_metadata_path=str(train_metadata_path),
    )


def _run_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    settings: TrainingSettings,
) -> tuple[float, float]:
    """Run one train or validation epoch and return `(loss, accuracy)`."""

    model.train(mode=train)
    loss_total = 0.0
    correct_total = 0
    sample_total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(data_loader, start=1):
            images, labels = _unpack_batch(batch)
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if train and optimizer is not None:
                loss.backward()
                if settings.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clipping)
                optimizer.step()

            batch_size = int(labels.shape[0])
            loss_total += float(loss.detach().cpu().item()) * batch_size
            predictions = torch.argmax(logits.detach(), dim=1)
            correct_total += int((predictions == labels).sum().cpu().item())
            sample_total += batch_size

            should_log = (
                train
                and settings.log_every_batches > 0
                and batch_index % settings.log_every_batches == 0
            )
            if should_log:
                print(f"batch={batch_index} loss={float(loss.detach().cpu().item()):.6f}")

    if sample_total == 0:
        raise ValueError("DataLoader produced zero samples.")
    return loss_total / sample_total, correct_total / sample_total


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract image and label tensors from common batch formats."""

    if isinstance(batch, Mapping):
        image = batch.get("image")
        label = batch.get("label")
        if label is None and "labels" in batch:
            label = batch["labels"]
        if image is None or label is None:
            raise KeyError("Batch mapping must contain `image` and `label` keys.")
        return image, label
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("Unsupported batch format for training loop.")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    run_context: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    normalization_metadata: Mapping[str, Any] | None,
    source_reference_metadata: Mapping[str, Any] | None,
) -> None:
    """Write a model checkpoint."""

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "run_context": dict(run_context),
            "config_snapshot": dict(config_snapshot),
            "normalization_metadata": dict(normalization_metadata or {}),
            "source_reference_metadata": dict(source_reference_metadata or {}),
        },
        path,
    )


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    """Write per-epoch training history."""

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "learning_rate",
        "epoch_seconds",
        "best_so_far",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write UTF-8 JSON with stable formatting."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _settings_as_dict(settings: TrainingSettings) -> dict[str, Any]:
    """Convert nested training settings to plain dictionaries."""

    payload = asdict(settings)
    payload["optimizer"]["betas"] = list(payload["optimizer"]["betas"])
    return payload


def _utc_now() -> str:
    """Return current UTC time as an ISO string."""

    return datetime.now(timezone.utc).isoformat()
