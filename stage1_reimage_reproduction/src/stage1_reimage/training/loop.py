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

How to read this file:
    `fit_model()` is the main function. It receives a model and DataLoaders,
    repeatedly calls `_run_epoch()`, saves `best.pt` when validation loss
    improves, and writes training history files for later review.
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
    """Stage 1 training settings parsed from config.

    This object freezes training hyperparameters after reading YAML, so the
    training loop uses explicit values instead of repeatedly indexing config.
    """

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
    """Build `TrainingSettings` from the `training` config section.

    Output:
        A settings object passed into `fit_model()` and `_run_epoch()`.
    """

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
    """Apply Stage 1 Xavier initialization in-place.

    The model object is modified directly. After this function returns, Conv2d
    and Linear weights have fresh Xavier values, and BatchNorm starts with
    weight=1 and bias=0.
    """

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
    """Build the configured training loss.

    Input to this loss during training:
        logits: `(batch_size, 2)` from `StockCNNI20`.
        labels: `(batch_size,)`, integer class ids 0 or 1.
    """

    if settings.loss != "cross_entropy":
        raise ValueError(f"Unsupported loss: {settings.loss}")
    return nn.CrossEntropyLoss()


def build_optimizer(model: nn.Module, settings: TrainingSettings) -> torch.optim.Optimizer:
    """Build the configured optimizer.

    The optimizer owns references to model parameters. Later, `optimizer.step()`
    updates those parameters after `loss.backward()` computes gradients.
    """

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
    """Train a model and write best/last checkpoints plus history/metadata.

    Input:
        model: `StockCNNI20`, not yet necessarily moved to device.
        train_loader: batches of image `(B, 1, 64, 60)` and label `(B,)`.
        val_loader: same batch structure, but no gradient update.

    Output:
        `TrainingResult` with file paths to `best.pt`, `last.pt`,
        `train_history.csv`, and `train_metadata.json`.
    """

    device = torch.device(device)
    checkpoint_path = Path(checkpoint_dir)
    metrics_path = Path(metrics_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    # Reset model weights before training this seed. This is why independent
    # seed runs can produce different checkpoints.
    initialize_model_weights(model, settings.initialization_name)

    # Move model parameters to CPU/GPU. Every batch tensor must later be moved
    # to the same device before calling `model(images)`.
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
        # Training epoch: gradients are enabled and optimizer updates weights.
        train_loss, train_accuracy = _run_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            train=True,
            settings=settings,
        )
        # Validation epoch: no gradients and no optimizer step. This estimates
        # whether the current checkpoint generalizes better than previous ones.
        val_loss, val_accuracy = _run_epoch(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=None,
            device=device,
            train=False,
            settings=settings,
        )

        # Lower validation loss means the current model is the new best model.
        improved = val_loss < (best_val_loss - settings.early_stopping.min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            # `best.pt` is the checkpoint used later for prediction export.
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

        # One dictionary becomes one row in `train_history.csv`.
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
    # `last.pt` records the final training state even if it is not the best.
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
    # These metadata files are for audit/review; they are not read by the model.
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
    """Run one train or validation epoch and return `(loss, accuracy)`.

    Batch data path:
        DataLoader batch
        -> images `(B, 1, 64, 60)`, labels `(B,)`
        -> model logits `(B, 2)`
        -> loss scalar
        -> if training, gradients update model weights
    """

    model.train(mode=train)
    loss_total = 0.0
    correct_total = 0
    sample_total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(data_loader, start=1):
            # Extract tensors from the dataset batch. Metadata is ignored in
            # training because future returns/StockID/Date must not be inputs.
            images, labels = _unpack_batch(batch)

            # Move tensors to the same device as the model. Shape remains:
            # images `(B, 1, 64, 60)`, labels `(B,)`.
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            # Forward pass. `logits` shape is `(B, 2)`, where column 0 is
            # Down/non-positive score and column 1 is Up score.
            logits = model(images)

            # CrossEntropyLoss internally applies log-softmax, so logits should
            # be raw scores, not probabilities.
            loss = loss_fn(logits, labels)
            if train and optimizer is not None:
                # Backward computes gradients for every trainable parameter.
                loss.backward()
                if settings.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clipping)

                # Optimizer uses gradients to update model weights.
                optimizer.step()

            batch_size = int(labels.shape[0])
            loss_total += float(loss.detach().cpu().item()) * batch_size

            # During training history we use argmax over logits as the predicted
            # class. Final paper-style probability outputs are computed later in
            # evaluation code.
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
    """Extract image and label tensors from common batch formats.

    Expected Stage 1 batch:
        `{"image": tensor(B,1,64,60), "label": tensor(B), "metadata": ...}`.
    This function returns only image and label because training must ignore
    metadata fields.
    """

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
    """Write a model checkpoint.

    The checkpoint is a PyTorch file containing model weights, optimizer state,
    current epoch, best validation loss, config snapshot, and normalization
    metadata. Evaluation later loads `model_state_dict` from this file.
    """

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
    """Write per-epoch training history.

    Each row describes one completed epoch: train loss/accuracy, validation
    loss/accuracy, learning rate, elapsed time, and whether it became best.
    """

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
