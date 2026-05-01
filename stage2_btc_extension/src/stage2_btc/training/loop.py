"""Stage 2 BTC training loop.

역할:
    BTC image Dataset과 Stock_CNN-style model을 받아 학습하고 checkpoint/history를
    저장한다. 이 파일은 prediction CSV나 trading metric을 만들지 않는다.

Tensor 규칙:
    batch["image"]: `(batch_size, 1, height, width)`
    batch["label"]: `(batch_size,)`
    model output logits: `(batch_size, 2)`
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

from stage2_btc.config import get_config_section
from stage2_btc.models.stock_cnn import initialize_model_weights


@dataclass(frozen=True)
class TrainingResult:
    """학습 완료 후 저장된 주요 output 경로와 best epoch 정보."""

    best_epoch: int
    best_val_loss: float
    stopped_epoch: int
    stopped_early: bool
    best_checkpoint_path: str
    last_checkpoint_path: str
    train_history_path: str
    train_metadata_path: str

    def as_dict(self) -> dict[str, Any]:
        """JSON 저장 가능한 dict로 변환한다."""

        return asdict(self)


def fit_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    config: Mapping[str, Any],
    device: torch.device | str,
    checkpoint_dir: str | Path,
    metrics_dir: str | Path,
    run_context: Mapping[str, Any],
    normalization_metadata: Mapping[str, Any],
) -> TrainingResult:
    """BTC baseline model을 학습하고 checkpoint/history를 저장한다."""

    training_config = get_config_section(config, "training")
    device = torch.device(device)
    checkpoint_path = Path(checkpoint_dir)
    metrics_path = Path(metrics_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    initialize_model_weights(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1.0e-5)),
    )

    use_amp = bool(training_config.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    max_epochs = int(training_config.get("max_epochs", 100))
    early_config = training_config.get("early_stopping", {})
    patience = int(early_config.get("patience", 2))
    log_every = int(training_config.get("log_every_batches", 20))

    best_loss = float("inf")
    best_epoch = 0
    stopped_early = False
    patience_counter = 0
    history_rows: list[dict[str, Any]] = []

    best_file = checkpoint_path / "best.pt"
    last_file = checkpoint_path / "last.pt"
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        train_stats = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            scaler=scaler,
            use_amp=use_amp,
            log_every=log_every,
        )
        validation_stats = _run_epoch(
            model,
            validation_loader,
            criterion,
            optimizer=None,
            device=device,
            train=False,
            scaler=None,
            use_amp=False,
            log_every=0,
        )

        improved = validation_stats["loss"] < best_loss
        if improved:
            best_loss = float(validation_stats["loss"])
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_loss,
                    "run_context": dict(run_context),
                    "normalization": dict(normalization_metadata),
                },
                best_file,
            )
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": float(validation_stats["loss"]),
                "run_context": dict(run_context),
                "normalization": dict(normalization_metadata),
            },
            last_file,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],
            "val_loss": validation_stats["loss"],
            "val_acc": validation_stats["accuracy"],
            "seconds": round(time.time() - start, 3),
            "best": bool(improved),
        }
        history_rows.append(row)
        print(
            f"[stage2 train] epoch {epoch}/{max_epochs} "
            f"train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f} "
            f"train_acc={row['train_acc']:.4f} val_acc={row['val_acc']:.4f} "
            f"best={improved}",
            flush=True,
        )

        if patience_counter >= patience:
            stopped_early = True
            print(
                f"[stage2 train] early stopping at epoch {epoch} "
                f"best_epoch={best_epoch} best_val_loss={best_loss:.6f}",
                flush=True,
            )
            break

    history_file = metrics_path / "train_history.csv"
    metadata_file = metrics_path / "train_metadata.json"
    _write_csv(history_file, history_rows)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_context": dict(run_context),
        "normalization": dict(normalization_metadata),
        "training_config": dict(training_config),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_loss),
        "stopped_epoch": int(history_rows[-1]["epoch"]),
        "stopped_early": bool(stopped_early),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return TrainingResult(
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        stopped_epoch=int(history_rows[-1]["epoch"]),
        stopped_early=stopped_early,
        best_checkpoint_path=str(best_file),
        last_checkpoint_path=str(last_file),
        train_history_path=str(history_file),
        train_metadata_path=str(metadata_file),
    )


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
    log_every: int,
) -> dict[str, float]:
    """train 또는 validation epoch 하나를 실행한다."""

    model.train(mode=train)
    total_loss = 0.0
    total_correct = 0
    total_rows = 0
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device=device, dtype=torch.float32)
        labels = batch["label"].to(device=device, dtype=torch.long)
        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if train and optimizer is not None:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        total_rows += batch_size
        if train and log_every and batch_index % log_every == 0:
            print(f"batch={batch_index} loss={float(loss.detach().cpu()):.6f}", flush=True)

    return {
        "loss": total_loss / max(total_rows, 1),
        "accuracy": total_correct / max(total_rows, 1),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """history row들을 CSV로 저장한다."""

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
