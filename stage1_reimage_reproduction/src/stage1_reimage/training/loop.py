"""1단계 training loop와 checkpoint utility.

논문/근거 맥락:
    local Re-image summary는 cross-entropy loss, Adam learning rate 1e-5,
    batch size 128, Xavier initialization, dropout 0.5, validation loss가
    2 epoch 동안 개선되지 않으면 early stopping을 보고한다. exact seed, epoch 수,
    Adam betas/eps, weight decay, mixed precision은 보고되지 않았으므로 config와
    metadata에 implementation choice로 기록한다.

범위:
    이 module은 generic training loop를 구현한다. final prediction CSV,
    loss/accuracy 외 evaluation metric, portfolio result, Grad-CAM은 구현하지 않는다.

읽는 법:
    `fit_model()`이 main 함수다. model과 DataLoader를 받아 `_run_epoch()`를 반복
    호출하고, validation loss가 개선되면 `best.pt`를 저장하며, 나중 review를 위해
    training history file을 작성한다.
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
    """1단계 Adam optimizer setting."""

    name: str
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclass(frozen=True)
class EarlyStoppingSettings:
    """validation loss 기반 early stopping 설정."""

    monitor: str
    mode: str
    patience: int
    min_delta: float
    restore_best: bool


@dataclass(frozen=True)
class TrainingSettings:
    """config에서 parsing한 1단계 training setting.

    YAML을 읽은 뒤 training hyperparameter를 이 객체로 고정한다. training loop는
    config를 반복해서 indexing하지 않고 이 명시적 값을 사용한다.
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
    """`fit_model`이 반환하는 학습 결과 요약."""

    best_epoch: int
    best_val_loss: float
    stopped_epoch: int
    stopped_early: bool
    best_checkpoint_path: str
    last_checkpoint_path: str
    train_history_path: str
    train_metadata_path: str

    def as_dict(self) -> dict[str, Any]:
        """JSON으로 저장 가능한 result metadata를 반환한다."""

        return asdict(self)


def training_settings_from_config(config: Mapping[str, Any]) -> TrainingSettings:
    """config의 `training` section에서 `TrainingSettings`를 만든다.

    출력:
        `fit_model()`과 `_run_epoch()`에 전달되는 settings 객체.
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
    """1단계 Xavier initialization을 model에 in-place로 적용한다.

    model 객체를 직접 수정한다. 이 함수가 끝나면 Conv2d와 Linear weight는 새 Xavier
    값이 되고, BatchNorm은 weight=1, bias=0에서 시작한다.
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
    """config에 지정된 training loss를 만든다.

    training 중 이 loss가 받는 입력:
        logits: `(batch_size, 2)` from `StockCNNI20`.
        labels: `(batch_size,)`, integer class ids 0 or 1.
    """

    if settings.loss != "cross_entropy":
        raise ValueError(f"Unsupported loss: {settings.loss}")
    return nn.CrossEntropyLoss()


def build_optimizer(model: nn.Module, settings: TrainingSettings) -> torch.optim.Optimizer:
    """config에 지정된 optimizer를 만든다.

    optimizer는 model parameter 참조를 가진다. 나중에 `loss.backward()`가 gradient를
    계산한 뒤 `optimizer.step()`이 그 parameter를 업데이트한다.
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
    """model을 학습하고 best/last checkpoint와 history/metadata를 저장한다.

    입력:
        model: `StockCNNI20`. 아직 device로 이동하지 않았을 수 있다.
        train_loader: image `(B, 1, 64, 60)`, label `(B,)` batch.
        val_loader: 같은 batch 구조. 단 gradient update는 하지 않는다.

    출력:
        `best.pt`, `last.pt`, `train_history.csv`, `train_metadata.json` 경로가
        들어 있는 `TrainingResult`.
    """

    device = torch.device(device)
    checkpoint_path = Path(checkpoint_dir)
    metrics_path = Path(metrics_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    # 이 seed의 training을 시작하기 전에 model weight를 새로 초기화한다. 그래서
    # independent seed run마다 다른 checkpoint가 나올 수 있다.
    initialize_model_weights(model, settings.initialization_name)

    # model parameter를 CPU/GPU로 이동한다. 이후 batch tensor도 `model(images)` 호출
    # 전에 같은 device로 이동해야 한다.
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
        # 학습 epoch: gradient를 켜고 optimizer가 weight를 업데이트한다.
        train_loss, train_accuracy = _run_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            train=True,
            settings=settings,
        )
        # 검증 epoch: gradient와 optimizer step이 없다. 현재 checkpoint가 이전보다
        # 더 잘 일반화되는지 추정한다.
        val_loss, val_accuracy = _run_epoch(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=None,
            device=device,
            train=False,
            settings=settings,
        )

        # validation loss가 낮아지면 현재 model이 새로운 best model이다.
        improved = val_loss < (best_val_loss - settings.early_stopping.min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            # `best.pt`는 나중 prediction export에서 사용하는 checkpoint다.
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

        # dictionary 하나가 `train_history.csv`의 row 하나가 된다.
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
    # `last.pt`는 best가 아니더라도 마지막 training state를 기록한다.
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
    # 이 metadata file은 audit/review용이다. model이 읽는 input은 아니다.
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
    """train 또는 validation epoch 하나를 실행하고 `(loss, accuracy)`를 반환한다.

    Batch 데이터 흐름:
        DataLoader batch
        -> images `(B, 1, 64, 60)`, labels `(B,)`
        -> model logits `(B, 2)`
        -> loss scalar
        -> training이면 gradient가 model weight를 update
    """

    model.train(mode=train)
    loss_total = 0.0
    correct_total = 0
    sample_total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(data_loader, start=1):
            # dataset batch에서 tensor만 꺼낸다. metadata는 future return/StockID/Date를
            # 포함하므로 training input으로 사용하지 않는다.
            images, labels = _unpack_batch(batch)

            # tensor를 model과 같은 device로 옮긴다. shape는 그대로 유지된다:
            # images `(B, 1, 64, 60)`, labels `(B,)`.
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            # 순전파. `logits` shape는 `(B, 2)`이고 column 0은 Down/non-positive
            # score, column 1은 Up score다.
            logits = model(images)

            # CrossEntropyLoss는 내부에서 log-softmax를 적용하므로 logits는 probability가
            # 아니라 raw score여야 한다.
            loss = loss_fn(logits, labels)
            if train and optimizer is not None:
                # 역전파는 모든 trainable parameter의 gradient를 계산한다.
                loss.backward()
                if settings.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clipping)

                # Optimizer는 gradient를 사용해 model weight를 업데이트한다.
                optimizer.step()

            batch_size = int(labels.shape[0])
            loss_total += float(loss.detach().cpu().item()) * batch_size

            # training history에서는 logits의 argmax를 predicted class로 사용한다.
            # 최종 paper-style probability output은 나중 evaluation code에서 계산한다.
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
    """batch에서 image와 label tensor를 꺼낸다.

    1단계 expected batch:
        `{"image": tensor(B,1,64,60), "label": tensor(B), "metadata": ...}`.
    training은 metadata field를 무시해야 하므로 이 함수는 image와 label만 반환한다.
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
    """model checkpoint를 저장한다.

    checkpoint는 model weight, optimizer state, 현재 epoch, best validation loss,
    config snapshot, normalization metadata를 담은 PyTorch file이다. evaluation은
    나중에 이 file에서 `model_state_dict`를 load한다.
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
    """epoch별 training history를 저장한다.

    row 하나는 완료된 epoch 하나를 설명한다: train loss/accuracy, validation
    loss/accuracy, learning rate, elapsed time, best 여부.
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
    """UTF-8 JSON을 안정적인 formatting으로 저장한다."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _settings_as_dict(settings: TrainingSettings) -> dict[str, Any]:
    """nested training setting을 plain dictionary로 바꾼다."""

    payload = asdict(settings)
    payload["optimizer"]["betas"] = list(payload["optimizer"]["betas"])
    return payload


def _utc_now() -> str:
    """현재 UTC 시간을 ISO string으로 반환한다."""

    return datetime.now(timezone.utc).isoformat()
