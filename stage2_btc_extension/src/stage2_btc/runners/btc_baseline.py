"""Shared Stage 2 BTC baseline runner helpers.

역할:
    train/evaluate/trading/Grad-CAM script가 같은 data preparation 로직을 쓰도록
    묶는다. 이렇게 해야 split, normalization, sample universe가 script마다
    달라지는 일을 막을 수 있다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from stage2_btc.data import (
    BtcImageDataset,
    add_moving_average_column,
    build_btc_samples,
    build_btc_splits,
    find_btc_ohlcv_source,
    fit_pixel_normalization,
    load_btc_ohlcv,
)
from stage2_btc.paths import Stage2Paths


@dataclass(frozen=True)
class PreparedBtcData:
    """Stage 2 experiment 하나에 필요한 data 객체 모음."""

    source_file: str
    ohlcv: pd.DataFrame
    samples: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    normalization: Any
    datasets: dict[str, BtcImageDataset]


def prepare_btc_experiment_data(
    config: Mapping[str, Any],
    paths: Stage2Paths,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
    max_test_rows: int | None = None,
) -> PreparedBtcData:
    """BTC CSV부터 Dataset까지 한 번에 준비한다."""

    source_file = find_btc_ohlcv_source(config, paths)
    ohlcv = add_moving_average_column(load_btc_ohlcv(source_file), image_window)
    samples = build_btc_samples(
        ohlcv,
        config,
        image_window=int(image_window),
        return_horizon=int(return_horizon),
    )
    splits = build_btc_splits(samples, config)
    splits = {
        "train": _limit_rows(splits["train"], max_train_rows),
        "validation": _limit_rows(splits["validation"], max_validation_rows),
        "test": _limit_rows(splits["test"], max_test_rows),
    }
    normalization = fit_pixel_normalization(
        ohlcv,
        splits["train"],
        config,
        image_window=int(image_window),
        image_spec=str(image_spec),
    )
    datasets = {
        name: BtcImageDataset(
            ohlcv,
            frame,
            config,
            image_window=int(image_window),
            image_spec=str(image_spec),
            normalization=normalization,
        )
        for name, frame in splits.items()
    }
    return PreparedBtcData(
        source_file=str(source_file),
        ohlcv=ohlcv,
        samples=samples,
        splits=splits,
        normalization=normalization,
        datasets=datasets,
    )


def build_dataloaders(
    datasets: Mapping[str, BtcImageDataset],
    config: Mapping[str, Any],
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    """train/validation/test DataLoader를 만든다."""

    runtime = config["runtime"]
    training = config["training"]
    evaluation = config["evaluation"]
    num_workers = int(runtime.get("num_workers", 0))
    loader_common = {
        "num_workers": num_workers,
        "pin_memory": bool(runtime.get("pin_memory", False)),
        "persistent_workers": bool(runtime.get("persistent_workers", False)) and num_workers > 0,
    }
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(training.get("batch_size", 128)),
            shuffle=shuffle_train,
            **loader_common,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=int(evaluation.get("batch_size", training.get("batch_size", 128))),
            shuffle=False,
            **loader_common,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=int(evaluation.get("batch_size", training.get("batch_size", 128))),
            shuffle=False,
            **loader_common,
        ),
    }


def split_summary(splits: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """split별 row 수와 positive rate를 요약한다."""

    summary: dict[str, Any] = {}
    for name, frame in splits.items():
        summary[name] = {
            "num_rows": int(len(frame)),
            "positive_rate": None if frame.empty else float(frame["label"].mean()),
            "date_min": None if frame.empty else str(frame["Date"].min().date()),
            "date_max": None if frame.empty else str(frame["Date"].max().date()),
        }
    return summary


def _limit_rows(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """smoke test용 row 제한을 적용한다."""

    if limit is None or int(limit) <= 0:
        return frame.reset_index(drop=True)
    return frame.head(int(limit)).reset_index(drop=True)
