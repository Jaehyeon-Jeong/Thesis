"""BTC label, split, normalization, and Dataset utilities.

역할:
    BTC OHLCV에서 image 종료일 `t`와 미래 horizon `R`을 맞춰 sample table을 만든다.
    이후 train/validation/test split, train-only pixel normalization, PyTorch
    Dataset 생성을 담당한다.

Leakage guard:
    image는 항상 `t`까지의 OHLCV와 과거 MA만 사용한다.
    label은 `t+R` close로 만든다.
    train/test period 끝에서는 `label_end_date <= split_signal_end`를 만족하지 않는
    sample을 제거한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from stage2_btc.config import get_config_section, get_window_config
from stage2_btc.imaging import generate_btc_chart_image


@dataclass(frozen=True)
class NormalizationStats:
    """train image에서 fit한 pixel normalization 값."""

    pixel_mean: float
    pixel_std: float
    pixel_scale: float
    epsilon: float
    num_images: int
    num_pixels: int

    def as_dict(self) -> dict[str, Any]:
        """JSON 저장용 dictionary로 변환한다."""

        return {
            "pixel_mean": float(self.pixel_mean),
            "pixel_std": float(self.pixel_std),
            "pixel_scale": float(self.pixel_scale),
            "epsilon": float(self.epsilon),
            "num_images": int(self.num_images),
            "num_pixels": int(self.num_pixels),
            "fit_on": "train",
        }


def add_moving_average_column(
    ohlcv: pd.DataFrame,
    image_window: int,
) -> pd.DataFrame:
    """BTC close price로 window-specific SMA column을 추가한다.

    `min_periods=image_window`라서 충분한 과거 close가 없는 row의 MA는 NaN이다.
    sample builder는 MA가 NaN인 image window를 제외한다.
    """

    frame = ohlcv.copy()
    ma_column = f"MA_{int(image_window)}"
    frame[ma_column] = (
        frame["Close"].rolling(window=int(image_window), min_periods=int(image_window)).mean()
    )
    return frame


def build_btc_samples(
    ohlcv: pd.DataFrame,
    config: Mapping[str, Any],
    image_window: int,
    return_horizon: int,
) -> pd.DataFrame:
    """BTC image sample metadata와 binary label을 만든다.

    출력 row 하나는 image 하나에 해당한다. 실제 image array는 저장하지 않고,
    `end_index`, `start_index`, `label_end_index`를 저장해 Dataset에서 필요할 때
    생성한다.
    """

    data_config = get_config_section(config, "data")
    frame = add_moving_average_column(ohlcv, image_window=image_window)
    ma_column = f"MA_{int(image_window)}"
    report_end_date = pd.Timestamp(str(data_config.get("report_end_date", "2024-12-31")))

    rows: list[dict[str, Any]] = []
    min_end_index = int(image_window) - 1
    max_end_index = len(frame) - int(return_horizon) - 1
    for end_index in range(min_end_index, max_end_index + 1):
        start_index = end_index - int(image_window) + 1
        label_end_index = end_index + int(return_horizon)
        signal_date = pd.Timestamp(frame.loc[end_index, "Date"])
        label_end_date = pd.Timestamp(frame.loc[label_end_index, "Date"])
        if signal_date > report_end_date:
            continue

        window_ma = frame.loc[start_index:end_index, ma_column]
        if window_ma.isna().any():
            continue

        entry_close = float(frame.loc[end_index, "Close"])
        exit_close = float(frame.loc[label_end_index, "Close"])
        future_return = exit_close / entry_close - 1.0
        rows.append(
            {
                "sample_index": len(rows),
                "start_index": int(start_index),
                "end_index": int(end_index),
                "label_end_index": int(label_end_index),
                "Date": signal_date,
                "label_end_date": label_end_date,
                "entry_close": entry_close,
                "exit_close": exit_close,
                "future_return": float(future_return),
                "label": int(future_return > 0.0),
                "image_window": int(image_window),
                "return_horizon": int(return_horizon),
                "ma_column": ma_column,
            }
        )

    samples = pd.DataFrame(rows)
    if samples.empty:
        raise ValueError(
            f"No BTC samples built for image_window={image_window}, "
            f"return_horizon={return_horizon}."
        )
    return samples


def build_btc_splits(samples: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """sample table을 train/validation/test로 나눈다.

    논문식 기본값:
        `2018-2020` train/validation pool을 seed 42로 70/30 random split하고,
        `2021-2024`를 chronological test holdout으로 둔다.
    """

    split_config = get_config_section(config, "split")
    train_val_start = pd.Timestamp(str(split_config["train_val_start_date"]))
    train_val_end = pd.Timestamp(str(split_config["train_val_end_date"]))
    test_start = pd.Timestamp(str(split_config["test_start_date"]))
    test_end = pd.Timestamp(str(split_config["test_end_date"]))
    purge_label_end_date = bool(split_config.get("purge_label_end_date", True))

    train_val_mask = samples["Date"].between(train_val_start, train_val_end)
    test_mask = samples["Date"].between(test_start, test_end)
    if purge_label_end_date:
        train_val_mask &= samples["label_end_date"].le(train_val_end)
        test_mask &= samples["label_end_date"].le(test_end)

    train_val = samples.loc[train_val_mask].copy().reset_index(drop=True)
    test = samples.loc[test_mask].copy().reset_index(drop=True)
    if train_val.empty or test.empty:
        raise ValueError(
            f"Empty split produced. train_val={len(train_val)}, test={len(test)}"
        )

    rng = np.random.default_rng(int(split_config.get("split_seed", 42)))
    order = rng.permutation(len(train_val))
    train_count = int(round(len(train_val) * float(split_config.get("train_ratio", 0.70))))
    train_indices = order[:train_count]
    validation_indices = order[train_count:]

    train = train_val.iloc[train_indices].copy().sort_values("Date").reset_index(drop=True)
    validation = (
        train_val.iloc[validation_indices].copy().sort_values("Date").reset_index(drop=True)
    )
    test = test.sort_values("Date").reset_index(drop=True)
    for split_name, frame in [("train", train), ("validation", validation), ("test", test)]:
        frame["split"] = split_name
    return {"train": train, "validation": validation, "test": test}


def fit_pixel_normalization(
    ohlcv: pd.DataFrame,
    samples: pd.DataFrame,
    config: Mapping[str, Any],
    image_window: int,
    image_spec: str,
) -> NormalizationStats:
    """train sample image들만 사용해 pixel mean/std를 계산한다."""

    normalization_config = get_config_section(config, "normalization")
    pixel_scale = float(normalization_config.get("pixel_scale", 255.0))
    epsilon = float(normalization_config.get("epsilon", 1.0e-8))

    total_sum = 0.0
    total_square_sum = 0.0
    total_pixels = 0
    for _, sample in samples.iterrows():
        image = generate_image_for_sample(ohlcv, sample, config, image_window, image_spec)
        scaled = image.astype(np.float64) / pixel_scale
        total_sum += float(scaled.sum())
        total_square_sum += float(np.square(scaled).sum())
        total_pixels += int(scaled.size)

    if total_pixels == 0:
        raise ValueError("Cannot fit normalization on zero pixels.")
    mean = total_sum / total_pixels
    variance = max(total_square_sum / total_pixels - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    return NormalizationStats(
        pixel_mean=float(mean),
        pixel_std=float(std),
        pixel_scale=pixel_scale,
        epsilon=epsilon,
        num_images=int(len(samples)),
        num_pixels=int(total_pixels),
    )


def generate_image_for_sample(
    ohlcv: pd.DataFrame,
    sample: Mapping[str, Any] | pd.Series,
    config: Mapping[str, Any],
    image_window: int,
    image_spec: str,
) -> np.ndarray:
    """sample metadata의 index 범위로 BTC chart image 하나를 생성한다."""

    window_config = get_window_config(config, image_window)
    imaging_config = get_config_section(config, "imaging")
    start_index = int(sample["start_index"])
    end_index = int(sample["end_index"])
    ma_column = str(sample.get("ma_column", f"MA_{int(image_window)}"))
    window = ohlcv.iloc[start_index : end_index + 1]
    ma_values = ohlcv.loc[start_index:end_index, ma_column].to_numpy(dtype=float)
    return generate_btc_chart_image(
        window,
        ma_values=ma_values,
        image_spec=image_spec,
        window_config=window_config,
        pixels_per_day=int(imaging_config.get("pixels_per_day", 3)),
        background_value=int(imaging_config.get("background_value", 0)),
        foreground_value=int(imaging_config.get("foreground_value", 255)),
    )


class BtcImageDataset(Dataset):
    """BTC generated chart image를 PyTorch batch로 제공하는 Dataset.

    `__getitem__` 출력:
        dict with
        - `image`: `(1, height, width)` float32 tensor
        - `label`: scalar long tensor
        - `metadata`: prediction CSV로 다시 저장할 sample 정보
    """

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        samples: pd.DataFrame,
        config: Mapping[str, Any],
        image_window: int,
        image_spec: str,
        normalization: NormalizationStats,
    ) -> None:
        self.ohlcv = ohlcv
        self.samples = samples.reset_index(drop=True).copy()
        self.config = config
        self.image_window = int(image_window)
        self.image_spec = str(image_spec)
        self.normalization = normalization

    def __len__(self) -> int:
        """dataset sample 수를 반환한다."""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """index 하나를 image tensor, label, metadata로 변환한다."""

        sample = self.samples.iloc[int(index)]
        image = generate_image_for_sample(
            self.ohlcv,
            sample,
            self.config,
            self.image_window,
            self.image_spec,
        )
        scaled = image.astype(np.float32) / float(self.normalization.pixel_scale)
        normalized = (
            scaled - float(self.normalization.pixel_mean)
        ) / (float(self.normalization.pixel_std) + float(self.normalization.epsilon))
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(dtype=torch.float32)
        metadata = _sample_to_metadata(sample, self.image_spec)
        return {
            "image": tensor,
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
            "metadata": metadata,
        }


def _sample_to_metadata(sample: pd.Series, image_spec: str) -> dict[str, Any]:
    """sample row를 prediction/export용 metadata dict로 바꾼다."""

    date = pd.Timestamp(sample["Date"])
    label_end_date = pd.Timestamp(sample["label_end_date"])
    return {
        "sample_index": int(sample["sample_index"]),
        "start_index": int(sample["start_index"]),
        "end_index": int(sample["end_index"]),
        "label_end_index": int(sample["label_end_index"]),
        "Date": date.isoformat(),
        "label_end_date": label_end_date.isoformat(),
        "entry_close": float(sample["entry_close"]),
        "exit_close": float(sample["exit_close"]),
        "future_return": float(sample["future_return"]),
        "label": int(sample["label"]),
        "image_window": int(sample["image_window"]),
        "image_spec": str(image_spec),
        "return_horizon": int(sample["return_horizon"]),
    }
