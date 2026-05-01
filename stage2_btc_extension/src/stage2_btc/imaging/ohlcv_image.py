"""BTC OHLCV chart image generation.

역할:
    Re-image 방식의 1-channel chart image를 BTC OHLCV에서 직접 만든다.

입력/출력 shape:
    입력 window는 pandas DataFrame row slice다.
    출력 image는 `uint8` numpy array `(height, width)`이고 값은 `0` 또는 `255`다.

중요한 설계:
    MA는 BTC CSV에 없으므로 close price로 계산한 SMA 값을 사용한다. 이 함수는
    이미 sample window에 맞춰 전달된 `ma_values`만 그린다. MA 계산 자체는
    label/split 단계에서 전체 시계열의 과거 값만 사용해 수행한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def generate_btc_chart_image(
    ohlcv_window: pd.DataFrame,
    ma_values: np.ndarray,
    image_spec: str,
    window_config: Mapping[str, Any],
    pixels_per_day: int = 3,
    background_value: int = 0,
    foreground_value: int = 255,
) -> np.ndarray:
    """OHLCV window 하나를 Re-image-style grayscale chart로 변환한다.

    입력:
        `ohlcv_window`: image 종료 시점까지의 BTC OHLCV row. 길이는 I5/I20/I60.
        `ma_values`: 같은 길이의 SMA 값. MA를 그리지 않는 spec에서도 common sample
            universe를 위해 계산되어 있을 수 있다.
        `image_spec`: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb` 중 하나.

    출력:
        image array `(height, width)`. 모델에는 이후 `(1, height, width)` tensor로
        변환되어 들어간다.
    """

    height = int(window_config["height"])
    width = int(window_config["width"])
    include_volume = image_spec in {"ohlc_vb", "ohlc_ma_vb"}
    include_ma = image_spec in {"ohlc_ma", "ohlc_ma_vb"}
    if image_spec not in {"ohlc", "ohlc_vb", "ohlc_ma", "ohlc_ma_vb"}:
        raise ValueError(f"Unsupported image_spec: {image_spec}")

    # Re-image 계획 기준:
    #   - volume이 있는 spec은 image 하단을 volume 영역으로 나누고, price는 상단
    #     price area에만 그린다.
    #   - volume이 없는 spec은 bottom volume 영역을 비워두지 않고, price chart가
    #     전체 image height를 사용한다. 그래야 window 내부 low가 image 맨 아래까지
    #     rescale된다.
    price_height = int(window_config["price_height"]) if include_volume else height
    volume_height = int(window_config["volume_height"]) if include_volume else 0
    gap_height = int(window_config["gap_height"]) if include_volume else 0

    expected_width = len(ohlcv_window) * int(pixels_per_day)
    if expected_width != width:
        raise ValueError(f"Window width mismatch. expected={expected_width}, config={width}")

    image = np.full((height, width), int(background_value), dtype=np.uint8)
    opens = ohlcv_window["Open"].to_numpy(dtype=float)
    highs = ohlcv_window["High"].to_numpy(dtype=float)
    lows = ohlcv_window["Low"].to_numpy(dtype=float)
    closes = ohlcv_window["Close"].to_numpy(dtype=float)
    volumes = ohlcv_window["Volume"].to_numpy(dtype=float)

    price_min = float(np.min(lows))
    price_max = float(np.max(highs))
    open_rows = _values_to_price_rows(opens, price_min, price_max, price_height)
    high_rows = _values_to_price_rows(highs, price_min, price_max, price_height)
    low_rows = _values_to_price_rows(lows, price_min, price_max, price_height)
    close_rows = _values_to_price_rows(closes, price_min, price_max, price_height)

    for day in range(len(ohlcv_window)):
        left_col = day * int(pixels_per_day)
        center_col = left_col + 1
        right_col = left_col + 2

        image[open_rows[day], left_col] = foreground_value
        top = min(high_rows[day], low_rows[day])
        bottom = max(high_rows[day], low_rows[day])
        image[top : bottom + 1, center_col] = foreground_value
        image[close_rows[day], right_col] = foreground_value

    if include_ma:
        ma_rows = _values_to_price_rows(ma_values.astype(float), price_min, price_max, price_height)
        _draw_connected_line(image, ma_rows, pixels_per_day, foreground_value)

    if include_volume:
        volume_top = price_height + gap_height
        max_volume = float(np.max(volumes))
        if max_volume > 0:
            for day, volume in enumerate(volumes):
                bar_height = max(1, int(round((float(volume) / max_volume) * volume_height)))
                center_col = day * int(pixels_per_day) + 1
                start_row = volume_top + volume_height - bar_height
                image[start_row : volume_top + volume_height, center_col] = foreground_value

    return image


def _values_to_price_rows(
    values: np.ndarray,
    price_min: float,
    price_max: float,
    price_height: int,
) -> np.ndarray:
    """price 값을 chart image의 row index로 변환한다.

    row 0이 image 위쪽이다. 따라서 가격이 높을수록 작은 row index가 된다.
    """

    if price_max == price_min:
        return np.full(values.shape, price_height // 2, dtype=int)
    normalized = (values - price_min) / (price_max - price_min)
    rows = np.rint((1.0 - normalized) * (price_height - 1)).astype(int)
    return np.clip(rows, 0, price_height - 1)


def _draw_connected_line(
    image: np.ndarray,
    row_values: np.ndarray,
    pixels_per_day: int,
    foreground_value: int,
) -> None:
    """MA point들을 day center column 기준으로 이어서 그린다."""

    centers = np.arange(len(row_values)) * int(pixels_per_day) + 1
    for idx in range(len(row_values) - 1):
        c0, c1 = int(centers[idx]), int(centers[idx + 1])
        r0, r1 = int(row_values[idx]), int(row_values[idx + 1])
        steps = max(abs(c1 - c0), abs(r1 - r0), 1)
        cols = np.rint(np.linspace(c0, c1, steps + 1)).astype(int)
        rows = np.rint(np.linspace(r0, r1, steps + 1)).astype(int)
        image[rows, cols] = foreground_value
