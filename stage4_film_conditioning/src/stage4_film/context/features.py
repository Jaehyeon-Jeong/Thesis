"""Stage 4 structured numeric context feature builder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from stage4_film.config import get_context_config


def make_context_output_name(
    image_window: int,
    image_spec: str,
    return_horizon: int,
    context_window: int,
) -> str:
    """Model-method와 독립적인 context feature output 이름을 만든다."""

    return (
        f"stage4_context_i{int(image_window)}_"
        f"{image_spec}_r{int(return_horizon)}_c{int(context_window)}"
    )


def build_context_feature_table(
    *,
    ohlcv: pd.DataFrame,
    samples_by_split: Mapping[str, pd.DataFrame],
    fear_greed: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Stage 4 primary structured context table을 만든다.

    출력은 raw context feature와 split/sample metadata를 모두 포함한다. 정규화는
    `normalization.py`가 별도로 train-only 정책으로 수행한다.
    """

    context_config = get_context_config(config)
    context_window = int(context_config.get("context_window", 60))

    technical = _compute_ohlcv_context_features(ohlcv, context_window)
    fg_calendar = _compute_fear_greed_calendar_features(
        fear_greed,
        sample_min_date=min(
            pd.to_datetime(frame["Date"]).min() for frame in samples_by_split.values()
        ),
        sample_max_date=max(
            pd.to_datetime(frame["Date"]).max() for frame in samples_by_split.values()
        ),
        context_window=context_window,
    )

    rows: list[pd.DataFrame] = []
    for split_name, samples in samples_by_split.items():
        frame = samples.copy()
        frame["split"] = split_name
        rows.append(frame)
    samples = pd.concat(rows, ignore_index=True)
    samples["Date"] = pd.to_datetime(samples["Date"]).dt.normalize()

    table = samples.merge(technical, left_on="end_index", right_index=True, how="left")
    table = table.merge(fg_calendar, on="Date", how="left")

    primary_features = [str(feature) for feature in context_config["primary_features"]]
    for feature in primary_features:
        table[f"{feature}_missing"] = ~np.isfinite(pd.to_numeric(table[feature], errors="coerce"))

    ordered_metadata = [
        "sample_index",
        "split",
        "Date",
        "start_index",
        "end_index",
        "label_end_index",
        "label_end_date",
        "image_window",
        "return_horizon",
        "entry_close",
        "exit_close",
        "future_return",
        "label",
        "fg_source_date",
        "fg_age_days",
        "fg_missing",
        "fg_exact_date_match",
    ]
    ordered_columns = (
        [column for column in ordered_metadata if column in table.columns]
        + primary_features
        + [f"{feature}_missing" for feature in primary_features]
        + [column for column in table.columns if column not in set(ordered_metadata + primary_features)]
    )
    table = table.loc[:, list(dict.fromkeys(ordered_columns))]
    return table.sort_values(["split", "Date", "sample_index"]).reset_index(drop=True)


def build_context_feature_audit(
    context_table: pd.DataFrame,
    primary_features: list[str],
) -> dict[str, Any]:
    """context table의 missing rate와 split별 row count를 요약한다."""

    split_counts = {
        split: int(len(frame))
        for split, frame in context_table.groupby("split", sort=True)
    }
    missing_by_split: dict[str, dict[str, float]] = {}
    for split, frame in context_table.groupby("split", sort=True):
        missing_by_split[str(split)] = {
            feature: float(pd.to_numeric(frame[feature], errors="coerce").isna().mean())
            for feature in primary_features
        }
    warnings: list[str] = []
    train_missing = missing_by_split.get("train", {})
    for feature, rate in train_missing.items():
        if rate > 0.30:
            warnings.append(f"train missing rate > 30% for {feature}: {rate:.3f}")
    for split, feature_rates in missing_by_split.items():
        for feature, rate in feature_rates.items():
            if rate > 0.05:
                warnings.append(f"{split} missing rate > 5% for {feature}: {rate:.3f}")

    return {
        "num_rows": int(len(context_table)),
        "split_counts": split_counts,
        "date_min": _date_or_none(pd.to_datetime(context_table["Date"]).min()),
        "date_max": _date_or_none(pd.to_datetime(context_table["Date"]).max()),
        "primary_features": primary_features,
        "missing_rate_by_split": missing_by_split,
        "warnings": warnings,
        "fg_age_days": {
            "max": _safe_float(pd.to_numeric(context_table["fg_age_days"], errors="coerce").max()),
            "mean": _safe_float(pd.to_numeric(context_table["fg_age_days"], errors="coerce").mean()),
        },
    }


def _compute_ohlcv_context_features(
    ohlcv: pd.DataFrame,
    context_window: int,
) -> pd.DataFrame:
    """BTC OHLCV에서 Bollinger, MFI, realized volatility feature를 계산한다."""

    frame = ohlcv.copy().reset_index(drop=True)
    close = pd.to_numeric(frame["Close"], errors="coerce")
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce")

    middle = close.rolling(context_window, min_periods=context_window).mean()
    std = close.rolling(context_window, min_periods=context_window).std(ddof=0)
    upper = middle + 2.0 * std
    lower = middle - 2.0 * std
    band_width = upper - lower

    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    typical_diff = typical_price.diff()
    positive_flow = raw_money_flow.where(typical_diff > 0.0, 0.0)
    negative_flow = raw_money_flow.where(typical_diff < 0.0, 0.0)
    positive_sum = positive_flow.rolling(context_window, min_periods=context_window).sum()
    negative_sum = negative_flow.rolling(context_window, min_periods=context_window).sum()
    money_flow_ratio = positive_sum / negative_sum.replace(0.0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + money_flow_ratio)
    mfi = mfi.where(~((negative_sum == 0.0) & (positive_sum > 0.0)), 100.0)
    mfi = mfi.where(~((positive_sum == 0.0) & (negative_sum > 0.0)), 0.0)

    log_return = np.log(close / close.shift(1))
    rv_window = max(int(context_window) - 1, 1)
    realized_volatility = log_return.rolling(rv_window, min_periods=rv_window).std(ddof=0)

    return pd.DataFrame(
        {
            f"bb_percent_b_{context_window}": (close - lower) / band_width,
            f"bb_bandwidth_{context_window}": band_width / middle,
            f"mfi_{context_window}": mfi,
            f"rv_{context_window}": realized_volatility,
        },
        index=frame.index,
    )


def _compute_fear_greed_calendar_features(
    fear_greed: pd.DataFrame,
    *,
    sample_min_date: pd.Timestamp,
    sample_max_date: pd.Timestamp,
    context_window: int,
) -> pd.DataFrame:
    """F&G를 daily calendar로 맞추고 trailing context feature를 계산한다."""

    fg = fear_greed.copy()
    fg["fg_date"] = pd.to_datetime(fg["fg_date"]).dt.normalize()
    fg = fg.sort_values("fg_date").drop_duplicates("fg_date", keep="last")

    start_date = min(pd.Timestamp(sample_min_date).normalize(), fg["fg_date"].min())
    end_date = max(pd.Timestamp(sample_max_date).normalize(), fg["fg_date"].max())
    calendar = pd.DataFrame({"Date": pd.date_range(start_date, end_date, freq="D")})
    calendar = calendar.merge(
        fg.loc[:, ["fg_date", "fg_value", "fg_classification"]],
        left_on="Date",
        right_on="fg_date",
        how="left",
    )
    calendar["fg_source_date"] = calendar["fg_date"]
    calendar["fg_value_asof"] = calendar["fg_value"].ffill()
    calendar["fg_source_date"] = calendar["fg_source_date"].ffill()
    calendar["fg_classification_asof"] = calendar["fg_classification"].ffill()
    calendar["fg_missing"] = calendar["fg_value_asof"].isna()
    calendar["fg_exact_date_match"] = calendar["fg_date"].notna()
    calendar["fg_age_days"] = (calendar["Date"] - calendar["fg_source_date"]).dt.days

    value = pd.to_numeric(calendar["fg_value_asof"], errors="coerce")
    calendar["fg_value"] = value
    calendar[f"fg_mean_{context_window}"] = value.rolling(
        context_window,
        min_periods=context_window,
    ).mean()
    calendar[f"fg_delta_{context_window}"] = value - value.shift(context_window)
    calendar[f"fg_std_{context_window}"] = value.rolling(
        context_window,
        min_periods=context_window,
    ).std(ddof=0)

    output = calendar.loc[
        :,
        [
            "Date",
            "fg_source_date",
            "fg_age_days",
            "fg_missing",
            "fg_exact_date_match",
            "fg_value",
            f"fg_mean_{context_window}",
            f"fg_delta_{context_window}",
            f"fg_std_{context_window}",
        ],
    ].copy()
    output["fg_source_date"] = pd.to_datetime(output["fg_source_date"]).dt.date.astype("string")
    return output


def _date_or_none(value: Any) -> str | None:
    """Timestamp/date 값을 ISO date string으로 변환한다."""

    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _safe_float(value: Any) -> float | None:
    """NaN이면 None, 아니면 float."""

    if pd.isna(value):
        return None
    return float(value)
