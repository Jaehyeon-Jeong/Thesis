"""Train-only context feature preprocessing for Stage 4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stage4_film.config import get_context_config

TRANSFORM_BY_FEATURE: dict[str, str] = {
    "fg_value": "divide_100",
    "fg_mean_60": "divide_100",
    "fg_delta_60": "divide_100",
    "fg_std_60": "divide_100",
    "bb_percent_b_60": "identity",
    "bb_bandwidth_60": "log1p_nonnegative",
    "mfi_60": "divide_100",
    "rv_60": "log1p_nonnegative",
}


@dataclass(frozen=True)
class ContextScaler:
    """Train split으로 fit한 context preprocessing 통계."""

    feature_order: list[str]
    transforms: dict[str, str]
    medians: dict[str, float]
    q01: dict[str, float]
    q99: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]
    epsilon: float
    missing_rate_by_split: dict[str, dict[str, float]]

    def as_dict(self) -> dict[str, Any]:
        """JSON 저장용 dictionary로 변환한다."""

        return {
            "feature_order": self.feature_order,
            "transforms": self.transforms,
            "medians": self.medians,
            "q01": self.q01,
            "q99": self.q99,
            "means": self.means,
            "stds": self.stds,
            "epsilon": float(self.epsilon),
            "missing_rate_by_split": self.missing_rate_by_split,
            "fit_on": "train",
        }


def fit_transform_context_features(
    context_table: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, ContextScaler]:
    """context raw feature를 transform/impute/clip/normalize한다.

    모든 통계는 `split == train` row에서만 fit한다.
    """

    context_config = get_context_config(config)
    feature_order = [str(feature) for feature in context_config["primary_features"]]
    preprocessing = context_config.get("preprocessing", {})
    if not isinstance(preprocessing, Mapping):
        raise TypeError("context.preprocessing must be a mapping.")
    clipping = preprocessing.get("clipping", {})
    scaling = preprocessing.get("scaling", {})
    if not isinstance(clipping, Mapping):
        raise TypeError("context.preprocessing.clipping must be a mapping.")
    if not isinstance(scaling, Mapping):
        raise TypeError("context.preprocessing.scaling must be a mapping.")

    lower_q = float(clipping.get("lower_quantile", 0.01))
    upper_q = float(clipping.get("upper_quantile", 0.99))
    epsilon = float(scaling.get("epsilon", 1.0e-8))

    frame = context_table.copy()
    missing_rate_by_split = _missing_rates(frame, feature_order)
    transforms = {
        feature: TRANSFORM_BY_FEATURE.get(feature, _infer_transform(feature))
        for feature in feature_order
    }

    transformed_columns: dict[str, pd.Series] = {}
    for feature in feature_order:
        transformed_columns[feature] = _apply_transform(
            pd.to_numeric(frame[feature], errors="coerce"),
            transforms[feature],
        )
        frame[f"{feature}_transformed"] = transformed_columns[feature]

    train_mask = frame["split"].astype(str).eq("train")
    if not train_mask.any():
        raise ValueError("Cannot fit context scaler because split == train is empty.")

    medians: dict[str, float] = {}
    q01: dict[str, float] = {}
    q99: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for feature in feature_order:
        transformed = pd.to_numeric(frame[f"{feature}_transformed"], errors="coerce")
        train_values = transformed.loc[train_mask].replace([np.inf, -np.inf], np.nan)
        median = float(train_values.median())
        if np.isnan(median):
            raise ValueError(f"Cannot fit context median for all-missing feature: {feature}")
        imputed = transformed.replace([np.inf, -np.inf], np.nan).fillna(median)
        lower = float(imputed.loc[train_mask].quantile(lower_q))
        upper = float(imputed.loc[train_mask].quantile(upper_q))
        if lower > upper:
            lower, upper = upper, lower
        clipped = imputed.clip(lower=lower, upper=upper)
        train_clipped = clipped.loc[train_mask]
        mean = float(train_clipped.mean())
        std = float(train_clipped.std(ddof=0))
        if not np.isfinite(std) or std < epsilon:
            std = 1.0

        medians[feature] = median
        q01[feature] = lower
        q99[feature] = upper
        means[feature] = mean
        stds[feature] = std

        frame[f"{feature}_imputed_clipped"] = clipped
        frame[f"{feature}_normalized"] = (clipped - mean) / max(std, epsilon)

    scaler = ContextScaler(
        feature_order=feature_order,
        transforms=transforms,
        medians=medians,
        q01=q01,
        q99=q99,
        means=means,
        stds=stds,
        epsilon=epsilon,
        missing_rate_by_split=missing_rate_by_split,
    )
    return frame, scaler


def normalized_feature_columns(feature_order: list[str]) -> list[str]:
    """정규화된 feature column 이름 목록을 반환한다."""

    return [f"{feature}_normalized" for feature in feature_order]


def _missing_rates(frame: pd.DataFrame, feature_order: list[str]) -> dict[str, dict[str, float]]:
    """raw feature missing rate를 split별로 계산한다."""

    result: dict[str, dict[str, float]] = {}
    for split, split_frame in frame.groupby("split", sort=True):
        result[str(split)] = {
            feature: float(pd.to_numeric(split_frame[feature], errors="coerce").isna().mean())
            for feature in feature_order
        }
    return result


def _infer_transform(feature: str) -> str:
    """feature 이름 기반 fallback transform을 고른다."""

    if feature.startswith("fg_") or feature.startswith("mfi_"):
        return "divide_100"
    if "bandwidth" in feature or feature.startswith("rv_"):
        return "log1p_nonnegative"
    return "identity"


def _apply_transform(values: pd.Series, transform: str) -> pd.Series:
    """raw context value에 feature-specific transform을 적용한다."""

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if transform == "identity":
        return numeric
    if transform == "divide_100":
        return numeric / 100.0
    if transform == "log1p_nonnegative":
        clean = numeric.where(numeric >= 0.0)
        return np.log1p(clean)
    raise ValueError(f"Unsupported context transform: {transform}")
