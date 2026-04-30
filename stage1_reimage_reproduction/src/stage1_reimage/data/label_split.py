"""Labels, splits, and train-only normalization for Stage 1.

Paper/source context:
    Re-image uses a binary label where positive future return is the Up class.
    The local Stage 1 plan fixes train/validation years to 1993-2000, test
    years to 2001-2019, and uses a 70/30 random train/validation split because
    the paper summary does not report the exact random seed.

Leakage rule:
    Future-return columns are labels/evaluation metadata only. They are never
    model inputs. Pixel normalization is fitted on the training split only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from stage1_reimage.config import get_config_section
from stage1_reimage.data.monthly20 import Monthly20MemmapDataset, ShardInfo

TARGET_COLUMNS: dict[str, str] = {
    "stage1_i20_r5": "Ret_5d",
    "stage1_i20_r20": "Ret_20d",
    "stage1_i20_r60": "Ret_60d",
}

HORIZON_SPECS: dict[str, dict[str, str]] = {
    horizon_name: {
        "image_window": "I20",
        "target_return_name": target_column,
    }
    for horizon_name, target_column in TARGET_COLUMNS.items()
}


@dataclass(frozen=True)
class SplitSettings:
    """Split parameters for Stage 1."""

    train_val_year_start: int
    train_val_year_end: int
    test_year_start: int
    test_year_end: int
    train_ratio: float
    validation_ratio: float
    split_seed: int
    stratify: bool


@dataclass(frozen=True)
class NormalizationSettings:
    """Pixel normalization parameters."""

    pixel_scale: float
    epsilon: float
    fit_on: str
    scope: str


@dataclass(frozen=True)
class PixelNormalizationStats:
    """Train-only scalar pixel normalization statistics."""

    target_return_name: str
    train_pixel_mean: float
    train_pixel_std: float
    pixel_scale: float
    epsilon: float
    num_train_images_available: int
    num_train_images_used: int
    num_pixels_used: int
    sampled_for_smoke: bool

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable normalization metadata."""

        return {
            "target_return_name": self.target_return_name,
            "train_pixel_mean": self.train_pixel_mean,
            "train_pixel_std": self.train_pixel_std,
            "pixel_scale": self.pixel_scale,
            "epsilon": self.epsilon,
            "num_train_images_available": self.num_train_images_available,
            "num_train_images_used": self.num_train_images_used,
            "num_pixels_used": self.num_pixels_used,
            "sampled_for_smoke": self.sampled_for_smoke,
            "fit_on": "train",
        }


class HorizonSplitImageDataset(Dataset):
    """Image/label dataset for one horizon and one split.

    Input source:
        `Monthly20MemmapDataset` provides raw rendered I20 images and metadata.

    Output:
        `image`: normalized tensor `(1, 64, 60)`.
        `label`: integer tensor, class 1 means positive future return.
        `metadata`: row metadata preserved for later prediction outputs.
    """

    def __init__(
        self,
        base_dataset: Monthly20MemmapDataset,
        split_frame: pd.DataFrame,
        split_name: str,
        normalization_stats: PixelNormalizationStats,
        max_rows: int | None = None,
    ) -> None:
        selected = split_frame[split_frame["split"].eq(split_name)].copy()
        if max_rows is not None and max_rows > 0:
            selected = selected.head(max_rows).copy()
        if selected.empty:
            raise ValueError(f"No rows available for split: {split_name}")

        self.base_dataset = base_dataset
        self.frame = selected.reset_index(drop=True)
        self.split_name = split_name
        self.normalization_stats = normalization_stats

    def __len__(self) -> int:
        """Return number of rows in this split dataset."""

        return int(len(self.frame))

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return normalized image, integer label, and metadata."""

        row = self.frame.iloc[index]
        image = self.base_dataset.get_image_tensor(
            int(row["shard_index"]),
            int(row["local_row"]),
        )
        image = (image - self.normalization_stats.train_pixel_mean) / (
            self.normalization_stats.train_pixel_std
        )
        metadata = row.to_dict()
        date_value = metadata.get("Date")
        if hasattr(date_value, "isoformat"):
            metadata["Date"] = date_value.isoformat()
        metadata["StockID"] = str(metadata.get("StockID"))
        return {
            "image": image,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "metadata": metadata,
        }


def split_settings_from_config(config: Mapping[str, Any]) -> SplitSettings:
    """Build split settings from the `split` config section."""

    split_config = get_config_section(config, "split")
    train_val_years = _parse_year_range(split_config["train_val_years"])
    test_years = _parse_year_range(split_config["test_years"])
    stratify = bool(split_config.get("stratify", False))
    if stratify:
        raise NotImplementedError(
            "Stage 1 default split is non-stratified to match the current plan."
        )

    return SplitSettings(
        train_val_year_start=train_val_years[0],
        train_val_year_end=train_val_years[1],
        test_year_start=test_years[0],
        test_year_end=test_years[1],
        train_ratio=float(split_config.get("train_ratio", 0.70)),
        validation_ratio=float(split_config.get("validation_ratio", 0.30)),
        split_seed=int(split_config.get("split_seed", 42)),
        stratify=stratify,
    )


def normalization_settings_from_config(config: Mapping[str, Any]) -> NormalizationSettings:
    """Build normalization settings from the `normalization` config section."""

    normalization_config = get_config_section(config, "normalization")
    return NormalizationSettings(
        pixel_scale=float(normalization_config.get("pixel_scale", 255.0)),
        epsilon=float(normalization_config.get("epsilon", 1.0e-8)),
        fit_on=str(normalization_config.get("fit_on", "train")),
        scope=str(normalization_config.get("scope", "per_horizon")),
    )


def build_base_metadata(shards: Sequence[ShardInfo]) -> pd.DataFrame:
    """Load label metadata and add stable row identifiers.

    Output columns include the original label columns plus:
    - `year`
    - `local_row`
    - `shard_index`
    - `global_index`
    """

    frames: list[pd.DataFrame] = []
    global_offset = 0
    for shard_index, shard in enumerate(shards):
        frame = pd.read_feather(shard.label_path).copy()
        frame["year"] = int(shard.year)
        frame["local_row"] = np.arange(len(frame), dtype=np.int64)
        frame["shard_index"] = int(shard_index)
        frame["global_index"] = np.arange(
            global_offset,
            global_offset + len(frame),
            dtype=np.int64,
        )
        global_offset += len(frame)
        frames.append(frame)

    metadata = pd.concat(frames, ignore_index=True)
    metadata["Date"] = pd.to_datetime(metadata["Date"], errors="raise")
    return metadata


def build_horizon_frame(
    base_metadata: pd.DataFrame,
    horizon_name: str,
) -> pd.DataFrame:
    """Create horizon-specific labels after target-return NaN filtering."""

    if horizon_name not in TARGET_COLUMNS:
        available = ", ".join(TARGET_COLUMNS)
        raise KeyError(f"Unknown horizon {horizon_name}. Available: {available}")

    target_column = TARGET_COLUMNS[horizon_name]
    valid = base_metadata[target_column].notna()
    horizon_frame = base_metadata.loc[valid].copy()
    horizon_frame["target_return_name"] = target_column
    horizon_frame["target_return"] = horizon_frame[target_column].astype(float)
    horizon_frame["label"] = (horizon_frame["target_return"] > 0.0).astype(np.int8)
    horizon_frame["horizon_name"] = horizon_name
    return horizon_frame


def assign_splits(
    horizon_frame: pd.DataFrame,
    settings: SplitSettings,
) -> pd.DataFrame:
    """Assign train/validation/test split after horizon filtering."""

    frame = horizon_frame.copy()
    frame["split"] = ""

    train_val_mask = frame["year"].between(
        settings.train_val_year_start,
        settings.train_val_year_end,
        inclusive="both",
    )
    test_mask = frame["year"].between(
        settings.test_year_start,
        settings.test_year_end,
        inclusive="both",
    )
    if int((train_val_mask & test_mask).sum()) != 0:
        raise ValueError("Train/validation years overlap with test years.")

    train_val_indices = frame.index[train_val_mask].to_numpy()
    rng = np.random.default_rng(settings.split_seed)
    shuffled = rng.permutation(train_val_indices)
    train_count = int(round(len(shuffled) * settings.train_ratio))
    train_indices = shuffled[:train_count]
    validation_indices = shuffled[train_count:]

    frame.loc[train_indices, "split"] = "train"
    frame.loc[validation_indices, "split"] = "validation"
    frame.loc[test_mask, "split"] = "test"

    unassigned = frame["split"].eq("")
    if bool(unassigned.any()):
        years = sorted(int(value) for value in frame.loc[unassigned, "year"].unique())
        raise ValueError(f"Rows fall outside configured split years: {years}")

    return frame


def make_split_summary(
    split_frame: pd.DataFrame,
    settings: SplitSettings,
    horizon_name: str,
) -> dict[str, Any]:
    """Summarize counts and class balance after split assignment."""

    target_return_name = TARGET_COLUMNS[horizon_name]
    by_split: dict[str, dict[str, Any]] = {}
    for split_name in ["train", "validation", "test"]:
        split_part = split_frame[split_frame["split"].eq(split_name)]
        num_rows = int(len(split_part))
        positive_rows = int(split_part["label"].sum())
        non_positive_rows = int(num_rows - positive_rows)
        by_split[split_name] = {
            "num_rows": num_rows,
            "positive_rows": positive_rows,
            "non_positive_rows": non_positive_rows,
            "positive_rate": float(positive_rows / num_rows) if num_rows else None,
            "year_min": int(split_part["year"].min()) if num_rows else None,
            "year_max": int(split_part["year"].max()) if num_rows else None,
        }

    return {
        "horizon_name": horizon_name,
        "target_return_name": target_return_name,
        "label_rule": "1 if target_return > 0 else 0",
        "zero_return_class": 0,
        "train_val_years": [
            settings.train_val_year_start,
            settings.train_val_year_end,
        ],
        "test_years": [settings.test_year_start, settings.test_year_end],
        "train_ratio": settings.train_ratio,
        "validation_ratio": settings.validation_ratio,
        "split_seed": settings.split_seed,
        "stratify": settings.stratify,
        "total_valid_rows": int(len(split_frame)),
        "by_split": by_split,
    }


def compute_pixel_normalization(
    dataset: Monthly20MemmapDataset,
    split_frame: pd.DataFrame,
    settings: NormalizationSettings,
    target_return_name: str,
    max_images: int | None = None,
    chunk_size: int = 4096,
) -> PixelNormalizationStats:
    """Compute scalar pixel mean/std on the training split only.

    If `max_images` is provided, only the first `max_images` training rows are
    used. This is intended for local smoke checks and is marked in the output.
    Full Kaggle runs should pass `max_images=None`.
    """

    if settings.fit_on != "train":
        raise ValueError(f"Stage 1 normalization must fit on train, got: {settings.fit_on}")
    if settings.scope != "per_horizon":
        raise ValueError(f"Stage 1 normalization scope must be per_horizon, got: {settings.scope}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    train_rows = split_frame[split_frame["split"].eq("train")]
    num_train_available = int(len(train_rows))
    if num_train_available == 0:
        raise ValueError("No training rows available for normalization.")

    if max_images is not None and max_images > 0:
        train_rows = train_rows.head(max_images)
    sampled_for_smoke = len(train_rows) < num_train_available

    pixel_sum = 0.0
    pixel_square_sum = 0.0
    pixel_count = 0

    for shard_index, shard_rows in train_rows.groupby("shard_index", sort=True):
        local_rows = shard_rows["local_row"].to_numpy(dtype=np.int64)
        for start in range(0, len(local_rows), chunk_size):
            row_chunk = local_rows[start : start + chunk_size]
            images = (
                dataset.get_image_arrays(int(shard_index), row_chunk).astype(np.float32)
                / settings.pixel_scale
            )
            pixel_sum += float(images.sum(dtype=np.float64))
            pixel_square_sum += float(np.square(images, dtype=np.float32).sum(dtype=np.float64))
            pixel_count += int(images.size)

    mean = pixel_sum / pixel_count
    variance = max(pixel_square_sum / pixel_count - mean * mean, 0.0)
    std = max(float(np.sqrt(variance)), settings.epsilon)

    return PixelNormalizationStats(
        target_return_name=target_return_name,
        train_pixel_mean=float(mean),
        train_pixel_std=std,
        pixel_scale=settings.pixel_scale,
        epsilon=settings.epsilon,
        num_train_images_available=num_train_available,
        num_train_images_used=int(len(train_rows)),
        num_pixels_used=int(pixel_count),
        sampled_for_smoke=sampled_for_smoke,
    )


def write_horizon_metadata(
    output_dir: str | Path,
    split_summary: Mapping[str, Any],
    normalization_stats: PixelNormalizationStats,
    split_frame: pd.DataFrame | None = None,
    write_split_index: bool = False,
) -> dict[str, str]:
    """Write split and normalization metadata for one horizon."""

    horizon_dir = Path(output_dir)
    horizon_dir.mkdir(parents=True, exist_ok=True)

    split_summary_path = horizon_dir / "split_summary.json"
    normalization_path = horizon_dir / "normalization.json"
    _write_json(split_summary_path, split_summary)
    _write_json(normalization_path, normalization_stats.as_dict())

    written = {
        "split_summary": str(split_summary_path),
        "normalization": str(normalization_path),
    }
    if write_split_index:
        if split_frame is None:
            raise ValueError("split_frame is required when write_split_index=True.")
        split_index_path = horizon_dir / "split_index.csv"
        index_columns = [
            "year",
            "local_row",
            "global_index",
            "Date",
            "StockID",
            "target_return_name",
            "target_return",
            "label",
            "split",
        ]
        split_frame.loc[:, index_columns].to_csv(split_index_path, index=False)
        written["split_index"] = str(split_index_path)
    return written


def _parse_year_range(raw_years: Any) -> tuple[int, int]:
    """Parse an inclusive `[start, end]` year range."""

    if not isinstance(raw_years, Sequence) or isinstance(raw_years, str):
        raise TypeError("Year range must be a two-item sequence.")
    years = [int(value) for value in raw_years]
    if len(years) != 2:
        raise ValueError(f"Year range must contain two values: {years}")
    if years[0] > years[1]:
        raise ValueError(f"Year range start must be <= end: {years}")
    return years[0], years[1]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write UTF-8 JSON with stable formatting."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
