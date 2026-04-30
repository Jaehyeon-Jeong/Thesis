"""Lazy loader for the public `monthly_20d` Re-image data.

Paper/source context:
    Stage 1 uses author-provided rendered 20-day images. The local data audit
    confirmed these files are I20 full-spec images (`OHLC + 20-day MA +
    volume`) with labels in matching `.feather` files. The data loader does not
    construct labels or splits; those are implemented in later gates.

Shape contract:
    Raw image row: (64, 60) uint8
    Returned image tensor: (1, 64, 60) float32, scaled to [0, 1]
    Batch shape after PyTorch collation: (batch_size, 1, 64, 60)
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from stage1_reimage.config import get_config_section

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 60
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH
IMAGE_FILE_TEMPLATE = "20d_month_has_vb_20_ma_{year}_images.dat"
LABEL_FILE_TEMPLATE = "20d_month_has_vb_20_ma_{year}_labels_w_delay.feather"
REQUIRED_LABEL_COLUMNS: tuple[str, ...] = (
    "Date",
    "StockID",
    "MarketCap",
    "Ret_5d",
    "Ret_20d",
    "Ret_60d",
    "Ret_month",
    "EWMA_vol",
)


@dataclass(frozen=True)
class ShardInfo:
    """Metadata for one year of image and label files."""

    year: int
    image_path: Path
    label_path: Path
    num_rows: int
    image_file_size: int
    label_num_rows: int
    date_min: str
    date_max: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable shard summary."""

        return {
            "year": self.year,
            "image_path": str(self.image_path),
            "label_path": str(self.label_path),
            "num_rows": self.num_rows,
            "image_file_size": self.image_file_size,
            "label_num_rows": self.label_num_rows,
            "date_min": self.date_min,
            "date_max": self.date_max,
        }


def years_from_config(config: Mapping[str, Any]) -> list[int]:
    """Parse the `data.expected_years` config value.

    The current config uses `[1993, 2019]` as an inclusive range. If a future
    config provides a longer list, it is treated as the explicit year list.
    """

    data_config = get_config_section(config, "data")
    raw_years = data_config.get("expected_years", [1993, 2019])
    if not isinstance(raw_years, Sequence) or isinstance(raw_years, str):
        raise TypeError("data.expected_years must be a sequence of integer years.")

    years = [int(year) for year in raw_years]
    if len(years) == 2 and years[0] < years[1] and years[1] - years[0] > 1:
        return list(range(years[0], years[1] + 1))
    return years


def infer_image_row_count(image_path: Path) -> tuple[int, int]:
    """Infer row count from a raw uint8 `.dat` image file."""

    file_size = image_path.stat().st_size
    if file_size % IMAGE_PIXELS != 0:
        raise ValueError(
            f"Image file size is not divisible by {IMAGE_PIXELS}: {image_path}"
        )
    return file_size // IMAGE_PIXELS, file_size


def discover_monthly20_shards(
    data_root: str | Path,
    years: Iterable[int],
    validate: bool = True,
) -> list[ShardInfo]:
    """Discover and validate year-aligned image/label shards.

    Validation checks:
    - expected image and label files exist,
    - image byte count reshapes to `(N, 64, 60)`,
    - image row count equals label row count,
    - required label columns exist,
    - label `Date` values belong to the file year.
    """

    root = Path(data_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    shards: list[ShardInfo] = []
    seen_years: set[int] = set()
    for year in [int(value) for value in years]:
        if year in seen_years:
            raise ValueError(f"Duplicate year in shard discovery: {year}")
        seen_years.add(year)

        image_path = root / IMAGE_FILE_TEMPLATE.format(year=year)
        label_path = root / LABEL_FILE_TEMPLATE.format(year=year)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file for {year}: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {year}: {label_path}")

        image_rows, image_file_size = infer_image_row_count(image_path)
        label_frame = pd.read_feather(label_path)
        if validate:
            _validate_label_frame(label_frame, label_path, year)
            if image_rows != len(label_frame):
                raise ValueError(
                    "Image/label row count mismatch for "
                    f"{year}: image_rows={image_rows}, label_rows={len(label_frame)}"
                )

        dates = pd.to_datetime(label_frame["Date"])
        shards.append(
            ShardInfo(
                year=year,
                image_path=image_path,
                label_path=label_path,
                num_rows=image_rows,
                image_file_size=image_file_size,
                label_num_rows=len(label_frame),
                date_min=str(dates.min().date()),
                date_max=str(dates.max().date()),
            )
        )

    return shards


def build_dataset_from_config(config: Mapping[str, Any]) -> "Monthly20MemmapDataset":
    """Build a `Monthly20MemmapDataset` using the environment config."""

    data_config = get_config_section(config, "data")
    data_root = Path(str(data_config["monthly20_root"])).expanduser()
    shards = discover_monthly20_shards(data_root=data_root, years=years_from_config(config))
    expected_num_image_files = data_config.get("expected_num_image_files")
    expected_num_label_files = data_config.get("expected_num_label_files")
    if expected_num_image_files is not None and len(shards) != int(expected_num_image_files):
        raise ValueError(
            "Unexpected number of discovered image shards: "
            f"expected={expected_num_image_files}, actual={len(shards)}"
        )
    if expected_num_label_files is not None and len(shards) != int(expected_num_label_files):
        raise ValueError(
            "Unexpected number of discovered label shards: "
            f"expected={expected_num_label_files}, actual={len(shards)}"
        )
    return Monthly20MemmapDataset(shards)


class Monthly20MemmapDataset(Dataset):
    """PyTorch Dataset over validated `monthly_20d` shards.

    The image files stay memory-mapped; the label frames are loaded once because
    they are much smaller than the images and are needed for metadata alignment.
    This class intentionally returns raw future-return columns only as metadata.
    They must not be passed into the Stage 1 CNN as input features.
    """

    def __init__(self, shards: Sequence[ShardInfo]) -> None:
        if not shards:
            raise ValueError("At least one shard is required.")
        self.shards = list(shards)
        self._label_frames = [pd.read_feather(shard.label_path) for shard in self.shards]
        self._image_maps = [
            np.memmap(
                shard.image_path,
                dtype=np.uint8,
                mode="r",
                shape=(shard.num_rows, IMAGE_HEIGHT, IMAGE_WIDTH),
            )
            for shard in self.shards
        ]
        cumulative_total = 0
        self._cumulative_end_rows: list[int] = []
        for shard in self.shards:
            cumulative_total += shard.num_rows
            self._cumulative_end_rows.append(cumulative_total)

    def __len__(self) -> int:
        """Return total number of rows across all shards."""

        return self._cumulative_end_rows[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample.

        Output:
            image: torch.float32 tensor with shape `(1, 64, 60)`
            metadata: dictionary containing original label columns plus `year`
                and `local_row`
        """

        shard_index, local_row = self.locate(index)
        image_tensor = self.get_image_tensor(shard_index, local_row)
        metadata = self.get_metadata(shard_index, local_row)
        return {"image": image_tensor, "metadata": metadata}

    def locate(self, index: int) -> tuple[int, int]:
        """Map a global row index to `(shard_index, local_row)`."""

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        shard_index = bisect_right(self._cumulative_end_rows, index)
        previous_end = 0 if shard_index == 0 else self._cumulative_end_rows[shard_index - 1]
        return shard_index, index - previous_end

    def get_image_tensor(self, shard_index: int, local_row: int) -> torch.Tensor:
        """Read one image lazily and return `(1, 64, 60)` float32 tensor."""

        image = self._image_maps[shard_index][local_row].astype(np.float32, copy=True)
        image /= 255.0
        image = image[np.newaxis, :, :]
        return torch.from_numpy(image)

    def get_metadata(self, shard_index: int, local_row: int) -> dict[str, Any]:
        """Return original label metadata plus shard identifiers."""

        row = self._label_frames[shard_index].iloc[local_row]
        metadata = row.to_dict()
        date_value = metadata.get("Date")
        if hasattr(date_value, "isoformat"):
            metadata["Date"] = date_value.isoformat()
        metadata["StockID"] = str(metadata.get("StockID"))
        metadata["year"] = self.shards[shard_index].year
        metadata["local_row"] = int(local_row)
        return metadata

    def shard_summary(self) -> list[dict[str, Any]]:
        """Return JSON-serializable summaries for all shards."""

        return [shard.as_dict() for shard in self.shards]


def _validate_label_frame(label_frame: pd.DataFrame, label_path: Path, year: int) -> None:
    """Validate label columns and date range for one shard."""

    missing_columns = [
        column for column in REQUIRED_LABEL_COLUMNS if column not in label_frame.columns
    ]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"Missing label column(s) in {label_path}: {missing_list}")

    dates = pd.to_datetime(label_frame["Date"], errors="raise")
    unexpected_years = sorted(set(int(value) for value in dates.dt.year.unique()))
    if unexpected_years != [year]:
        raise ValueError(
            f"Label Date values in {label_path} do not all belong to {year}: "
            f"{unexpected_years}"
        )
