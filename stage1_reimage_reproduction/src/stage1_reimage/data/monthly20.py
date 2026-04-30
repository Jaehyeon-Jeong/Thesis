"""공개 `monthly_20d` Re-image data를 lazy loading하는 loader.

논문/데이터 맥락:
    1단계는 저자가 제공한 rendered 20-day image를 사용한다. local data audit에서
    이 파일들이 I20 full-spec image(`OHLC + 20-day MA + volume`)이고, matching
    `.feather` label file이 있다는 점을 확인했다. 이 loader는 label이나 split을
    만들지 않는다. label/split은 다음 단계 파일에서 처리한다.

Shape 규칙:
    raw image row: (64, 60) uint8
    반환 image tensor: (1, 64, 60) float32, [0, 1]로 scaling
    PyTorch DataLoader batch: (batch_size, 1, 64, 60)

읽는 법:
    `.dat` 파일은 JPEG/PNG가 아니라 flat byte array다. 모든 이미지가 정확히
    `64 * 60` pixel이므로, 별도 image decoding 없이 bytes를
    `(num_images, 64, 60)`으로 reshape할 수 있다.
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
    """한 연도 image/label file에 대한 metadata.

    하나의 shard는 1993 같은 calendar year 하나에 대응한다. image `.dat`와
    label `.feather`는 같은 row 수를 가져야 하며, 두 파일의 row `i`는 같은
    stock/date sample을 설명해야 한다.
    """

    year: int
    image_path: Path
    label_path: Path
    num_rows: int
    image_file_size: int
    label_num_rows: int
    date_min: str
    date_max: str

    def as_dict(self) -> dict[str, Any]:
        """JSON으로 저장 가능한 shard summary를 반환한다."""

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
    """`data.expected_years` config 값을 parsing한다.

    현재 config의 `[1993, 2019]`는 inclusive range로 해석한다. 미래 config가 더 긴
    list를 제공하면 explicit year list로 처리한다.
    """

    # Config에는 `[1993, 2019]`로 저장되어 있다. 1단계에서는 이 값을 두 개의
    # 연도만 의미하는 것이 아니라 1993부터 2019까지 전체 범위로 해석한다.
    data_config = get_config_section(config, "data")
    raw_years = data_config.get("expected_years", [1993, 2019])
    if not isinstance(raw_years, Sequence) or isinstance(raw_years, str):
        raise TypeError("data.expected_years must be a sequence of integer years.")

    years = [int(year) for year in raw_years]
    if len(years) == 2 and years[0] < years[1] and years[1] - years[0] > 1:
        return list(range(years[0], years[1] + 1))
    return years


def infer_image_row_count(image_path: Path) -> tuple[int, int]:
    """raw uint8 `.dat` image file에서 row 수를 추론한다.

    파일 크기가 `N * 64 * 60` byte라면 이 파일에는 `N`개 image가 들어 있다.
    그래서 memmap을 만들기 전에 `IMAGE_PIXELS`로 나누어떨어지는지 확인한다.
    """

    file_size = image_path.stat().st_size
    if file_size % IMAGE_PIXELS != 0:
        raise ValueError(
            f"Image file size is not divisible by {IMAGE_PIXELS}: {image_path}"
        )
    # 예: file_size가 384000이면 384000 / 3840 = 100이므로 image 100개다.
    return file_size // IMAGE_PIXELS, file_size


def discover_monthly20_shards(
    data_root: str | Path,
    years: Iterable[int],
    validate: bool = True,
) -> list[ShardInfo]:
    """연도별 image/label shard를 찾고 검증한다.

    검증 내용:
    - 기대한 image/label file이 존재하는지
    - image byte count가 `(N, 64, 60)`으로 reshape 가능한지
    - image row count와 label row count가 같은지
    - 필수 label column이 존재하는지
    - label `Date` 값이 해당 file year에 속하는지
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

        # public monthly_20d dataset이 고정한 정확한 file name을 만든다.
        # file name 자체에도 volume bars와 MA가 포함되어 있다는 정보가 들어 있다.
        image_path = root / IMAGE_FILE_TEMPLATE.format(year=year)
        label_path = root / LABEL_FILE_TEMPLATE.format(year=year)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file for {year}: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {year}: {label_path}")

        # `image_rows`는 `.dat` 파일에서 읽을 수 있는 sample 수다.
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
    """환경 config를 사용해서 `Monthly20MemmapDataset`을 만든다.

    출력:
        나중에 sample 하나를
        `{"image": tensor(1, 64, 60), "metadata": {...}}` 형태로 반환하는 dataset.
    """

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
    """검증된 `monthly_20d` shard 위에서 동작하는 PyTorch Dataset.

    image file은 memory-map 상태로 두고, label frame은 image보다 작고 metadata
    alignment에 자주 필요하므로 한 번만 load한다. 이 class는 raw future-return
    column을 metadata로만 반환한다. 이 값들은 1단계 CNN input feature로 넣으면 안 된다.
    """

    def __init__(self, shards: Sequence[ShardInfo]) -> None:
        if not shards:
            raise ValueError("At least one shard is required.")
        self.shards = list(shards)

        # label metadata는 image byte보다 작고 Date/StockID/return column을 자주
        # 참조하므로 memory에 올려둔다.
        self._label_frames = [pd.read_feather(shard.label_path) for shard in self.shards]

        # image byte는 disk에 둔 채 memory map으로 접근한다. 이렇게 하면 전체
        # rendered image를 한 번에 RAM에 올리지 않아도 된다. 각 memmap shape는
        # `(num_rows_for_year, 64, 60)`, dtype은 `uint8`이다.
        self._image_maps = [
            np.memmap(
                shard.image_path,
                dtype=np.uint8,
                mode="r",
                shape=(shard.num_rows, IMAGE_HEIGHT, IMAGE_WIDTH),
            )
            for shard in self.shards
        ]
        # `_cumulative_end_rows`는 전체 dataset index를 `(year shard, 그 연도 내부 row)`
        # pair로 되돌리는 데 쓴다. 예:
        #   index 150000 -> 1995 shard_index, 1995 file 내부 local_row
        cumulative_total = 0
        self._cumulative_end_rows: list[int] = []
        for shard in self.shards:
            cumulative_total += shard.num_rows
            self._cumulative_end_rows.append(cumulative_total)

    def __len__(self) -> int:
        """모든 shard를 합친 전체 row 수를 반환한다."""

        return self._cumulative_end_rows[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """sample 하나를 반환한다.

        출력:
            image: `(1, 64, 60)` shape의 torch.float32 tensor.
            metadata: 원본 label column에 `year`, `local_row`가 추가된 dictionary
        """

        # 전체 dataset index 하나를 정확한 연도 파일과 그 안의 row 번호로 바꾼다.
        shard_index, local_row = self.locate(index)

        # 이 image tensor만 model input으로 사용된다. metadata는 sample과 같이
        # 이동하지만 CNN에 넣으면 안 된다.
        image_tensor = self.get_image_tensor(shard_index, local_row)
        metadata = self.get_metadata(shard_index, local_row)
        return {"image": image_tensor, "metadata": metadata}

    def locate(self, index: int) -> tuple[int, int]:
        """global row index를 `(shard_index, local_row)`로 mapping한다."""

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        # `bisect_right`는 `index`보다 큰 첫 cumulative row boundary를 찾는다.
        # 그 boundary가 바로 해당 row가 들어 있는 year shard를 알려준다.
        shard_index = bisect_right(self._cumulative_end_rows, index)
        previous_end = 0 if shard_index == 0 else self._cumulative_end_rows[shard_index - 1]
        return shard_index, index - previous_end

    def get_image_tensor(self, shard_index: int, local_row: int) -> torch.Tensor:
        """image 하나를 lazy하게 읽고 `(1, 64, 60)` float32 tensor로 반환한다.

        데이터 이동:
            disk memmap uint8 `(64, 60)`
            -> NumPy float32 `(64, 60)`, `[0, 1]` scaling
            -> channel dimension 추가 `(1, 64, 60)`
            -> DataLoader batch 생성을 위한 PyTorch tensor
        """

        image = self.get_image_array(shard_index, local_row).astype(np.float32, copy=True)
        # raw pixel은 0-255다. 255로 나누면 이후 train-only mean/std normalization
        # 전에 sample 간 scale이 맞춰진다.
        image /= 255.0
        # CNN은 channel dimension을 기대한다. 이미지는 grayscale이므로 channel 수는
        # 1이고, `(64, 60)`이 `(1, 64, 60)`이 된다.
        image = image[np.newaxis, :, :]
        return torch.from_numpy(image)

    def get_image_array(self, shard_index: int, local_row: int) -> np.ndarray:
        """memmap에서 raw `(64, 60)` uint8 image 하나를 읽는다."""

        return self._image_maps[shard_index][local_row]

    def get_image_arrays(self, shard_index: int, local_rows: np.ndarray) -> np.ndarray:
        """하나의 shard memmap에서 raw `(N, 64, 60)` uint8 image를 읽는다.

        이 batch-style reader는 normalization statistic을 효율적으로 계산할 때
        사용한다. PyTorch tensor를 반환하지 않는다.
        """

        return np.asarray(self._image_maps[shard_index][local_rows])

    def get_metadata(self, shard_index: int, local_row: int) -> dict[str, Any]:
        """원본 label metadata와 shard 식별자를 반환한다.

        이 dictionary에는 `Ret_20d` 같은 future-return column도 포함된다. 하지만
        이 값들은 label/evaluation field일 뿐 model input이 아니다. prediction CSV를
        나중에 해석하기 위해 보존한다.
        """

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
        """모든 shard의 JSON-serializable summary를 반환한다."""

        return [shard.as_dict() for shard in self.shards]


def _validate_label_frame(label_frame: pd.DataFrame, label_path: Path, year: int) -> None:
    """shard 하나의 label column과 date range를 검증한다."""

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
