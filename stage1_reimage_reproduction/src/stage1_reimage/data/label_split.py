"""1단계 label, split, train-only normalization.

논문/근거 맥락:
    Re-image는 미래 수익률이 양수이면 Up class로 두는 binary label을 사용한다.
    로컬 1단계 plan은 train/validation year를 1993-2000, test year를
    2001-2019로 고정한다. 논문 요약에서 정확한 random seed는 보고되지 않았으므로
    train/validation은 70/30 random split로 명시했다.

Leakage rule:
    future-return column은 label/evaluation metadata일 뿐 model input이 아니다.
    pixel normalization은 training split에서만 fit한다.

읽는 법:
    `monthly20.py`는 image를 읽는 방법을 알고 있다. 이 파일은 그 image에 binary
    label을 붙이고, split을 부여하고, train-only normalization을 적용해서
    training 가능한 sample로 만든다.
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
    # 각 실험은 같은 I20 image를 사용하지만 target future-return horizon이 다르다.
    # 예: stage1_i20_r20은 future Ret_20d를 target으로 사용한다.
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
    """1단계 split parameter.

    이 객체는 data row가 아니라 `assign_splits()`가 모든 sample을 train,
    validation, test 중 하나로 표시할 때 사용하는 규칙이다.
    """

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
    """픽셀 정규화 설정값.

    raw image reader는 pixel을 `[0, 1]` 범위로 반환한다. 이 settings 객체는
    normalization code가 train image에서만 mean/std를 계산하도록 지정한다.
    """

    pixel_scale: float
    epsilon: float
    fit_on: str
    scope: str


@dataclass(frozen=True)
class PixelNormalizationStats:
    """training split에서만 계산한 scalar pixel normalization 통계값.

    이 값들은 horizon별로 train image에서 한 번 계산된다. 이후
    `HorizonSplitImageDataset.__getitem__()`이 모든 image를
    `(image - train_pixel_mean) / train_pixel_std`로 변환할 때 사용한다.
    """

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
        """JSON으로 저장 가능한 normalization metadata를 반환한다."""

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
    """하나의 horizon과 하나의 split에 대한 image/label dataset.

    입력 source:
        `Monthly20MemmapDataset`이 raw rendered I20 image와 metadata를 제공한다.

    출력:
        `image`: normalized tensor `(1, 64, 60)`.
        `label`: integer tensor. class 1은 positive future return.
        `metadata`: 나중 prediction output을 위해 보존하는 row metadata.
    """

    def __init__(
        self,
        base_dataset: Monthly20MemmapDataset,
        split_frame: pd.DataFrame,
        split_name: str,
        normalization_stats: PixelNormalizationStats,
        max_rows: int | None = None,
    ) -> None:
        # `split_frame`에는 한 horizon의 모든 valid row가 들어 있다. 여기서는
        # caller가 요청한 split, 예를 들어 train 또는 validation row만 남긴다.
        selected = split_frame[split_frame["split"].eq(split_name)].copy()
        if max_rows is not None and max_rows > 0:
            # `max_rows`는 smoke test 전용이다. local에서 수백만 sample을 돌리지 않고
            # code path만 확인할 수 있게 한다.
            selected = selected.head(max_rows).copy()
        if selected.empty:
            raise ValueError(f"No rows available for split: {split_name}")

        self.base_dataset = base_dataset
        self.frame = selected.reset_index(drop=True)
        self.split_name = split_name
        self.normalization_stats = normalization_stats

    def __len__(self) -> int:
        """이 split dataset의 row 수를 반환한다."""

        return int(len(self.frame))

    def __getitem__(self, index: int) -> dict[str, Any]:
        """normalized image, integer label, metadata를 반환한다.

        한 row의 출력:
            image: tensor `(1, 64, 60)`, float32, train mean/std로 normalize됨.
            label: scalar tensor, 0 또는 1.
            metadata: prediction CSV 해석에 쓰이는 dictionary.
        """

        row = self.frame.iloc[index]

        # 저장된 shard/local row id를 사용해서 memmap dataset에서 정확히 같은 image
        # row를 가져온다. 이렇게 label row와 image row의 alignment를 유지한다.
        image = self.base_dataset.get_image_tensor(
            int(row["shard_index"]),
            int(row["local_row"]),
        )

        # Train-only normalization. validation/test pixel을 mean/std 계산에 쓰지 않으므로
        # normalization leakage를 방지한다.
        image = (image - self.normalization_stats.train_pixel_mean) / (
            self.normalization_stats.train_pixel_std
        )

        # Metadata는 Date/StockID/return을 보존한다. 나중 evaluation CSV에서 어떤
        # stock-date image가 어떤 probability를 만들었는지 확인하기 위해 필요하다.
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
    """config의 `split` section에서 split setting을 만든다.

    출력:
        `assign_splits()`가 사용하는 작은 settings 객체.
    """

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
    """config의 `normalization` section에서 normalization setting을 만든다.

    출력:
        `compute_pixel_normalization()`이 사용하는 settings 객체.
    """

    normalization_config = get_config_section(config, "normalization")
    return NormalizationSettings(
        pixel_scale=float(normalization_config.get("pixel_scale", 255.0)),
        epsilon=float(normalization_config.get("epsilon", 1.0e-8)),
        fit_on=str(normalization_config.get("fit_on", "train")),
        scope=str(normalization_config.get("scope", "per_horizon")),
    )


def build_base_metadata(shards: Sequence[ShardInfo]) -> pd.DataFrame:
    """label metadata를 읽고 안정적인 row identifier를 추가한다.

    출력 column은 원본 label column에 아래 column을 추가한다:
    - `year`
    - `local_row`
    - `shard_index`
    - `global_index`
    """

    # 최종 metadata frame은 모든 연도의 image마다 row 하나를 가진다. image pixel은
    # 포함하지 않고 Date, StockID, return, row id만 포함한다.
    frames: list[pd.DataFrame] = []
    global_offset = 0
    for shard_index, shard in enumerate(shards):
        frame = pd.read_feather(shard.label_path).copy()

        # 나중에 matching image를 찾을 수 있도록 row id를 추가한다:
        #   shard_index -> 어느 연도 `.dat` file인지
        #   local_row   -> 그 file 내부 몇 번째 row인지
        #   global_index -> 모든 연도를 합친 stable id
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
    """target-return NaN filtering 이후 horizon별 label을 만든다.

    입력:
        `base_metadata`: 모든 label file에서 읽은 전체 row.
        `horizon_name`: `stage1_i20_r5`, `stage1_i20_r20`, `stage1_i20_r60` 중 하나.

    출력:
        usable sample마다 row 하나를 갖는 DataFrame. `target_return`, binary `label`,
        image를 가져오는 데 필요한 row id를 포함한다.
    """

    if horizon_name not in TARGET_COLUMNS:
        available = ", ".join(TARGET_COLUMNS)
        raise KeyError(f"Unknown horizon {horizon_name}. Available: {available}")

    # 예: horizon_name `stage1_i20_r20`은 target column `Ret_20d`로 mapping된다.
    target_column = TARGET_COLUMNS[horizon_name]

    # future return이 missing인 row는 supervised training에 사용할 수 없다.
    valid = base_metadata[target_column].notna()
    horizon_frame = base_metadata.loc[valid].copy()
    horizon_frame["target_return_name"] = target_column
    horizon_frame["target_return"] = horizon_frame[target_column].astype(float)

    # 이진분류 target:
    #   future return > 0  -> class 1 (Up)
    #   future return <= 0 -> class 0 (Down/non-positive)
    horizon_frame["label"] = (horizon_frame["target_return"] > 0.0).astype(np.int8)
    horizon_frame["horizon_name"] = horizon_name
    return horizon_frame


def assign_splits(
    horizon_frame: pd.DataFrame,
    settings: SplitSettings,
) -> pd.DataFrame:
    """horizon filtering 이후 train/validation/test split을 부여한다.

    출력:
        `horizon_frame`과 같은 row에 새 `split` column을 추가한 DataFrame.
        이후 `HorizonSplitImageDataset`은 이 column을 기준으로 train/val/test
        dataset을 만든다.
    """

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

    # 1993-2000 row만 train/validation으로 random 분할한다. test year는 절대
    # training으로 섞이지 않는다.
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
    """split assignment 이후 row count와 class balance를 요약한다."""

    # 이 summary는 JSON으로 저장된다. 나중에 dataset을 다시 만들지 않아도 split별
    # row 수와 positive 수를 확인할 수 있다.
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
    progress_label: str | None = None,
    progress_every_chunks: int = 25,
) -> PixelNormalizationStats:
    """training split에서만 scalar pixel mean/std를 계산한다.

    `max_images`가 있으면 첫 `max_images`개 training row만 사용한다. 이는 local
    smoke check용이며 output에 표시된다. full Kaggle run에서는 `max_images=None`을
    넘겨야 한다.

    Tensor/data 이동:
        raw shard images `(N, 64, 60)` uint8
        -> `[0, 1]` 범위의 float32
        -> 선택된 train pixel 전체의 scalar mean/std
        -> `PixelNormalizationStats`
    """

    if settings.fit_on != "train":
        raise ValueError(f"Stage 1 normalization must fit on train, got: {settings.fit_on}")
    if settings.scope != "per_horizon":
        raise ValueError(f"Stage 1 normalization scope must be per_horizon, got: {settings.scope}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    # train row만 사용한다. validation/test pixel이 normalization parameter에 영향을
    # 주면 evaluation data가 새는 것이므로 사용하지 않는다.
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
    chunks_done = 0
    images_done = 0
    total_images = int(len(train_rows))

    if progress_label is not None:
        print(
            f"[normalization:{progress_label}] start "
            f"images={total_images:,} chunk_size={chunk_size}",
            flush=True,
        )

    for shard_index, shard_rows in train_rows.groupby("shard_index", sort=True):
        local_rows = shard_rows["local_row"].to_numpy(dtype=np.int64)
        for start in range(0, len(local_rows), chunk_size):
            row_chunk = local_rows[start : start + chunk_size]

            # 여기서 `images` shape는 `(chunk_size, 64, 60)`이다. normalization은
            # pixel 값만 필요하므로 channel dimension은 없다.
            images = (
                dataset.get_image_arrays(int(shard_index), row_chunk).astype(np.float32)
                / settings.pixel_scale
            )
            pixel_sum += float(images.sum(dtype=np.float64))
            pixel_square_sum += float(np.square(images, dtype=np.float32).sum(dtype=np.float64))
            pixel_count += int(images.size)
            chunks_done += 1
            images_done += int(len(row_chunk))

            should_log_progress = (
                progress_label is not None
                and progress_every_chunks > 0
                and (
                    chunks_done == 1
                    or chunks_done % progress_every_chunks == 0
                    or images_done >= total_images
                )
            )
            if should_log_progress:
                progress = images_done / total_images if total_images else 1.0
                print(
                    f"[normalization:{progress_label}] "
                    f"{images_done:,}/{total_images:,} images "
                    f"({progress:.1%})",
                    flush=True,
                )

    # 모든 pixel을 memory에 보관하지 않기 위해 E[x^2] - E[x]^2 방식으로 std를 계산한다.
    mean = pixel_sum / pixel_count
    variance = max(pixel_square_sum / pixel_count - mean * mean, 0.0)
    std = max(float(np.sqrt(variance)), settings.epsilon)

    if progress_label is not None:
        print(
            f"[normalization:{progress_label}] done "
            f"mean={mean:.8f} std={std:.8f}",
            flush=True,
        )

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
    """한 horizon의 split과 normalization metadata를 저장한다.

    이 JSON file은 audit output이다. 특정 horizon에서 label/split/normalization이
    어떻게 만들어졌는지 설명하지만 model input은 아니다.
    """

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
    """inclusive `[start, end]` year range를 parsing한다."""

    if not isinstance(raw_years, Sequence) or isinstance(raw_years, str):
        raise TypeError("Year range must be a two-item sequence.")
    years = [int(value) for value in raw_years]
    if len(years) != 2:
        raise ValueError(f"Year range must contain two values: {years}")
    if years[0] > years[1]:
        raise ValueError(f"Year range start must be <= end: {years}")
    return years[0], years[1]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """UTF-8 JSON을 안정적인 formatting으로 저장한다."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
