# Stage 1 Data Loading Detail Plan

## English

Status:
- Stage 1-2 completed as a detail plan.
- No data-loading code has been implemented yet.

## Purpose

Define exactly how Stage 1 will load the author-provided public `monthly_20d`
stock images and label files before writing code.

This plan only covers data loading and row alignment. Horizon-specific label
construction is handled in Stage 1-3.

## Checked Inputs

Data root:
- `../테스트/Test/img_data/monthly_20d`

File pattern:
- Image file: `20d_month_has_vb_20_ma_{year}_images.dat`
- Label file: `20d_month_has_vb_20_ma_{year}_labels_w_delay.feather`
- Years: 1993 through 2019

Confirmed from Stage 0:
- 27 image `.dat` files.
- 27 label `.feather` files.
- Every image file has a matching label file.
- Every `.dat` file reshapes to `(N, 64, 60)`.
- For every year, inferred image rows equal label rows.

Additional checks during 1-2:
- `.dat` files are raw byte data.
- First bytes are values such as `0` and `255`, consistent with black/white grayscale images.
- `numpy`, `pandas`, `pyarrow`, and `torch` are available locally.
- Total image data is about `7.9G`; label feather files are about `102M`.

## Loading Decisions

### File Discovery

Use deterministic year-based discovery:

```text
for year in 1993..2019:
  image_path = data_root / f"20d_month_has_vb_20_ma_{year}_images.dat"
  label_path = data_root / f"20d_month_has_vb_20_ma_{year}_labels_w_delay.feather"
```

Validation required before training:
- Both files must exist for every year.
- No duplicate year pair is allowed.
- File years must exactly match 1993-2019 unless the user explicitly changes the scope.

### Image Data

Image dtype:
- Read `.dat` as `uint8`.
- Pixel values are interpreted as grayscale bytes, expected mainly `0` and `255`.

Image shape:
- Per-year image array shape: `(num_rows_for_year, 64, 60)`.
- Model tensor shape after transform: `(1, 64, 60)`.
- Batch tensor shape: `(batch_size, 1, 64, 60)`.

Read strategy:
- Use memory-mapped loading by default, not eager full loading.
- Reason: image files total about `7.9G`, and full eager loading would be unnecessary and fragile.

Planned transform per sample:
- Read one image as `uint8` with shape `(64, 60)`.
- Convert to `float32`.
- Scale to `[0, 1]` by dividing by `255.0`.
- Add channel dimension to produce `(1, 64, 60)`.

Normalization:
- Do not finalize mean/std normalization in 1-2.
- Stage 1-4 will decide train-only pixel mean/std standardization.
- If standardization is used, it must be fitted on train data only.

### Label Data

Read label files with `pandas.read_feather`.

Observed columns:
- `Date`
- `StockID`
- `MarketCap`
- `Ret_5d`
- `Ret_20d`
- `Ret_60d`
- `Ret_month`
- `EWMA_vol`

Observed dtypes from the 1993 label file:
- `Date`: `datetime64[ns]`
- `StockID`: `object`
- `MarketCap`: `float32`
- return columns: `float64`
- `EWMA_vol`: `float64`

Label loading rule:
- Preserve original row order.
- Preserve all metadata columns through label construction and prediction output.
- Convert `StockID` to string only if needed for stable CSV output; do not use it as model input.

### Row Alignment

The data loader must treat each image row and label row with the same local row
index as the same sample.

Required validation per year:
- `image_file_size % (64 * 60) == 0`
- `image_file_size / (64 * 60) == len(label_dataframe)`
- `Date` values should fall inside the file year.
- Required label columns must be present.

If any validation fails:
- Stop before training.
- Write the failing year and reason to a data-audit output.

### Indexing Strategy

Use a shard-aware index.

Planned base sample key:
- `year`
- `local_row`
- `Date`
- `StockID`

The dataset should map a global sample index to:
- year shard
- local row index within that shard
- memmapped image row
- label metadata row

Horizon-specific filtering is not done in 1-2.
It will be added in 1-3 after choosing `Ret_5d`, `Ret_20d`, or `Ret_60d`.

### Shuffling Rule

The base loaded data must not be shuffled before:
- horizon label filtering,
- split assignment,
- and normalization statistics are defined.

Training dataloaders may shuffle only the training subset after Stage 1-4 fixes
the split.

## Planned Data Loader Interface

The eventual implementation should expose two levels:

1. Shard metadata loader:
   - discovers year pairs,
   - validates row counts,
   - records image path, label path, row count, date range.

2. Dataset object:
   - reads images lazily from memmap,
   - returns metadata and labels after Stage 1-3,
   - returns tensors with shape `(1, 64, 60)`.

Expected sample dictionary after Stage 1-3:

```python
{
    "image": Tensor[1, 64, 60],
    "label": int,
    "target_return": float,
    "date": Timestamp,
    "stock_id": str,
    "year": int,
    "local_row": int,
}
```

## Explicit Non-input Columns

The following columns are metadata or labels, not image-model inputs:
- `Date`
- `StockID`
- `MarketCap`
- `Ret_5d`
- `Ret_20d`
- `Ret_60d`
- `Ret_month`
- `EWMA_vol`

For the Stage 1 CNN baseline, model input is only the image tensor.

## Risks and Open Items for Later Checklist Steps

Deferred to Stage 1-3:
- Horizon-specific NaN filtering.
- Zero-return label handling confirmation.
- Exact prediction-output schema per horizon.

Deferred to Stage 1-4:
- Train/validation/test split implementation.
- Train-only normalization statistics.
- Whether labels are loaded all at once or per split after filtering.

Deferred to Stage 1-6:
- DataLoader worker count.
- Pin memory and device transfer settings.
- Performance tuning.

## 한국어

상태:
- 1-2를 data loading 세부계획으로 완료했습니다.
- data-loading 코드는 아직 구현하지 않았습니다.

## 목적

코드를 쓰기 전에 1단계에서 저자 공개 `monthly_20d` stock image와 label file을
어떻게 읽을지 정확히 정의합니다.

이 문서는 data loading과 row alignment만 다룹니다. horizon별 label 생성은
1-3에서 다룹니다.

## 확인한 입력

Data root:
- `../테스트/Test/img_data/monthly_20d`

파일 패턴:
- Image file: `20d_month_has_vb_20_ma_{year}_images.dat`
- Label file: `20d_month_has_vb_20_ma_{year}_labels_w_delay.feather`
- 연도: 1993년부터 2019년까지

0단계에서 확인한 내용:
- image `.dat` 파일 27개.
- label `.feather` 파일 27개.
- 모든 image file은 matching label file을 가집니다.
- 모든 `.dat` 파일은 `(N, 64, 60)`으로 reshape됩니다.
- 모든 연도에서 inferred image row 수와 label row 수가 일치합니다.

1-2에서 추가 확인한 내용:
- `.dat` 파일은 raw byte data입니다.
- 첫 byte 값은 `0`, `255` 등으로, black/white grayscale image와 일치합니다.
- 로컬에 `numpy`, `pandas`, `pyarrow`, `torch`가 사용 가능합니다.
- 전체 image data는 약 `7.9G`, label feather file은 약 `102M`입니다.

## Loading 결정

### 파일 탐색

연도 기반 deterministic discovery를 사용합니다.

```text
for year in 1993..2019:
  image_path = data_root / f"20d_month_has_vb_20_ma_{year}_images.dat"
  label_path = data_root / f"20d_month_has_vb_20_ma_{year}_labels_w_delay.feather"
```

학습 전 필수 검증:
- 모든 연도에서 두 파일이 모두 존재해야 합니다.
- 같은 연도의 중복 pair는 허용하지 않습니다.
- 사용자가 범위를 바꾸지 않는 한 file year는 정확히 1993-2019여야 합니다.

### Image Data

Image dtype:
- `.dat`는 `uint8`로 읽습니다.
- pixel value는 grayscale byte로 해석하며, 주로 `0`과 `255`를 기대합니다.

Image shape:
- 연도별 image array shape: `(num_rows_for_year, 64, 60)`.
- transform 이후 model tensor shape: `(1, 64, 60)`.
- batch tensor shape: `(batch_size, 1, 64, 60)`.

읽기 전략:
- 기본은 full eager loading이 아니라 memory-mapped loading입니다.
- 이유: image file 총량이 약 `7.9G`라서 전체를 RAM에 올리는 방식은 불필요하고 취약합니다.

Sample별 transform 계획:
- image 하나를 `(64, 60)`의 `uint8`로 읽습니다.
- `float32`로 변환합니다.
- `255.0`으로 나누어 `[0, 1]`로 scale합니다.
- channel dimension을 추가해 `(1, 64, 60)`으로 만듭니다.

Normalization:
- 1-2에서는 mean/std normalization을 확정하지 않습니다.
- train-only pixel mean/std standardization 여부는 1-4에서 결정합니다.
- standardization을 쓰면 train data에서만 fit해야 합니다.

### Label Data

Label file은 `pandas.read_feather`로 읽습니다.

확인된 columns:
- `Date`
- `StockID`
- `MarketCap`
- `Ret_5d`
- `Ret_20d`
- `Ret_60d`
- `Ret_month`
- `EWMA_vol`

1993 label file에서 확인한 dtype:
- `Date`: `datetime64[ns]`
- `StockID`: `object`
- `MarketCap`: `float32`
- return columns: `float64`
- `EWMA_vol`: `float64`

Label loading rule:
- 원래 row order를 보존합니다.
- 모든 metadata column은 label construction과 prediction output까지 보존합니다.
- CSV output 안정성이 필요할 때만 `StockID`를 string으로 변환합니다. model input으로는 사용하지 않습니다.

### Row Alignment

같은 연도의 image row와 label row는 같은 local row index일 때 같은 sample로 봅니다.

연도별 필수 검증:
- `image_file_size % (64 * 60) == 0`
- `image_file_size / (64 * 60) == len(label_dataframe)`
- `Date` 값은 해당 file year 안에 있어야 합니다.
- 필수 label column이 모두 있어야 합니다.

검증 실패 시:
- training 전에 중단합니다.
- 실패한 연도와 이유를 data-audit output에 기록합니다.

### Indexing Strategy

Shard-aware index를 사용합니다.

기본 sample key:
- `year`
- `local_row`
- `Date`
- `StockID`

Dataset은 global sample index를 다음으로 매핑해야 합니다.
- year shard
- 해당 shard 내부 local row index
- memmapped image row
- label metadata row

Horizon-specific filtering은 1-2에서 하지 않습니다.
`Ret_5d`, `Ret_20d`, `Ret_60d` 중 target을 고르는 1-3에서 추가합니다.

### Shuffling Rule

다음이 정의되기 전에는 base loaded data를 shuffle하지 않습니다.
- horizon label filtering
- split assignment
- normalization statistics

Training dataloader의 shuffle은 1-4에서 split을 고정한 뒤 training subset에만 허용합니다.

## 예정 Data Loader Interface

최종 구현은 두 층으로 나눕니다.

1. Shard metadata loader:
   - year pair 탐색
   - row count 검증
   - image path, label path, row count, date range 기록

2. Dataset object:
   - memmap에서 image lazy read
   - 1-3 이후 metadata와 label 반환
   - `(1, 64, 60)` shape의 tensor 반환

1-3 이후 예상 sample dictionary:

```python
{
    "image": Tensor[1, 64, 60],
    "label": int,
    "target_return": float,
    "date": Timestamp,
    "stock_id": str,
    "year": int,
    "local_row": int,
}
```

## 명시적 non-input columns

아래 column은 metadata 또는 label이지 image-model input이 아닙니다.
- `Date`
- `StockID`
- `MarketCap`
- `Ret_5d`
- `Ret_20d`
- `Ret_60d`
- `Ret_month`
- `EWMA_vol`

1단계 CNN baseline의 model input은 image tensor뿐입니다.

## 이후 체크리스트로 넘길 위험/미결정 항목

1-3으로 넘김:
- Horizon별 NaN filtering.
- zero-return label 처리 재확인.
- horizon별 prediction-output schema.

1-4로 넘김:
- train/validation/test split 구현.
- train-only normalization statistics.
- labels를 전체 로딩할지, filtering 이후 split 단위로 로딩할지.

1-6으로 넘김:
- DataLoader worker count.
- pin memory와 device transfer settings.
- performance tuning.
