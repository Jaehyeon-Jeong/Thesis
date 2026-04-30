# monthly_20d Data Check

## English

Status: 0-1 through 0-5 completed.

This file will be filled one sub-step at a time.

## 0-1. Local File Structure Check

Checked path:
- `../테스트/Test/img_data/monthly_20d`

Saved inventory:
- `data_inventory/monthly20_files.csv`

Findings:
- The local `monthly_20d` folder exists.
- It contains 54 data files:
  - 27 image `.dat` files
  - 27 label `.feather` files
- The available years are 1993 through 2019.
- Each year appears to have one image file and one label file.
- All listed data files contain `20d` in the file name.
- No local file names indicate 5-day image shards.
- No local file names indicate 60-day image shards.
- File names include `has_vb` and `20_ma`, but the actual image content
  still needs to be checked in 0-3.

Interpretation for Stage 1:
- The local public stock image data currently supports direct checks for
  20-day images.
- 5-day and 60-day stock-image empirical reproduction cannot be assumed from
  this local folder.

## 0-2. Label File Check

Checked files:
- 27 files matching `*_labels_w_delay.feather`
- Label column description file: `../테스트/Test/img_data/label_columns.txt`

Saved inventories:
- `data_inventory/monthly20_labels_by_year.csv`
- `data_inventory/monthly20_labels.csv`

Observed columns:
- `Date`
- `StockID`
- `MarketCap`
- `Ret_5d`
- `Ret_20d`
- `Ret_60d`
- `Ret_month`
- `EWMA_vol`

Column meanings from `label_columns.txt`:
- `Date`: last day of the 20-day rolling window for the chart.
- `StockID`: CRSP PERMNO.
- `MarketCap`: market capitalization in dollars, recorded in thousands.
- `Ret_{t}d`: future holding-period return for t = 5, 20, 60.
- `Ret_month`: holding-period return from current month-end to next month-end.
- `EWMA_vol`: exponentially weighted volatility with alpha 0.05; one-day delay is included.

Row and horizon findings:
- Total label rows across 1993-2019: 2,196,994.
- All 27 label files have the same columns.
- Available future-return label columns are `Ret_5d`, `Ret_20d`, `Ret_60d`, and `Ret_month`.
- Therefore the public `monthly_20d` image data can support multiple horizons
  from the same 20-day image: `I20/R5`, `I20/R20`, `I20/R60`, and possibly
  `I20/Ret_month`.

Missing-value findings:
- `Ret_5d`: 6,270 null rows.
- `Ret_20d`: 16,384 null rows.
- `Ret_60d`: 45,781 null rows.
- `Ret_month`: 22,736 null rows.

Implication for Stage 1:
- The binary label should be constructed after filtering out rows where the
  chosen return column is missing.
- For each horizon, the valid sample count is different.
- The label rule remains: `1` if the chosen future return is greater than zero,
  otherwise `0`.

## 0-3. Image Content Check

Checked files:
- 27 files matching `*_images.dat`

Saved outputs:
- `data_inventory/monthly20_image_shape_check.csv`
- `data_inventory/monthly20_sample_images_2019.csv`
- `outputs/figures/sample_images/monthly20_2019_sample_contact_sheet.png`
- Individual sample PNG files under `outputs/figures/sample_images/`

Shape findings:
- Every `.dat` image file can be reshaped as `(N, 64, 60)`.
- For every year, `N` inferred from the `.dat` byte size matches the number of
  rows in the corresponding `.feather` label file.
- Therefore the local public images are confirmed as 20-day images with shape
  `64 x 60`.

Content findings from sample images:
- The file names include `has_vb` and `20_ma`.
- Sample images show the expected black background and white chart elements.
- The bottom rows contain white pixels, consistent with a lower volume-bar area.
- A white moving-average-like line is visually present in the samples, but this
  is an image-level visual check rather than a reconstruction from raw OHLC.

Implication for Stage 1:
- The available author image data should be treated as the full 20-day
  specification: `I20` with OHLC + 20-day MA + volume bars.
- The ablation specs A/B/C/D cannot be recovered from this `.dat` alone unless
  separate raw OHLCV or separate pre-rendered ablation images are obtained.

## 0-5. Stage 1 Feasibility Summary

Stage 1 goal:
- Reproduce the Re-image paper pipeline as accurately as possible before moving to BTC.
- Use the paper/local summary and `lich99/Stock_CNN` as implementation references.
- Keep all paper/GitHub mismatches documented before implementation decisions.

What is directly feasible with the current local data:

| Item | Feasible? | Reason |
| --- | --- | --- |
| Author-provided stock image loading | Yes | All `.dat` files reshape to `(N, 64, 60)`, and image counts match label rows. |
| 20-day full-spec image experiment | Yes | File names and sample images indicate `20d + 20_ma + has_vb`; visual check confirms MA-like line and volume bars. |
| `I20/R5` | Yes | `Ret_5d` exists; valid non-null rows: 2,190,724. |
| `I20/R20` | Yes | `Ret_20d` exists; valid non-null rows: 2,180,610. |
| `I20/R60` | Yes | `Ret_60d` exists; valid non-null rows: 2,151,213. |
| 1993-2000 train/validation and 2001-2019 test split | Yes | Annual files exist from 1993 through 2019. |
| I20 CNN architecture following `Stock_CNN` | Yes | `models/baseline.py` was checked and saved under `references/`. |
| Cross-sectional stock prediction outputs | Yes | Labels contain `Date`, `StockID`, `MarketCap`, and future stock returns. |
| Grad-CAM Figure 13-style output for I20/R20 | Yes, after training | 2019 images exist; Grad-CAM requires a trained model and target class logits. |

What is not directly feasible with the current local data:

| Item | Feasible? | Blocker |
| --- | --- | --- |
| Direct `I5` stock reproduction | No | No local 5-day rendered stock image shard was found. |
| Direct `I60` stock reproduction | No | No local 60-day rendered stock image shard was found. |
| A/B/C/D image-spec ablation from current `.dat` | No | Current images are already rendered as full `OHLC + MA + Volume`; components cannot be separated from the bitmap. |
| Exact image generator validation against CRSP raw OHLCV | No | Current folder contains rendered images and labels, not the raw CRSP OHLCV path used to render them. |
| Exact paper-wide reproduction across all windows/specs | Not with current data alone | Requires 5-day/60-day image shards, raw OHLCV, or additional author data. |

Available Stage 1 label targets:

| Target | Meaning | Valid rows | Positive rate among valid rows | Stage 1 status |
| --- | --- | ---: | ---: | --- |
| `Ret_5d` | Future 5-trading-day holding-period return | 2,190,724 | 50.48% | Main feasible target: `I20/R5` |
| `Ret_20d` | Future 20-trading-day holding-period return | 2,180,610 | 51.23% | Main feasible target: `I20/R20` |
| `Ret_60d` | Future 60-trading-day holding-period return | 2,151,213 | 53.17% | Main feasible target: `I20/R60` |
| `Ret_month` | Month-end to next month-end holding-period return | 2,174,258 | 52.18% | Optional only; not one of the fixed R=5/20/60 horizons. |

Stage 1 implementation boundary:
- Start with the author-provided `monthly_20d` rendered images.
- Treat the image tensor convention as `(batch, channel, height=64, width=60)`.
- Use binary labels: `1` if selected future return is greater than zero, otherwise `0`.
- Filter missing return rows separately for each horizon before constructing labels.
- Follow `lich99/Stock_CNN/models/baseline.py` for the I20 model core unless the user decides otherwise.
- Train/evaluate separately for `I20/R5`, `I20/R20`, and `I20/R60`.
- Save predictions with at least `Date`, `StockID`, target return, binary label, predicted class, and up probability.
- Generate Grad-CAM only after training, and label it as a class-discriminative heatmap rather than a raw feature map.

Stage 1 reporting language:
- The current Stage 1 reproduction is a public-data reproduction of the author-provided 20-day full-spec image pipeline.
- It is not a complete reproduction of every Re-image paper window and ablation because the local public data does not include 5-day images, 60-day images, or separated image-spec variants.
- This limitation must be stated before comparing Stage 1 results with the full paper tables.

## 한국어

상태: 0-1부터 0-5까지 완료.

이 파일은 0단계 세부 작업을 하나씩 진행하면서 채웁니다.

## 0-1. 로컬 파일 구조 확인

확인한 경로:
- `../테스트/Test/img_data/monthly_20d`

저장한 파일 목록:
- `data_inventory/monthly20_files.csv`

확인 결과:
- 로컬 `monthly_20d` 폴더가 존재합니다.
- 총 54개 데이터 파일이 있습니다.
  - image `.dat` 파일 27개
  - label `.feather` 파일 27개
- 사용 가능한 연도는 1993년부터 2019년까지입니다.
- 각 연도마다 image 파일 1개와 label 파일 1개가 있는 구조로 보입니다.
- 모든 데이터 파일 이름에는 `20d`가 들어 있습니다.
- 로컬 파일명 기준으로 5일 image shard는 보이지 않습니다.
- 로컬 파일명 기준으로 60일 image shard도 보이지 않습니다.
- 파일명에는 `has_vb`, `20_ma`가 포함되어 있지만, 실제 이미지 안에 volume과 MA가 보이는지는 0-3에서 확인해야 합니다.

1단계 관련 해석:
- 현재 로컬 공개 stock image 데이터는 20-day image 확인과 재현에 직접 사용할 수 있습니다.
- 이 폴더만으로 5-day, 60-day stock image 실증 재현이 가능하다고 가정하면 안 됩니다.

## 0-2. 라벨 파일 확인

확인한 파일:
- `*_labels_w_delay.feather` 형식의 label 파일 27개
- label 설명 파일: `../테스트/Test/img_data/label_columns.txt`

저장한 파일:
- `data_inventory/monthly20_labels_by_year.csv`
- `data_inventory/monthly20_labels.csv`

확인된 column:
- `Date`
- `StockID`
- `MarketCap`
- `Ret_5d`
- `Ret_20d`
- `Ret_60d`
- `Ret_month`
- `EWMA_vol`

`label_columns.txt` 기준 column 의미:
- `Date`: 20일 rolling window chart의 마지막 날짜입니다.
- `StockID`: CRSP PERMNO입니다.
- `MarketCap`: 천 달러 단위 market capitalization입니다.
- `Ret_{t}d`: t = 5, 20, 60일 future holding-period return입니다.
- `Ret_month`: 현재 month-end부터 다음 month-end까지의 holding-period return입니다.
- `EWMA_vol`: alpha 0.05의 지수 가중 변동성이며, one-day delay가 포함되어 있습니다.

row 및 horizon 확인:
- 1993-2019 전체 label row 수는 2,196,994개입니다.
- 27개 label 파일 모두 column 구성이 같습니다.
- 사용 가능한 미래수익률 column은 `Ret_5d`, `Ret_20d`, `Ret_60d`, `Ret_month`입니다.
- 따라서 공개 `monthly_20d` image 하나로 여러 horizon을 붙일 수 있습니다.
  - `I20/R5`
  - `I20/R20`
  - `I20/R60`
  - 필요하면 `I20/Ret_month`

결측치 확인:
- `Ret_5d`: 결측 6,270개
- `Ret_20d`: 결측 16,384개
- `Ret_60d`: 결측 45,781개
- `Ret_month`: 결측 22,736개

1단계 관련 해석:
- binary label은 선택한 return column의 결측 row를 제거한 뒤 만들어야 합니다.
- horizon마다 유효 sample 수가 다릅니다.
- label rule은 그대로 유지합니다: 선택한 future return이 0보다 크면 `1`, 아니면 `0`.

## 0-3. 이미지 내용 확인

확인한 파일:
- `*_images.dat` 형식의 image 파일 27개

저장한 산출물:
- `data_inventory/monthly20_image_shape_check.csv`
- `data_inventory/monthly20_sample_images_2019.csv`
- `outputs/figures/sample_images/monthly20_2019_sample_contact_sheet.png`
- `outputs/figures/sample_images/` 아래 개별 sample PNG 파일

shape 확인:
- 모든 `.dat` image 파일은 `(N, 64, 60)`으로 reshape 가능합니다.
- 모든 연도에서 `.dat` byte size로 추정한 image 수와 대응되는 `.feather` label row 수가 정확히 일치합니다.
- 따라서 로컬 공개 이미지는 `64 x 60` 크기의 20-day image로 확인됩니다.

sample image 확인:
- 파일명에는 `has_vb`, `20_ma`가 포함되어 있습니다.
- sample image는 검정 배경과 흰색 chart element 형태로 보입니다.
- 이미지 하단 row에도 흰 픽셀이 존재해서 volume bar 영역이 있는 것으로 보입니다.
- sample image에서 moving-average처럼 보이는 흰 선이 확인됩니다. 다만 이것은 raw OHLC로 재구성한 검증이 아니라 이미지 자체를 눈으로 확인한 결과입니다.

1단계 관련 해석:
- 현재 저자 공개 image data는 `I20`의 full specification, 즉 OHLC + 20-day MA + volume bar 포함 이미지로 보는 것이 타당합니다.
- 이 `.dat` 파일만으로는 OHLC only, OHLC+MA, OHLC+Volume, OHLC+MA+Volume 네 가지 ablation을 다시 분리할 수 없습니다.
- A/B/C/D ablation을 하려면 원시 OHLCV로 이미지를 다시 만들거나, 별도 pre-rendered ablation image data가 필요합니다.

## 0-5. 1단계 가능 범위 요약

1단계 목표:
- BTC로 넘어가기 전에 Re-image 논문 파이프라인을 최대한 정확하게 재현합니다.
- 논문/로컬 요약과 `lich99/Stock_CNN`을 구현 근거로 사용합니다.
- 논문과 GitHub 구현이 다른 부분은 구현 전에 문서화하고 결정합니다.

현재 로컬 데이터로 직접 가능한 것:

| 항목 | 가능 여부 | 이유 |
| --- | --- | --- |
| 저자 제공 stock image loading | 가능 | 모든 `.dat` 파일이 `(N, 64, 60)`으로 reshape되고, image 수와 label row 수가 일치합니다. |
| 20-day full-spec image 실험 | 가능 | 파일명과 sample image 기준으로 `20d + 20_ma + has_vb`이며, MA처럼 보이는 선과 volume bar가 확인됩니다. |
| `I20/R5` | 가능 | `Ret_5d`가 있고, 결측 제거 후 유효 row는 2,190,724개입니다. |
| `I20/R20` | 가능 | `Ret_20d`가 있고, 결측 제거 후 유효 row는 2,180,610개입니다. |
| `I20/R60` | 가능 | `Ret_60d`가 있고, 결측 제거 후 유효 row는 2,151,213개입니다. |
| 1993-2000 train/validation, 2001-2019 test split | 가능 | 1993년부터 2019년까지 연도별 파일이 있습니다. |
| `Stock_CNN` 기준 I20 CNN 구조 | 가능 | `models/baseline.py`를 확인했고 `references/` 아래에 저장했습니다. |
| 개별주 cross-sectional prediction output | 가능 | label에 `Date`, `StockID`, `MarketCap`, future return이 있습니다. |
| I20/R20 Figure 13 스타일 Grad-CAM | 학습 후 가능 | 2019년 image가 있고, Grad-CAM은 학습된 모델과 target class logit이 있어야 만들 수 있습니다. |

현재 로컬 데이터로 직접 불가능한 것:

| 항목 | 가능 여부 | 막히는 이유 |
| --- | --- | --- |
| `I5` stock 재현 | 불가능 | 로컬에 5-day rendered stock image shard가 없습니다. |
| `I60` stock 재현 | 불가능 | 로컬에 60-day rendered stock image shard가 없습니다. |
| 현재 `.dat`에서 A/B/C/D image-spec ablation 분리 | 불가능 | 현재 이미지는 이미 `OHLC + MA + Volume` full bitmap으로 렌더링되어 있어서 구성요소를 분리할 수 없습니다. |
| CRSP raw OHLCV 기준 image generator 정확 검증 | 불가능 | 현재 폴더에는 렌더링된 image와 label만 있고, 렌더링 전 raw CRSP OHLCV가 없습니다. |
| 논문 전체 window/spec 완전 재현 | 현재 데이터만으로는 불가능 | 5-day/60-day image shard, raw OHLCV, 또는 추가 저자 데이터가 필요합니다. |

1단계에서 사용할 수 있는 label target:

| Target | 의미 | 유효 row | 유효 row 중 positive 비율 | 1단계 상태 |
| --- | --- | ---: | ---: | --- |
| `Ret_5d` | 미래 5거래일 보유수익률 | 2,190,724 | 50.48% | 주 실험 가능: `I20/R5` |
| `Ret_20d` | 미래 20거래일 보유수익률 | 2,180,610 | 51.23% | 주 실험 가능: `I20/R20` |
| `Ret_60d` | 미래 60거래일 보유수익률 | 2,151,213 | 53.17% | 주 실험 가능: `I20/R60` |
| `Ret_month` | 현재 month-end부터 다음 month-end까지 보유수익률 | 2,174,258 | 52.18% | 선택 가능하지만 고정 R=5/20/60 실험은 아님 |

1단계 구현 경계:
- 먼저 저자 제공 `monthly_20d` rendered image를 사용합니다.
- tensor convention은 `(batch, channel, height=64, width=60)`으로 고정합니다.
- binary label은 선택한 future return이 0보다 크면 `1`, 아니면 `0`입니다.
- horizon별로 결측 return row를 제거한 뒤 label을 만듭니다.
- 사용자가 다르게 결정하지 않는 한 I20 model core는 `lich99/Stock_CNN/models/baseline.py`를 따릅니다.
- `I20/R5`, `I20/R20`, `I20/R60`을 각각 따로 train/evaluate합니다.
- prediction 저장 시 최소한 `Date`, `StockID`, target return, binary label, predicted class, up probability를 포함해야 합니다.
- Grad-CAM은 학습 이후에만 생성하고, raw feature map이 아니라 class-discriminative heatmap이라고 명시합니다.

1단계 보고 문장:
- 현재 1단계 재현은 저자 공개 20-day full-spec image data를 이용한 public-data reproduction입니다.
- 로컬 공개 데이터에는 5-day image, 60-day image, 분리된 image-spec variant가 없으므로 Re-image 논문의 모든 window와 ablation을 완전히 재현하는 것은 아닙니다.
- 이 제한은 논문 전체 table과 직접 비교하기 전에 반드시 명시해야 합니다.
