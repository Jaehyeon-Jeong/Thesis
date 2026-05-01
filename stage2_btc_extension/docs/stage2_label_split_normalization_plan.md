# Stage 2 BTC Label, Split, and Normalization Plan

## English

Status: planning complete for checklist 2-4. Implementation happens later in
`2-I4`.

Source basis:
- Re-image summary: `자료조사/Re-image 요약.md`
  - line 13: predict signs of future 5/20/60-day returns from 5/20/60-day charts.
  - line 49: binary classification, cross-entropy, softmax Up probability,
    50% threshold, training-pixel mean/std normalization.
- Stage 1 label/split implementation:
  - `stage1_reimage_reproduction/src/stage1_reimage/data/label_split.py`
- Stage 2 data audit:
  - `stage2_btc_extension/reports/data_audit/btc_ohlcv_audit.md`
- Stage 2 image plan:
  - `stage2_btc_extension/docs/stage2_image_generation_plan.md`

Important difference from Stage 1:
- Stage 1 uses a stock panel and author-provided `Ret_5d`, `Ret_20d`, and
  `Ret_60d` columns.
- Stage 2 uses one BTC time series, so future returns must be constructed from
  BTC close prices.
- Stage 2 must use chronological split. Random train/validation split is not
  used for BTC because adjacent rolling samples overlap strongly.

## Label Rule

For image end date `t` and return horizon `R`:

```text
future_return_{t,R} = Close_{t+R} / Close_t - 1
label_{t,R} = 1 if future_return_{t,R} > 0 else 0
```

Interpretation:
- `1`: Up / positive future return.
- `0`: Down or non-positive future return.
- Exact zero return belongs to class `0`.

BTC horizon meaning:
- BTC trades every calendar day in the audited daily file.
- Therefore `R=5`, `R=20`, and `R=60` mean 5, 20, and 60 daily bars.

No-leakage rule:
- Image input uses only rows from `t-I+1` through `t`.
- MA uses only `Close_t` and earlier closes.
- Label uses `Close_{t+R}` only after the image has been fixed.
- Future return, label, and `Close_{t+R}` are metadata/targets only and never
  model inputs.

## Primary Split

Primary Stage 2 reporting is capped at `2024-12-31`.

Reason:
- The user-provided dataset target is the Kaggle BTC OHLCV dataset originally
  described as 2018-2024.
- The local file is auto-updated through 2026-03-16, but 2025-2026 is kept as an
  optional later holdout rather than mixed into the first BTC baseline.

Chronological split by signal date:

| Split | Signal-date range | Purpose |
| --- | --- | --- |
| Train | `2018-01-01` to `2020-12-31` | model fitting |
| Validation | `2021-01-01` to `2021-12-31` | early stopping/model selection |
| Test | `2022-01-01` to `2024-12-31` | primary out-of-sample reporting |

Purge rule:

```text
label_end_date <= split_signal_end
```

This means samples near the end of each split are dropped if their future-return
label crosses into the next split. For example, an `R60` training signal in
December 2020 is removed because its label would use prices in 2021.

Feature-window rule:
- A validation/test image may use historical prices before the split start
  because those prices are known at the signal date.
- This is not look-ahead leakage.
- The strict boundary is on future label information, handled by the purge rule.

## Split Counts

All counts below use the common full-MA eligible universe from 2-3. The four
image specifications share the same rows within a given `I/R` setting.

| Image | Horizon | Split | Samples | Positive rate | First signal date | Last signal date | Last label date |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| I5 | R5 | train | 1083 | 0.5374 | 2018-01-09 | 2020-12-26 | 2020-12-31 |
| I5 | R5 | validation | 360 | 0.5389 | 2021-01-01 | 2021-12-26 | 2021-12-31 |
| I5 | R5 | test | 1091 | 0.5243 | 2022-01-01 | 2024-12-26 | 2024-12-31 |
| I5 | R20 | train | 1068 | 0.5496 | 2018-01-09 | 2020-12-11 | 2020-12-31 |
| I5 | R20 | validation | 345 | 0.5246 | 2021-01-01 | 2021-12-11 | 2021-12-31 |
| I5 | R20 | test | 1076 | 0.5558 | 2022-01-01 | 2024-12-11 | 2024-12-31 |
| I5 | R60 | train | 1028 | 0.5068 | 2018-01-09 | 2020-11-01 | 2020-12-31 |
| I5 | R60 | validation | 305 | 0.6328 | 2021-01-01 | 2021-11-01 | 2021-12-31 |
| I5 | R60 | test | 1036 | 0.5483 | 2022-01-01 | 2024-11-01 | 2024-12-31 |
| I20 | R5 | train | 1053 | 0.5461 | 2018-02-08 | 2020-12-26 | 2020-12-31 |
| I20 | R5 | validation | 360 | 0.5389 | 2021-01-01 | 2021-12-26 | 2021-12-31 |
| I20 | R5 | test | 1091 | 0.5243 | 2022-01-01 | 2024-12-26 | 2024-12-31 |
| I20 | R20 | train | 1038 | 0.5568 | 2018-02-08 | 2020-12-11 | 2020-12-31 |
| I20 | R20 | validation | 345 | 0.5246 | 2021-01-01 | 2021-12-11 | 2021-12-31 |
| I20 | R20 | test | 1076 | 0.5558 | 2022-01-01 | 2024-12-11 | 2024-12-31 |
| I20 | R60 | train | 998 | 0.5220 | 2018-02-08 | 2020-11-01 | 2020-12-31 |
| I20 | R60 | validation | 305 | 0.6328 | 2021-01-01 | 2021-11-01 | 2021-12-31 |
| I20 | R60 | test | 1036 | 0.5483 | 2022-01-01 | 2024-11-01 | 2024-12-31 |
| I60 | R5 | train | 973 | 0.5468 | 2018-04-29 | 2020-12-26 | 2020-12-31 |
| I60 | R5 | validation | 360 | 0.5389 | 2021-01-01 | 2021-12-26 | 2021-12-31 |
| I60 | R5 | test | 1091 | 0.5243 | 2022-01-01 | 2024-12-26 | 2024-12-31 |
| I60 | R20 | train | 958 | 0.5699 | 2018-04-29 | 2020-12-11 | 2020-12-31 |
| I60 | R20 | validation | 345 | 0.5246 | 2021-01-01 | 2021-12-11 | 2021-12-31 |
| I60 | R20 | test | 1076 | 0.5558 | 2022-01-01 | 2024-12-11 | 2024-12-31 |
| I60 | R60 | train | 918 | 0.5468 | 2018-04-29 | 2020-11-01 | 2020-12-31 |
| I60 | R60 | validation | 305 | 0.6328 | 2021-01-01 | 2021-11-01 | 2021-12-31 |
| I60 | R60 | test | 1036 | 0.5483 | 2022-01-01 | 2024-11-01 | 2024-12-31 |

CSV artifact:
- `stage2_btc_extension/reports/tables/stage2_label_split_plan_counts.csv`

Note:
- Validation positive rate for `R60` is high because 2021 is a strong BTC market
  regime. This is a real split-balance property, not a data-loading error.

## Normalization Plan

Pixel normalization follows Stage 1:

```text
normalized_image = (image / 255.0 - train_pixel_mean) / train_pixel_std
```

Fit policy:
- Fit only on training images.
- Do not use validation or test pixels.
- Save normalization statistics separately for each experiment tuple:
  `(image_window, image_spec, return_horizon)`.
- Store `train_pixel_mean`, `train_pixel_std`, number of training images, number
  of pixels, and whether a smoke-test sample was used.

Why per experiment:
- Image size differs by `I5/I20/I60`.
- Pixel distribution differs by image spec, especially when volume bars and MA
  are present.
- Stage 1 already uses horizon-level training-only normalization.

## Required Metadata

Each BTC sample should preserve:
- `sample_id`
- `image_window`
- `image_spec`
- `return_horizon`
- `image_start_date`
- `image_end_date`
- `label_end_date`
- `source_start_index`
- `source_end_index`
- `label_exit_index`
- `entry_close`
- `exit_close`
- `future_return`
- `label`
- `split`

Model input:
- Only the image tensor is passed to the Stage 2 baseline CNN.
- Returns, labels, dates, and prices are metadata or targets only.

## 한국어

상태: checklist 2-4 계획 완료. 실제 구현은 이후 `2-I4`에서 합니다.

근거:
- Re-image 요약: `자료조사/Re-image 요약.md`
  - line 13: 5/20/60일 chart에서 미래 5/20/60일 수익률 부호 예측.
  - line 49: binary classification, cross-entropy, softmax Up probability,
    50% threshold, training-pixel mean/std normalization.
- Stage 1 label/split implementation:
  - `stage1_reimage_reproduction/src/stage1_reimage/data/label_split.py`
- Stage 2 data audit:
  - `stage2_btc_extension/reports/data_audit/btc_ohlcv_audit.md`
- Stage 2 image plan:
  - `stage2_btc_extension/docs/stage2_image_generation_plan.md`

Stage 1과 중요한 차이:
- Stage 1은 stock panel이고 저자가 제공한 `Ret_5d`, `Ret_20d`, `Ret_60d` column을
  사용합니다.
- Stage 2는 BTC 단일 시계열이므로 BTC close price에서 future return을 직접
  만들어야 합니다.
- BTC는 rolling sample 간 overlap이 크므로 시간순 split을 사용합니다. random
  train/validation split은 사용하지 않습니다.

## Label Rule

image 종료일을 `t`, return horizon을 `R`이라고 하면:

```text
future_return_{t,R} = Close_{t+R} / Close_t - 1
label_{t,R} = 1 if future_return_{t,R} > 0 else 0
```

해석:
- `1`: Up / positive future return.
- `0`: Down 또는 non-positive future return.
- 정확히 0인 return은 class `0`입니다.

BTC horizon 의미:
- audit한 daily file에서 BTC는 매일 거래됩니다.
- 따라서 `R=5`, `R=20`, `R=60`은 각각 5, 20, 60 daily bar입니다.

No-leakage rule:
- image input은 `t-I+1`부터 `t`까지의 row만 사용합니다.
- MA는 `Close_t`와 그 이전 close만 사용합니다.
- label은 image가 고정된 뒤 `Close_{t+R}`로 만듭니다.
- future return, label, `Close_{t+R}`는 metadata/target일 뿐 model input이 아닙니다.

## Primary Split

Stage 2 기본 보고는 `2024-12-31`까지로 제한합니다.

이유:
- 사용자 제공 dataset target은 2018-2024로 설명된 Kaggle BTC OHLCV dataset입니다.
- local file은 2026-03-16까지 auto-updated되어 있지만, 2025-2026은 첫 BTC baseline에
  섞지 않고 optional later holdout으로 남깁니다.

signal date 기준 시간순 split:

| Split | Signal-date range | 목적 |
| --- | --- | --- |
| Train | `2018-01-01` to `2020-12-31` | model fitting |
| Validation | `2021-01-01` to `2021-12-31` | early stopping/model selection |
| Test | `2022-01-01` to `2024-12-31` | primary out-of-sample reporting |

Purge rule:

```text
label_end_date <= split_signal_end
```

각 split 끝부분에서 future-return label이 다음 split로 넘어가는 sample은 제거합니다.
예를 들어 2020년 12월의 `R60` train signal은 label 계산에 2021년 가격을 사용하므로
train에서 제거됩니다.

Feature-window rule:
- validation/test image가 split 시작일 이전의 과거 가격을 image window에 포함할 수는
  있습니다. signal date에는 이미 알려진 과거 정보이므로 look-ahead leakage가 아닙니다.
- 엄격한 경계는 future label 정보이며, 이것은 purge rule로 처리합니다.

## Split Counts

아래 count는 2-3에서 정한 full-MA eligible common universe 기준입니다. 같은 `I/R`
setting 안에서 네 가지 image spec은 같은 row를 공유합니다.

| Image | Horizon | Split | Samples | Positive rate | 첫 signal date | 마지막 signal date | 마지막 label date |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| I5 | R5 | train | 1083 | 0.5374 | 2018-01-09 | 2020-12-26 | 2020-12-31 |
| I5 | R5 | validation | 360 | 0.5389 | 2021-01-01 | 2021-12-26 | 2021-12-31 |
| I5 | R5 | test | 1091 | 0.5243 | 2022-01-01 | 2024-12-26 | 2024-12-31 |
| I5 | R20 | train | 1068 | 0.5496 | 2018-01-09 | 2020-12-11 | 2020-12-31 |
| I5 | R20 | validation | 345 | 0.5246 | 2021-01-01 | 2021-12-11 | 2021-12-31 |
| I5 | R20 | test | 1076 | 0.5558 | 2022-01-01 | 2024-12-11 | 2024-12-31 |
| I5 | R60 | train | 1028 | 0.5068 | 2018-01-09 | 2020-11-01 | 2020-12-31 |
| I5 | R60 | validation | 305 | 0.6328 | 2021-01-01 | 2021-11-01 | 2021-12-31 |
| I5 | R60 | test | 1036 | 0.5483 | 2022-01-01 | 2024-11-01 | 2024-12-31 |
| I20 | R5 | train | 1053 | 0.5461 | 2018-02-08 | 2020-12-26 | 2020-12-31 |
| I20 | R5 | validation | 360 | 0.5389 | 2021-01-01 | 2021-12-26 | 2021-12-31 |
| I20 | R5 | test | 1091 | 0.5243 | 2022-01-01 | 2024-12-26 | 2024-12-31 |
| I20 | R20 | train | 1038 | 0.5568 | 2018-02-08 | 2020-12-11 | 2020-12-31 |
| I20 | R20 | validation | 345 | 0.5246 | 2021-01-01 | 2021-12-11 | 2021-12-31 |
| I20 | R20 | test | 1076 | 0.5558 | 2022-01-01 | 2024-12-11 | 2024-12-31 |
| I20 | R60 | train | 998 | 0.5220 | 2018-02-08 | 2020-11-01 | 2020-12-31 |
| I20 | R60 | validation | 305 | 0.6328 | 2021-01-01 | 2021-11-01 | 2021-12-31 |
| I20 | R60 | test | 1036 | 0.5483 | 2022-01-01 | 2024-11-01 | 2024-12-31 |
| I60 | R5 | train | 973 | 0.5468 | 2018-04-29 | 2020-12-26 | 2020-12-31 |
| I60 | R5 | validation | 360 | 0.5389 | 2021-01-01 | 2021-12-26 | 2021-12-31 |
| I60 | R5 | test | 1091 | 0.5243 | 2022-01-01 | 2024-12-26 | 2024-12-31 |
| I60 | R20 | train | 958 | 0.5699 | 2018-04-29 | 2020-12-11 | 2020-12-31 |
| I60 | R20 | validation | 345 | 0.5246 | 2021-01-01 | 2021-12-11 | 2021-12-31 |
| I60 | R20 | test | 1076 | 0.5558 | 2022-01-01 | 2024-12-11 | 2024-12-31 |
| I60 | R60 | train | 918 | 0.5468 | 2018-04-29 | 2020-11-01 | 2020-12-31 |
| I60 | R60 | validation | 305 | 0.6328 | 2021-01-01 | 2021-11-01 | 2021-12-31 |
| I60 | R60 | test | 1036 | 0.5483 | 2022-01-01 | 2024-11-01 | 2024-12-31 |

CSV artifact:
Need more output? Already enough. Need maybe continue doc lower after. Fine. Need maybe incorporate into other docs. Already patch later. Need update README link. Need maybe create "reports/tables/README"? It may not exist. Let's check. Need maybe in artifact. Could create. Also update reports/README. Need include small CSV in publish. Need maybe create docs mention. We'll patch README and source_map/stage2 pipeline. Need create stage2 reports/tables README maybe optional. The folder exists now with csv and no README? Need update maybe. Use apply patch. Also maybe check csv head. Need maybe add to README docs. Let's patch. 
