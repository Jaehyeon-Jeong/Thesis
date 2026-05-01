# Stage 2 BTC Label, Split, and Normalization Plan

## English

Status: planning complete for checklist 2-4. This file was revised after
reviewing the Stage 1 paper-style train/validation split more carefully.
Implementation happens later in `2-I4`.

Source basis:
- Re-image summary: `자료조사/Re-image 요약.md`
  - line 41: 1993-2000 train/validation, 2001-2019 test, and a 70/30 random
    split inside the eight-year train/validation period.
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
- The default Stage 2 split still follows the paper-style principle: first
  separate a pre-test train/validation period from an out-of-sample test period,
  then randomly split the train/validation period 70/30.
- Because BTC is a single rolling time series, adjacent validation samples can
  overlap heavily with adjacent training samples. This is recorded as a
  limitation, and a chronological-validation variant can be added later as a
  robustness check. It is not the default Stage 2 baseline.

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

Paper-style Stage 2 split:

| Component | Signal-date range | Rule | Purpose |
| --- | --- | --- | --- |
| Train/validation pool | `2018-01-01` to `2020-12-31` | 70/30 random split after eligibility filtering | model fitting and early stopping |
| Test | `2021-01-01` to `2024-12-31` | chronological holdout | primary out-of-sample reporting |

Train/validation random split:
- Split unit: BTC rolling chart sample.
- Train ratio: `0.70`.
- Validation ratio: `0.30`.
- Split seed: `42`.
- Random generator: `numpy.random.default_rng(42)`, matching the Stage 1
  implementation style.
- Stratification: not used by default because the paper summary only reports a
  random 70/30 split.
- Class balance is still reported after the split.

Purge rule:

```text
label_end_date <= split_signal_end
```

This means:
- Train/validation-pool samples are kept only if their label also ends by
  `2020-12-31`.
- Test samples are kept only if their label ends by `2024-12-31`.
- The model never trains on a sample whose target reaches into the test period.

Feature-window rule:
- A test image may use historical prices before `2021-01-01` if its signal date
  is in the test period, because those historical prices are known at the signal
  date.
- This is not look-ahead leakage.
- The strict boundary is on future label information, handled by the purge rule.

BTC-specific caution:
- Random train/validation inside `2018-2020` is paper-aligned.
- However, BTC rolling windows are highly overlapping because they are sampled
  from one asset. Validation loss should therefore be interpreted as an early
  stopping/model-selection signal, not as the final generalization result.
- Final reporting relies on the chronological holdout test period `2021-2024`.

## Split Counts

All counts below use the common full-MA eligible universe from 2-3. The four
image specifications share the same rows within a given `I/R` setting.

| Image | Horizon | Split | Samples | Positive rate | First signal date | Last signal date | Last label date |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| I5 | R5 | train | 758 | 0.5396 | 2018-01-09 | 2020-12-25 | 2020-12-30 |
| I5 | R5 | validation | 325 | 0.5323 | 2018-01-13 | 2020-12-26 | 2020-12-31 |
| I5 | R5 | test | 1456 | 0.5261 | 2021-01-01 | 2024-12-26 | 2024-12-31 |
| I5 | R20 | train | 748 | 0.5481 | 2018-01-09 | 2020-12-11 | 2020-12-31 |
| I5 | R20 | validation | 320 | 0.5531 | 2018-01-10 | 2020-11-29 | 2020-12-19 |
| I5 | R20 | test | 1441 | 0.5413 | 2021-01-01 | 2024-12-11 | 2024-12-31 |
| I5 | R60 | train | 720 | 0.5042 | 2018-01-09 | 2020-11-01 | 2020-12-31 |
| I5 | R60 | validation | 308 | 0.5130 | 2018-01-10 | 2020-10-26 | 2020-12-25 |
| I5 | R60 | test | 1401 | 0.5432 | 2021-01-01 | 2024-11-01 | 2024-12-31 |
| I20 | R5 | train | 737 | 0.5495 | 2018-02-08 | 2020-12-26 | 2020-12-31 |
| I20 | R5 | validation | 316 | 0.5380 | 2018-02-09 | 2020-12-14 | 2020-12-19 |
| I20 | R5 | test | 1456 | 0.5261 | 2021-01-01 | 2024-12-26 | 2024-12-31 |
| I20 | R20 | train | 727 | 0.5433 | 2018-02-08 | 2020-12-11 | 2020-12-31 |
| I20 | R20 | validation | 311 | 0.5884 | 2018-02-09 | 2020-12-06 | 2020-12-26 |
| I20 | R20 | test | 1441 | 0.5413 | 2021-01-01 | 2024-12-11 | 2024-12-31 |
| I20 | R60 | train | 699 | 0.5293 | 2018-02-08 | 2020-10-31 | 2020-12-30 |
| I20 | R60 | validation | 299 | 0.5050 | 2018-02-09 | 2020-11-01 | 2020-12-31 |
| I20 | R60 | test | 1401 | 0.5432 | 2021-01-01 | 2024-11-01 | 2024-12-31 |
| I60 | R5 | train | 681 | 0.5360 | 2018-04-29 | 2020-12-25 | 2020-12-30 |
| I60 | R5 | validation | 292 | 0.5719 | 2018-04-30 | 2020-12-26 | 2020-12-31 |
| I60 | R5 | test | 1456 | 0.5261 | 2021-01-01 | 2024-12-26 | 2024-12-31 |
| I60 | R20 | train | 671 | 0.5499 | 2018-04-29 | 2020-12-10 | 2020-12-30 |
| I60 | R20 | validation | 287 | 0.6167 | 2018-04-30 | 2020-12-11 | 2020-12-31 |
| I60 | R20 | test | 1441 | 0.5413 | 2021-01-01 | 2024-12-11 | 2024-12-31 |
| I60 | R60 | train | 643 | 0.5537 | 2018-04-29 | 2020-10-31 | 2020-12-30 |
| I60 | R60 | validation | 275 | 0.5309 | 2018-04-30 | 2020-11-01 | 2020-12-31 |
| I60 | R60 | test | 1401 | 0.5432 | 2021-01-01 | 2024-11-01 | 2024-12-31 |

CSV artifact:
- `stage2_btc_extension/reports/tables/stage2_label_split_plan_counts.csv`

## Normalization Plan

Pixel normalization follows Stage 1:

```text
normalized_image = (image / 255.0 - train_pixel_mean) / train_pixel_std
```

Fit policy:
- Fit only on the 70% training subset.
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

상태: checklist 2-4 계획 완료. Stage 1의 논문식 train/validation split을 다시
확인한 뒤 이 파일을 수정했습니다. 실제 구현은 이후 `2-I4`에서 합니다.

근거:
- Re-image 요약: `자료조사/Re-image 요약.md`
  - line 41: 1993-2000 train/validation, 2001-2019 test, 8년
    train/validation 구간 내부 70/30 random split.
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
- Stage 2 기본 split은 논문식 원칙을 유지합니다. 먼저 pre-test train/validation
  기간과 out-of-sample test 기간을 분리하고, train/validation 기간 내부를 70/30
  random split합니다.
- 다만 BTC는 단일 rolling time series라서 인접 validation sample과 train sample이
  많이 겹칠 수 있습니다. 이 점은 제한사항으로 기록하고, chronological validation은
  나중 robustness check로 추가할 수 있습니다. 기본 Stage 2 baseline은 아닙니다.

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

논문식 Stage 2 split:

| Component | Signal-date range | Rule | 목적 |
| --- | --- | --- | --- |
| Train/validation pool | `2018-01-01` to `2020-12-31` | eligible sample을 70/30 random split | model fitting과 early stopping |
| Test | `2021-01-01` to `2024-12-31` | chronological holdout | primary out-of-sample reporting |

Train/validation random split:
- Split unit: BTC rolling chart sample.
- Train ratio: `0.70`.
- Validation ratio: `0.30`.
- Split seed: `42`.
- Random generator: Stage 1 구현 방식에 맞춰 `numpy.random.default_rng(42)`를 사용합니다.
- Stratification: 논문 요약은 random 70/30만 보고하므로 기본적으로 사용하지 않습니다.
- split 이후 class balance는 반드시 보고합니다.

Purge rule:

```text
label_end_date <= split_signal_end
```

의미:
- Train/validation pool sample은 label 종료일도 `2020-12-31` 이하여야 합니다.
- Test sample은 label 종료일도 `2024-12-31` 이하여야 합니다.
- 모델은 target이 test period로 넘어가는 sample로 학습하지 않습니다.

Feature-window rule:
- test image의 signal date가 test period 안에 있다면, image window가 `2021-01-01`
  이전의 과거 가격을 포함해도 됩니다. signal date 기준으로 이미 알려진 과거 정보이므로
  look-ahead leakage가 아닙니다.
- 엄격한 경계는 future label 정보이며, 이것은 purge rule로 처리합니다.

BTC-specific caution:
- `2018-2020` 내부 random train/validation은 논문 방식에 더 가깝습니다.
- 하지만 BTC rolling window는 단일 자산에서 나오므로 서로 많이 겹칩니다. 따라서
  validation loss는 early stopping/model-selection 신호로 해석하고, 최종 일반화 성능은
  chronological holdout test period `2021-2024`에서 봅니다.

## Split Counts

아래 count는 2-3에서 정한 full-MA eligible common universe 기준입니다. 같은 `I/R`
setting 안에서 네 가지 image spec은 같은 row를 공유합니다.

| Image | Horizon | Split | Samples | Positive rate | 첫 signal date | 마지막 signal date | 마지막 label date |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| I5 | R5 | train | 758 | 0.5396 | 2018-01-09 | 2020-12-25 | 2020-12-30 |
| I5 | R5 | validation | 325 | 0.5323 | 2018-01-13 | 2020-12-26 | 2020-12-31 |
| I5 | R5 | test | 1456 | 0.5261 | 2021-01-01 | 2024-12-26 | 2024-12-31 |
| I5 | R20 | train | 748 | 0.5481 | 2018-01-09 | 2020-12-11 | 2020-12-31 |
| I5 | R20 | validation | 320 | 0.5531 | 2018-01-10 | 2020-11-29 | 2020-12-19 |
| I5 | R20 | test | 1441 | 0.5413 | 2021-01-01 | 2024-12-11 | 2024-12-31 |
| I5 | R60 | train | 720 | 0.5042 | 2018-01-09 | 2020-11-01 | 2020-12-31 |
| I5 | R60 | validation | 308 | 0.5130 | 2018-01-10 | 2020-10-26 | 2020-12-25 |
| I5 | R60 | test | 1401 | 0.5432 | 2021-01-01 | 2024-11-01 | 2024-12-31 |
| I20 | R5 | train | 737 | 0.5495 | 2018-02-08 | 2020-12-26 | 2020-12-31 |
| I20 | R5 | validation | 316 | 0.5380 | 2018-02-09 | 2020-12-14 | 2020-12-19 |
| I20 | R5 | test | 1456 | 0.5261 | 2021-01-01 | 2024-12-26 | 2024-12-31 |
| I20 | R20 | train | 727 | 0.5433 | 2018-02-08 | 2020-12-11 | 2020-12-31 |
| I20 | R20 | validation | 311 | 0.5884 | 2018-02-09 | 2020-12-06 | 2020-12-26 |
| I20 | R20 | test | 1441 | 0.5413 | 2021-01-01 | 2024-12-11 | 2024-12-31 |
| I20 | R60 | train | 699 | 0.5293 | 2018-02-08 | 2020-10-31 | 2020-12-30 |
| I20 | R60 | validation | 299 | 0.5050 | 2018-02-09 | 2020-11-01 | 2020-12-31 |
| I20 | R60 | test | 1401 | 0.5432 | 2021-01-01 | 2024-11-01 | 2024-12-31 |
| I60 | R5 | train | 681 | 0.5360 | 2018-04-29 | 2020-12-25 | 2020-12-30 |
| I60 | R5 | validation | 292 | 0.5719 | 2018-04-30 | 2020-12-26 | 2020-12-31 |
| I60 | R5 | test | 1456 | 0.5261 | 2021-01-01 | 2024-12-26 | 2024-12-31 |
| I60 | R20 | train | 671 | 0.5499 | 2018-04-29 | 2020-12-10 | 2020-12-30 |
| I60 | R20 | validation | 287 | 0.6167 | 2018-04-30 | 2020-12-11 | 2020-12-31 |
| I60 | R20 | test | 1441 | 0.5413 | 2021-01-01 | 2024-12-11 | 2024-12-31 |
| I60 | R60 | train | 643 | 0.5537 | 2018-04-29 | 2020-10-31 | 2020-12-30 |
| I60 | R60 | validation | 275 | 0.5309 | 2018-04-30 | 2020-11-01 | 2020-12-31 |
| I60 | R60 | test | 1401 | 0.5432 | 2021-01-01 | 2024-11-01 | 2024-12-31 |

CSV artifact:
- `stage2_btc_extension/reports/tables/stage2_label_split_plan_counts.csv`

## Normalization Plan

Pixel normalization follows Stage 1:

```text
normalized_image = (image / 255.0 - train_pixel_mean) / train_pixel_std
```

Fit policy:
- 70% training subset에서만 fit합니다.
- validation/test pixel은 사용하지 않습니다.
- `(image_window, image_spec, return_horizon)` experiment별로 stats를 따로 저장합니다.
- `train_pixel_mean`, `train_pixel_std`, training image 수, pixel 수, smoke-test sample
  사용 여부를 저장합니다.

Experiment별로 따로 저장하는 이유:
- Image size가 `I5/I20/I60`마다 다릅니다.
- Pixel distribution이 image spec마다 다릅니다. 특히 volume bar와 MA line 포함 여부가
  영향을 줍니다.
- Stage 1도 horizon별 train-only normalization을 사용합니다.

## Required Metadata

각 BTC sample은 다음 metadata를 보존해야 합니다:
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
- Stage 2 baseline CNN에는 image tensor만 들어갑니다.
- return, label, date, price는 metadata 또는 target일 뿐입니다.
