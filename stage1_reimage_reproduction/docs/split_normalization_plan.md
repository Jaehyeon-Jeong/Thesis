# Stage 1 Split and Normalization Detail Plan

## English

Status:
- Stage 1-4 completed as a detail plan.
- No split or normalization code has been implemented yet.

## Purpose

Define the train/validation/test split and image normalization rules for Stage 1
before model implementation and training.

## Paper Reference

From the local Re-image summary:
- Train/validation period: 1993-2000.
- Test period: 2001-2019.
- Train/validation split: 70/30 random split inside the 1993-2000 period.
- Images are standardized using the training-data pixel mean and standard
  deviation.
- The paper summary notes that the exact random seed is not reported.

Stage 1 will follow this as closely as possible while explicitly recording the
seed used for reproducibility.

## Split Decision

Fixed periods:

| Split component | Years |
| --- | --- |
| Train/validation pool | 1993-2000 |
| Test | 2001-2019 |

Train/validation rule:
- Apply horizon-specific NaN filtering first.
- Within the filtered 1993-2000 pool, use a deterministic 70/30 random split.
- Default train ratio: `0.70`.
- Default validation ratio: `0.30`.
- Default split seed: `42`.
- Split unit: sample row, identified by `(year, local_row)`.
- Use the same split seed for all horizons, but the final rows differ by horizon
  because each horizon has different missing target-return rows.

Test rule:
- All valid rows from 2001-2019 for the selected horizon go to test.
- Test rows are never used for training, validation, early stopping, or
  normalization-stat fitting.

Stratification:
- The paper summary says random 70/30 split but does not specify stratification.
- Stage 1 default is non-stratified sample-level random split to stay close to
  the paper summary.
- Class balance must still be reported after the split.

## Horizon Counts by Period

Saved inventory:
- `data_inventory/stage1_horizon_counts_by_year.csv`
- `data_inventory/stage1_horizon_counts_by_period.csv`

Period summary after horizon-specific NaN filtering:

| Period | Target | Valid rows | Positive rows | Non-positive rows | Positive rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Train/validation | `Ret_5d` | 790,791 | 394,894 | 395,897 | 49.94% |
| Train/validation | `Ret_20d` | 786,765 | 389,288 | 397,477 | 49.48% |
| Train/validation | `Ret_60d` | 774,998 | 399,580 | 375,418 | 51.56% |
| Test | `Ret_5d` | 1,399,933 | 710,950 | 688,983 | 50.78% |
| Test | `Ret_20d` | 1,393,845 | 727,814 | 666,031 | 52.22% |
| Test | `Ret_60d` | 1,376,215 | 744,274 | 631,941 | 54.08% |

Expected train/validation sizes before rounding:

| Experiment | Train/validation pool | Train approx. | Validation approx. |
| --- | ---: | ---: | ---: |
| `stage1_i20_r5` | 790,791 | 553,554 | 237,237 |
| `stage1_i20_r20` | 786,765 | 550,736 | 236,029 |
| `stage1_i20_r60` | 774,998 | 542,499 | 232,499 |

The exact integer split will be produced by the deterministic split
implementation and saved with split metadata.

## Normalization Decision

Image scaling before normalization:
- Read `.dat` as `uint8`.
- Convert image pixels to `float32`.
- Scale to `[0, 1]` by dividing by `255.0`.

Standardization:
- Fit one scalar pixel mean and one scalar pixel standard deviation on the
  training split only.
- Do not include validation or test images in mean/std fitting.
- Apply the same training mean/std to train, validation, and test images.

Formula:

```text
x_scaled = uint8_image / 255.0
x_norm = (x_scaled - train_pixel_mean) / train_pixel_std
```

Numerical guard:
- Use an epsilon guard if needed: `max(train_pixel_std, 1e-8)`.

Per-horizon normalization:
- Because each horizon has a different filtered row set and therefore a
  different train split, store normalization statistics per horizon.

Required saved normalization metadata:

```text
outputs/metrics/stage1_i20_r5/normalization.json
outputs/metrics/stage1_i20_r20/normalization.json
outputs/metrics/stage1_i20_r60/normalization.json
```

Each file should include:
- `target_return_name`
- `train_years`
- `validation_ratio`
- `split_seed`
- `pixel_scale`
- `train_pixel_mean`
- `train_pixel_std`
- `num_train_images_used`

## Split Metadata Output

The implementation should save split metadata per horizon:

```text
outputs/metrics/stage1_i20_r5/split_summary.json
outputs/metrics/stage1_i20_r20/split_summary.json
outputs/metrics/stage1_i20_r60/split_summary.json
```

Recommended split metadata columns if a split index CSV is saved:
- `year`
- `local_row`
- `Date`
- `StockID`
- `target_return_name`
- `target_return`
- `label`
- `split`

## Leakage Rules

Strict rules:
- Split assignment happens after horizon-specific target-return NaN filtering.
- Test years 2001-2019 are held out completely.
- Train pixel mean/std is fitted only on the training subset.
- Validation loss can be used for early stopping.
- Test labels are used only after model training for final evaluation.
- No future-return column is used as a model input.

## Kaggle and Local Smoke-test Behavior

Full run:
- Kaggle Notebook is the primary full-run environment.
- Full run uses all valid rows under the rules above.

Local smoke test:
- Local smoke tests may use a tiny subset for shape/code validation.
- Any smoke-test subset must be explicitly marked as smoke-test output.
- Smoke-test metrics must not be reported as Stage 1 reproduction results.

## Implementation Notes for Later

Planned config fields:

```yaml
split:
  train_val_years: [1993, 2000]
  test_years: [2001, 2019]
  validation_ratio: 0.30
  train_ratio: 0.70
  split_seed: 42
  split_unit: sample
  stratify: false

normalization:
  pixel_scale: 255.0
  fit_on: train
  scope: per_horizon
  epsilon: 1.0e-8
```

## Deferred Items

Deferred to Stage 1-6:
- Exact epoch cap.
- Early stopping implementation details.
- Whether full training runs multiple seeds after a first single-seed baseline.

Deferred to Stage 1-6K:
- Kaggle input/output path mapping.
- Where split and normalization JSON files are copied in Kaggle outputs.

## 한국어

상태:
- 1-4를 split/normalization 세부계획으로 완료했습니다.
- split 또는 normalization 코드는 아직 구현하지 않았습니다.

## 목적

모델 구현과 학습 전에 1단계 train/validation/test split과 image normalization 규칙을 정의합니다.

## 논문 근거

로컬 Re-image 요약 기준:
- train/validation period: 1993-2000.
- test period: 2001-2019.
- train/validation split: 1993-2000 내부 70/30 random split.
- image는 training data의 pixel mean과 standard deviation으로 standardization합니다.
- 정확한 random seed는 논문에 보고되지 않았습니다.

1단계는 이 규칙을 최대한 따르되, 재현성을 위해 우리가 사용하는 seed를 명시합니다.

## Split 결정

고정 기간:

| Split component | Years |
| --- | --- |
| Train/validation pool | 1993-2000 |
| Test | 2001-2019 |

Train/validation rule:
- horizon별 NaN filtering을 먼저 적용합니다.
- filtering된 1993-2000 pool 안에서 deterministic 70/30 random split을 사용합니다.
- 기본 train ratio: `0.70`.
- 기본 validation ratio: `0.30`.
- 기본 split seed: `42`.
- split unit: `(year, local_row)`로 식별되는 sample row.
- 모든 horizon에 같은 split seed를 쓰지만, horizon별 missing target row가 다르므로 최종 row는 horizon마다 다릅니다.

Test rule:
- 선택한 horizon에서 2001-2019의 모든 valid row는 test로 갑니다.
- test row는 training, validation, early stopping, normalization-stat fitting에 절대 사용하지 않습니다.

Stratification:
- 논문 요약은 70/30 random split이라고만 하고 stratification 여부는 말하지 않습니다.
- 논문 요약에 가깝게 가기 위해 1단계 기본값은 non-stratified sample-level random split입니다.
- 단, split 이후 class balance는 반드시 보고합니다.

## Horizon별 Period Count

저장한 inventory:
- `data_inventory/stage1_horizon_counts_by_year.csv`
- `data_inventory/stage1_horizon_counts_by_period.csv`

horizon별 NaN filtering 이후 period summary:

| Period | Target | Valid rows | Positive rows | Non-positive rows | Positive rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Train/validation | `Ret_5d` | 790,791 | 394,894 | 395,897 | 49.94% |
| Train/validation | `Ret_20d` | 786,765 | 389,288 | 397,477 | 49.48% |
| Train/validation | `Ret_60d` | 774,998 | 399,580 | 375,418 | 51.56% |
| Test | `Ret_5d` | 1,399,933 | 710,950 | 688,983 | 50.78% |
| Test | `Ret_20d` | 1,393,845 | 727,814 | 666,031 | 52.22% |
| Test | `Ret_60d` | 1,376,215 | 744,274 | 631,941 | 54.08% |

rounding 전 예상 train/validation 크기:

| Experiment | Train/validation pool | Train approx. | Validation approx. |
| --- | ---: | ---: | ---: |
| `stage1_i20_r5` | 790,791 | 553,554 | 237,237 |
| `stage1_i20_r20` | 786,765 | 550,736 | 236,029 |
| `stage1_i20_r60` | 774,998 | 542,499 | 232,499 |

정확한 integer split은 deterministic split 구현에서 만들고 split metadata로 저장합니다.

## Normalization 결정

Normalization 전 image scaling:
- `.dat`는 `uint8`로 읽습니다.
- image pixel을 `float32`로 변환합니다.
- `255.0`으로 나누어 `[0, 1]`로 scale합니다.

Standardization:
- training split만 사용해 scalar pixel mean 하나와 scalar pixel standard deviation 하나를 fit합니다.
- validation 또는 test image는 mean/std fitting에 포함하지 않습니다.
- 같은 training mean/std를 train, validation, test image에 적용합니다.

공식:

```text
x_scaled = uint8_image / 255.0
x_norm = (x_scaled - train_pixel_mean) / train_pixel_std
```

Numerical guard:
- 필요하면 `max(train_pixel_std, 1e-8)` guard를 사용합니다.

Horizon별 normalization:
- horizon마다 filtered row set과 train split이 다르므로 normalization statistics는 horizon별로 저장합니다.

필수 저장 normalization metadata:

```text
outputs/metrics/stage1_i20_r5/normalization.json
outputs/metrics/stage1_i20_r20/normalization.json
outputs/metrics/stage1_i20_r60/normalization.json
```

각 파일에는 다음을 포함합니다.
- `target_return_name`
- `train_years`
- `validation_ratio`
- `split_seed`
- `pixel_scale`
- `train_pixel_mean`
- `train_pixel_std`
- `num_train_images_used`

## Split Metadata Output

구현 시 horizon별 split metadata를 저장해야 합니다.

```text
outputs/metrics/stage1_i20_r5/split_summary.json
outputs/metrics/stage1_i20_r20/split_summary.json
outputs/metrics/stage1_i20_r60/split_summary.json
```

split index CSV를 저장한다면 권장 columns:
- `year`
- `local_row`
- `Date`
- `StockID`
- `target_return_name`
- `target_return`
- `label`
- `split`

## Leakage 방지 규칙

엄격한 규칙:
- split assignment는 horizon별 target-return NaN filtering 이후 수행합니다.
- 2001-2019 test year는 완전히 hold out합니다.
- train pixel mean/std는 training subset에서만 fit합니다.
- validation loss는 early stopping에 사용할 수 있습니다.
- test label은 model training 이후 final evaluation에만 사용합니다.
- 어떤 future-return column도 model input으로 사용하지 않습니다.

## Kaggle과 Local Smoke-test 동작

Full run:
- Kaggle Notebook이 기본 full-run 환경입니다.
- full run은 위 규칙 아래 모든 valid row를 사용합니다.

Local smoke test:
- 로컬 smoke test는 shape/code validation을 위한 작은 subset을 사용할 수 있습니다.
- smoke-test subset은 반드시 smoke-test output으로 표시합니다.
- smoke-test metric은 Stage 1 reproduction result로 보고하면 안 됩니다.

## 이후 구현 메모

예정 config fields:

```yaml
split:
  train_val_years: [1993, 2000]
  test_years: [2001, 2019]
  validation_ratio: 0.30
  train_ratio: 0.70
  split_seed: 42
  split_unit: sample
  stratify: false

normalization:
  pixel_scale: 255.0
  fit_on: train
  scope: per_horizon
  epsilon: 1.0e-8
```

## 이후 항목으로 넘길 내용

1-6으로 넘김:
- 정확한 epoch cap.
- early stopping 구현 세부.
- 첫 single-seed baseline 이후 multiple seed run을 할지 여부.

1-6K로 넘김:
- Kaggle input/output path mapping.
- Kaggle output에서 split/normalization JSON을 어디에 복사할지.
