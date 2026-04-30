# Stage 1 Label Construction Detail Plan

## English

Status:
- Stage 1-3 completed as a detail plan.
- No label-construction code has been implemented yet.

## Purpose

Define how the author-provided future return columns become binary labels for
Stage 1 Re-image reproduction.

This plan starts from the row-aligned image/label records defined in
`docs/data_loading_plan.md`.

## Fixed Stage 1 Targets

The current public data contains 20-day full-spec images only. Therefore Stage 1
uses the same I20 image with three future-return horizons:

| Experiment | Image | Target return column | Meaning |
| --- | --- | --- | --- |
| `I20/R5` | 20-day full-spec image | `Ret_5d` | Future 5-trading-day holding-period return |
| `I20/R20` | 20-day full-spec image | `Ret_20d` | Future 20-trading-day holding-period return |
| `I20/R60` | 20-day full-spec image | `Ret_60d` | Future 60-trading-day holding-period return |

`Ret_month` exists but is not one of the fixed R=5/20/60 targets. It may be
preserved as metadata or used later as an optional experiment only after the
three fixed horizons are handled.

## Binary Label Rule

For a selected target return column `r`:

```text
label = 1 if r > 0
label = 0 otherwise
```

Interpretation:
- `1`: Up / positive future return.
- `0`: Down or non-positive future return.

Zero-return handling:
- Exact zero belongs to class `0`.
- Reason: the Re-image binary classification setup uses positive future return
  as the Up class; non-positive returns are not Up.

Missing-value handling:
- Rows with missing selected target return are removed for that horizon.
- Filtering is horizon-specific because `Ret_5d`, `Ret_20d`, and `Ret_60d`
  have different missing counts.

## Horizon-specific Valid Counts

Confirmed from Stage 0 inventory:

| Target | Non-null rows | Null rows | Positive rows | Non-positive rows | Positive rate among valid rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Ret_5d` | 2,190,724 | 6,270 | 1,105,844 | 1,084,880 | 50.48% |
| `Ret_20d` | 2,180,610 | 16,384 | 1,117,102 | 1,063,508 | 51.23% |
| `Ret_60d` | 2,151,213 | 45,781 | 1,143,854 | 1,007,359 | 53.17% |

Implication:
- Each horizon creates a different filtered dataset.
- Metrics and predictions must report the target horizon and number of valid rows.

## Required Metadata to Preserve

Each sample should preserve:
- `Date`
- `StockID`
- `MarketCap`
- `EWMA_vol`
- `Ret_5d`
- `Ret_20d`
- `Ret_60d`
- `Ret_month`
- `year`
- `local_row`
- `target_return_name`
- `target_return`
- `label`

Model input:
- Only the image tensor is used for the Stage 1 CNN baseline.
- Future returns and metadata must not be used as model inputs.

## Dataset Naming

Use explicit horizon names:

| Name | Target |
| --- | --- |
| `stage1_i20_r5` | `Ret_5d` |
| `stage1_i20_r20` | `Ret_20d` |
| `stage1_i20_r60` | `Ret_60d` |

Output directories should include the horizon name:

```text
outputs/predictions/stage1_i20_r5/
outputs/predictions/stage1_i20_r20/
outputs/predictions/stage1_i20_r60/

outputs/metrics/stage1_i20_r5/
outputs/metrics/stage1_i20_r20/
outputs/metrics/stage1_i20_r60/
```

## Prediction Output Schema

The final prediction CSV for each horizon must include at least:

| Column | Meaning |
| --- | --- |
| `Date` | Image end date, the last date in the 20-day chart window |
| `StockID` | CRSP PERMNO |
| `year` | File shard year |
| `local_row` | Row index inside the year shard |
| `target_return_name` | `Ret_5d`, `Ret_20d`, or `Ret_60d` |
| `target_return` | Original continuous future return |
| `label` | Binary label, `1` for positive return and `0` otherwise |
| `pred_class` | Predicted class from the model |
| `prob_up` | Softmax probability for class `1` |
| `logit_down` | Optional but recommended |
| `logit_up` | Optional but recommended |

Additional metadata that should be preserved when feasible:
- `MarketCap`
- `EWMA_vol`
- non-target return columns

## Leakage Rules

Strict rules:
- The model input is only the image ending at `Date = t`.
- The selected `Ret_*` column is used only as label/evaluation target.
- No future-return column may be used as an input feature.
- Label filtering must not use information from test labels to alter train
  preprocessing, except for dropping missing target rows within each horizon.
- Split assignment is handled in Stage 1-4 after labels are defined.

## Implementation Plan for Later

The eventual implementation should expose:

```python
TARGET_COLUMNS = {
    "i20_r5": "Ret_5d",
    "i20_r20": "Ret_20d",
    "i20_r60": "Ret_60d",
}
```

Planned per-horizon procedure:

```text
1. Start from row-aligned base metadata.
2. Select target return column.
3. Drop rows where target return is missing.
4. Create target_return = selected return value.
5. Create label = int(target_return > 0).
6. Preserve metadata and local row references.
7. Pass the filtered index list to the dataset/split logic.
```

## Deferred Items

Deferred to Stage 1-4:
- Train/validation/test split after horizon-specific filtering.
- Whether train/validation split should be random by sample inside 1993-2000 or
  controlled in another way to match the paper as closely as possible.

Deferred to Stage 1-7:
- Exact final prediction CSV filename.
- Whether to save all logits in every output.
- Portfolio/decile construction after classification predictions.

## 한국어

상태:
- 1-3을 label construction 세부계획으로 완료했습니다.
- label-construction 코드는 아직 구현하지 않았습니다.

## 목적

저자 제공 future return column을 1단계 Re-image 재현용 binary label로 어떻게 바꿀지 정의합니다.

이 계획은 `docs/data_loading_plan.md`에서 정의한 row-aligned image/label record에서 시작합니다.

## 고정된 1단계 Target

현재 public data에는 20-day full-spec image만 있습니다. 따라서 1단계는 같은 I20 image에
세 가지 future-return horizon을 붙입니다.

| Experiment | Image | Target return column | 의미 |
| --- | --- | --- | --- |
| `I20/R5` | 20-day full-spec image | `Ret_5d` | 미래 5거래일 보유수익률 |
| `I20/R20` | 20-day full-spec image | `Ret_20d` | 미래 20거래일 보유수익률 |
| `I20/R60` | 20-day full-spec image | `Ret_60d` | 미래 60거래일 보유수익률 |

`Ret_month`도 존재하지만 고정 R=5/20/60 target은 아닙니다. 세 고정 horizon을 처리한 뒤
optional experiment로만 다룹니다.

## Binary Label Rule

선택한 target return column을 `r`이라고 하면:

```text
label = 1 if r > 0
label = 0 otherwise
```

해석:
- `1`: Up / positive future return.
- `0`: Down 또는 non-positive future return.

Zero-return 처리:
- 정확히 0인 return은 class `0`입니다.
- 이유: Re-image binary classification은 positive future return을 Up class로 둡니다.
  0 또는 음수 return은 Up이 아닙니다.

Missing-value 처리:
- 선택한 target return이 missing인 row는 해당 horizon dataset에서 제거합니다.
- `Ret_5d`, `Ret_20d`, `Ret_60d`의 missing count가 다르므로 filtering은 horizon별로 따로 합니다.

## Horizon별 유효 Row 수

0단계 inventory 기준:

| Target | Non-null rows | Null rows | Positive rows | Non-positive rows | Valid row 기준 positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Ret_5d` | 2,190,724 | 6,270 | 1,105,844 | 1,084,880 | 50.48% |
| `Ret_20d` | 2,180,610 | 16,384 | 1,117,102 | 1,063,508 | 51.23% |
| `Ret_60d` | 2,151,213 | 45,781 | 1,143,854 | 1,007,359 | 53.17% |

의미:
- horizon마다 서로 다른 filtered dataset이 만들어집니다.
- metric과 prediction은 target horizon과 valid row 수를 반드시 함께 보고해야 합니다.

## 보존할 Metadata

각 sample은 다음을 보존해야 합니다.
- `Date`
- `StockID`
- `MarketCap`
- `EWMA_vol`
- `Ret_5d`
- `Ret_20d`
- `Ret_60d`
- `Ret_month`
- `year`
- `local_row`
- `target_return_name`
- `target_return`
- `label`

Model input:
- 1단계 CNN baseline에서 model input은 image tensor뿐입니다.
- future return과 metadata는 model input으로 쓰면 안 됩니다.

## Dataset Naming

horizon 이름을 명시적으로 씁니다.

| Name | Target |
| --- | --- |
| `stage1_i20_r5` | `Ret_5d` |
| `stage1_i20_r20` | `Ret_20d` |
| `stage1_i20_r60` | `Ret_60d` |

Output directory에도 horizon name을 포함합니다.

```text
outputs/predictions/stage1_i20_r5/
outputs/predictions/stage1_i20_r20/
outputs/predictions/stage1_i20_r60/

outputs/metrics/stage1_i20_r5/
outputs/metrics/stage1_i20_r20/
outputs/metrics/stage1_i20_r60/
```

## Prediction Output Schema

horizon별 최종 prediction CSV에는 최소한 다음 column을 포함합니다.

| Column | 의미 |
| --- | --- |
| `Date` | image end date, 즉 20-day chart window의 마지막 날짜 |
| `StockID` | CRSP PERMNO |
| `year` | file shard year |
| `local_row` | year shard 내부 row index |
| `target_return_name` | `Ret_5d`, `Ret_20d`, 또는 `Ret_60d` |
| `target_return` | 원래 continuous future return |
| `label` | binary label, positive return이면 `1`, 아니면 `0` |
| `pred_class` | model predicted class |
| `prob_up` | class `1`에 대한 softmax probability |
| `logit_down` | optional이지만 저장 권장 |
| `logit_up` | optional이지만 저장 권장 |

가능하면 추가로 보존할 metadata:
- `MarketCap`
- `EWMA_vol`
- non-target return columns

## Leakage 방지 규칙

엄격한 규칙:
- model input은 `Date = t`에서 끝나는 image뿐입니다.
- 선택한 `Ret_*` column은 label/evaluation target으로만 사용합니다.
- 어떤 future-return column도 input feature로 사용하면 안 됩니다.
- missing target row를 horizon별로 제거하는 것 외에, test label 정보를 train preprocessing 변경에 사용하면 안 됩니다.
- split assignment는 label 정의 이후 1-4에서 처리합니다.

## 이후 구현 계획

최종 구현에서는 다음 mapping을 둡니다.

```python
TARGET_COLUMNS = {
    "i20_r5": "Ret_5d",
    "i20_r20": "Ret_20d",
    "i20_r60": "Ret_60d",
}
```

horizon별 절차:

```text
1. row-aligned base metadata에서 시작합니다.
2. target return column을 선택합니다.
3. target return이 missing인 row를 제거합니다.
4. target_return = selected return value를 만듭니다.
5. label = int(target_return > 0)를 만듭니다.
6. metadata와 local row reference를 보존합니다.
7. filtered index list를 dataset/split logic에 넘깁니다.
```

## 이후 항목으로 넘길 내용

1-4로 넘김:
- horizon-specific filtering 이후 train/validation/test split.
- 논문에 최대한 맞추기 위해 1993-2000 train/validation split을 sample random으로 할지 다른 방식으로 제어할지.

1-7로 넘김:
- 최종 prediction CSV filename.
- 모든 logit을 항상 저장할지.
- classification prediction 이후 portfolio/decile construction.
