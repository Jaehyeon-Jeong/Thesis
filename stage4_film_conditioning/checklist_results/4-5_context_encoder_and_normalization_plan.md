# 4-5 Context Encoder and Normalization Plan

## English

Status: complete for planning.

Purpose:
- define the exact numeric context vector for the first Stage 4 main run;
- lock the train-only normalization and imputation policy;
- define a small shared MLP context encoder used by all four Stage 4 ablations;
- keep context handling auditable and leakage-safe before model code is written.

This is an implementation plan, not a reported detail from the Re-image paper.
It follows the Stage 4 advisor-direction interpretation: structured market
context should enter as a separate condition vector, then be encoded by an MLP
and used for concat/gating/FiLM.

## Fixed Stage 4 Setting

The first Stage 4 main run remains:

```text
image/model baseline = Stage 2 I60/R20/ohlc_ma_vb
context source       = structured numeric market context
news context         = excluded from 4-A/B/C/D; deferred to 4-N
```

For each sample:

```text
t = image end date
image window = [t - 59, ..., t]
label window = (t, ..., t + 20]
```

Every context value must be available at or before `t`.

## Primary Raw Context Features

The first context vector uses the matched 60-day context policy from 4-2:

| Order | Raw feature | Source | Window | Meaning |
| ---: | --- | --- | --- | --- |
| 1 | `fg_value` | F&G external data | as-of `<= t` | current/as-of sentiment regime |
| 2 | `fg_mean_60` | F&G external data | last 60 days through `t` | full-window sentiment regime mean |
| 3 | `fg_delta_60` | F&G external data | `t` vs `t-60` | full-window sentiment change |
| 4 | `fg_std_60` | F&G external data | last 60 days through `t` | sentiment instability |
| 5 | `bb_percent_b_60` | BTC OHLCV | last 60 days through `t` | close position inside 60-day Bollinger band |
| 6 | `bb_bandwidth_60` | BTC OHLCV | last 60 days through `t` | 60-day Bollinger width / middle band |
| 7 | `mfi_60` | BTC OHLCV | last 60 days through `t` | volume-aware buying/selling pressure |
| 8 | `rv_60` | BTC OHLCV | last 60 days through `t` | realized volatility |

Support metadata fields should also be saved:
- `fg_source_date`
- `fg_age_days`
- `fg_missing`
- raw feature missing flags
- split, label, future return, image end date, and sample index

The primary model input is the normalized numeric vector. Missing/source-age
support fields can be added to the model input only if the 4-I2 audit finds
meaningful F&G gaps. Default first run: use the 8 primary features only.

## Feature-Specific Transform Plan

The context table should keep all stages:

```text
raw value -> transformed value -> imputed/clipped value -> normalized value
```

Primary transforms:

| Feature | Transform before train-only scaling | Reason |
| --- | --- | --- |
| `fg_value` | divide by `100` | known 0-100 index scale |
| `fg_mean_60` | divide by `100` | known 0-100 index scale |
| `fg_delta_60` | divide by `100` | converts point change to compact scale |
| `fg_std_60` | divide by `100` | compact sentiment-volatility scale |
| `bb_percent_b_60` | identity | can be below 0 or above 1; keep signal before clipping |
| `bb_bandwidth_60` | `log1p(x)` | positive, skewed range/volatility feature |
| `mfi_60` | divide by `100` | known 0-100 oscillator scale |
| `rv_60` | `log1p(x)` | positive, skewed volatility feature |

All transformed features are then imputed, clipped, and standardized using
train-split statistics only.

## Train-Only Imputation and Clipping

Fit only on `split == train`.

Per transformed feature:
1. Compute train median.
2. Replace missing/non-finite values with the train median.
3. Compute train `q01` and `q99`.
4. Clip all splits to train `q01`/`q99`.
5. Compute train mean and standard deviation after imputation and clipping.
6. Normalize all splits:

```text
z = (x_clipped - train_mean) / max(train_std, epsilon)
```

Default:

```text
epsilon = 1e-8
clip_quantiles = [0.01, 0.99]
```

Why this policy:
- Train-only fit prevents validation/test leakage.
- Median imputation is robust for small train size.
- Quantile clipping avoids a single extreme volatility/bandwidth point
  dominating a small BTC training set.
- Z-score normalization gives the MLP stable input scale.

Failure/warning thresholds for 4-I2:
- If any primary feature has `> 30%` missing values in train, fail the run and
  audit the source.
- If any primary feature has `> 5%` missing values in any split, write a warning
  into the run manifest.
- If F&G coverage is poor, run an OHLCV-only context variant before adding F&G.

## Saved Artifacts

The Stage 4 runner must write:

```text
outputs/stage4/context/<experiment>/seed_<seed>/context_features.csv
outputs/stage4/context/<experiment>/seed_<seed>/context_scaler.json
outputs/stage4/context/<experiment>/seed_<seed>/context_feature_audit.json
outputs/stage4/context/<experiment>/seed_<seed>/context_encoder_config.json
```

`context_features.csv` should include:
- sample identifiers and split;
- raw values;
- transformed values;
- imputed/clipped values;
- normalized values;
- missing flags;
- F&G source date and age;
- label and future return for audit only.

`context_scaler.json` should include:
- feature order;
- transform type per feature;
- train median;
- train `q01`/`q99`;
- train mean/std;
- missing rates by split;
- source file paths and versions;
- created timestamp;
- Git commit.

## Context Encoder

Use one shared context encoder across 4-A/B/C/D so the fusion method is the
main thing being compared.

Default encoder:

```text
input_dim = 8
hidden_dim = 32
embedding_dim = 32
dropout = 0.10

normalized context vector
  -> Linear(input_dim, 32)
  -> ReLU
  -> Dropout(0.10)
  -> Linear(32, 32)
  -> ReLU
  -> context embedding
```

Do not use BatchNorm in the context encoder for the first run:
- the dataset is small;
- context features are already standardized;
- BatchNorm would make the context branch depend on batch composition.

The encoder output is:

```text
context_embedding: (batch_size, 32)
```

Each model family uses this same embedding:

| Track | Uses context embedding how |
| --- | --- |
| 4-A concat | append to final CNN feature before classifier |
| 4-B gating | generate feature/channel gates |
| 4-C gamma-only FiLM | generate block-wise gamma |
| 4-D full FiLM | generate block-wise gamma and beta |

Output heads for gates/gamma/beta are part of 4-6, not this step. 4-6 should
initialize modulation heads close to identity so the first update does not
destroy the Stage 2 baseline behavior.

## Overfitting Controls

The train split is small for `I60/R20`:

```text
train rows = 671
validation rows = 287
test rows = 1,441
```

So the first context branch must stay small:
- no raw 60-step context sequence;
- no deep context network;
- no large embedding dimension;
- no news/LLM vectors in 4-A/B/C/D;
- shared encoder across ablations;
- save all context values for audit.

Recommended first hyperparameters:

| Parameter | Value |
| --- | ---: |
| `context_hidden_dim` | `32` |
| `context_embedding_dim` | `32` |
| `context_dropout` | `0.10` |
| `context_clip_low_quantile` | `0.01` |
| `context_clip_high_quantile` | `0.99` |
| `context_epsilon` | `1e-8` |

## 4-5 Decision

Proceed to 4-6 with:

```text
primary_context_features = [
    fg_value,
    fg_mean_60,
    fg_delta_60,
    fg_std_60,
    bb_percent_b_60,
    bb_bandwidth_60,
    mfi_60,
    rv_60,
]

context_preprocessing =
    feature-specific transform
    -> train-only median imputation
    -> train-only 1/99% clipping
    -> train-only z-score normalization

context_encoder =
    Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU
```

## 한국어

상태: 계획 단계 완료.

목적:
- 첫 Stage 4 main run에서 사용할 numeric context vector를 정확히 정의합니다.
- train-only normalization과 imputation 정책을 고정합니다.
- 네 가지 Stage 4 ablation이 공통으로 사용할 작은 MLP context encoder를 정의합니다.
- 실제 모델 코드를 작성하기 전에 context 처리를 audit 가능하고 leakage-safe하게 만듭니다.

이 내용은 Re-image 논문에 보고된 디테일이 아니라 구현상 계획입니다. 교수님 방향성 해석에
따라 structured market context를 별도 condition vector로 넣고, MLP로 encoding한 뒤
concat/gating/FiLM에 사용합니다.

## 고정 Stage 4 설정

첫 Stage 4 main run은 그대로 다음입니다.

```text
image/model baseline = Stage 2 I60/R20/ohlc_ma_vb
context source       = structured numeric market context
news context         = 4-A/B/C/D에서는 제외, 4-N으로 연기
```

각 sample에서:

```text
t = image end date
image window = [t - 59, ..., t]
label window = (t, ..., t + 20]
```

모든 context 값은 반드시 `t` 또는 그 이전에 알 수 있어야 합니다.

## Primary raw context features

첫 context vector는 4-2에서 정한 matched 60-day context policy를 사용합니다.

| 순서 | Raw feature | Source | Window | 의미 |
| ---: | --- | --- | --- | --- |
| 1 | `fg_value` | F&G external data | as-of `<= t` | 현재/as-of sentiment regime |
| 2 | `fg_mean_60` | F&G external data | `t`까지 최근 60일 | 전체 window sentiment regime mean |
| 3 | `fg_delta_60` | F&G external data | `t` vs `t-60` | 전체 window sentiment change |
| 4 | `fg_std_60` | F&G external data | `t`까지 최근 60일 | sentiment instability |
| 5 | `bb_percent_b_60` | BTC OHLCV | `t`까지 최근 60일 | 60일 Bollinger band 안에서의 종가 위치 |
| 6 | `bb_bandwidth_60` | BTC OHLCV | `t`까지 최근 60일 | 60일 Bollinger width / middle band |
| 7 | `mfi_60` | BTC OHLCV | `t`까지 최근 60일 | volume-aware buying/selling pressure |
| 8 | `rv_60` | BTC OHLCV | `t`까지 최근 60일 | realized volatility |

함께 저장해야 하는 support metadata:
- `fg_source_date`
- `fg_age_days`
- `fg_missing`
- raw feature missing flags
- split, label, future return, image end date, sample index

Primary model input은 normalized numeric vector입니다. Missing/source-age support
field는 4-I2 audit에서 의미 있는 F&G gap이 발견될 때만 model input에 추가합니다.
기본 첫 run에서는 8개 primary feature만 사용합니다.

## Feature별 transform 계획

Context table은 모든 단계를 보존해야 합니다.

```text
raw value -> transformed value -> imputed/clipped value -> normalized value
```

Primary transform:

| Feature | Train-only scaling 전 transform | 이유 |
| --- | --- | --- |
| `fg_value` | `100`으로 나눔 | 0-100 index scale |
| `fg_mean_60` | `100`으로 나눔 | 0-100 index scale |
| `fg_delta_60` | `100`으로 나눔 | point change를 compact scale로 변환 |
| `fg_std_60` | `100`으로 나눔 | sentiment-volatility를 compact scale로 변환 |
| `bb_percent_b_60` | identity | 0 아래/1 위 값도 signal이므로 clipping 전에는 유지 |
| `bb_bandwidth_60` | `log1p(x)` | 양수이고 skew가 큰 range/volatility feature |
| `mfi_60` | `100`으로 나눔 | 0-100 oscillator scale |
| `rv_60` | `log1p(x)` | 양수이고 skew가 큰 volatility feature |

모든 transformed feature는 train split 통계만 사용해서 impute, clip, standardize합니다.

## Train-only imputation과 clipping

`split == train`에서만 fit합니다.

각 transformed feature별 처리:
1. train median 계산.
2. missing/non-finite 값을 train median으로 대체.
3. train `q01`, `q99` 계산.
4. 모든 split을 train `q01`/`q99`로 clip.
5. imputation과 clipping 후 train mean/std 계산.
6. 모든 split normalize:

```text
z = (x_clipped - train_mean) / max(train_std, epsilon)
```

기본값:

```text
epsilon = 1e-8
clip_quantiles = [0.01, 0.99]
```

이 정책을 쓰는 이유:
- Train-only fit으로 validation/test leakage를 막습니다.
- Median imputation은 작은 train size에서 robust합니다.
- Quantile clipping은 BTC train set이 작기 때문에 극단적 volatility/bandwidth 값 하나가
  context branch를 지배하는 것을 막습니다.
- Z-score normalization은 MLP 입력 scale을 안정화합니다.

4-I2에서 사용할 failure/warning threshold:
- 어떤 primary feature라도 train missing이 `> 30%`이면 run을 실패시키고 source를
  audit합니다.
- 어떤 primary feature라도 어느 split에서든 missing이 `> 5%`이면 run manifest에
  warning을 씁니다.
- F&G coverage가 좋지 않으면 F&G를 넣기 전에 OHLCV-only context variant를 먼저 실행합니다.

## 저장 artifact

Stage 4 runner는 다음을 저장해야 합니다.

```text
outputs/stage4/context/<experiment>/seed_<seed>/context_features.csv
outputs/stage4/context/<experiment>/seed_<seed>/context_scaler.json
outputs/stage4/context/<experiment>/seed_<seed>/context_feature_audit.json
outputs/stage4/context/<experiment>/seed_<seed>/context_encoder_config.json
```

`context_features.csv`에는 다음이 들어갑니다.
- sample identifiers and split;
- raw values;
- transformed values;
- imputed/clipped values;
- normalized values;
- missing flags;
- F&G source date and age;
- audit용 label and future return.

`context_scaler.json`에는 다음을 저장합니다.
- feature order;
- feature별 transform type;
- train median;
- train `q01`/`q99`;
- train mean/std;
- split별 missing rate;
- source file path/version;
- created timestamp;
- Git commit.

## Context encoder

4-A/B/C/D에서 하나의 공통 context encoder를 사용합니다. 그래야 비교 대상이 fusion
method 자체로 유지됩니다.

기본 encoder:

```text
input_dim = 8
hidden_dim = 32
embedding_dim = 32
dropout = 0.10

normalized context vector
  -> Linear(input_dim, 32)
  -> ReLU
  -> Dropout(0.10)
  -> Linear(32, 32)
  -> ReLU
  -> context embedding
```

첫 run에서는 context encoder에 BatchNorm을 쓰지 않습니다.
- dataset이 작습니다.
- context feature는 이미 standardize되어 있습니다.
- BatchNorm은 context branch를 batch composition에 의존하게 만듭니다.

Encoder output:

```text
context_embedding: (batch_size, 32)
```

각 model family는 이 embedding을 동일하게 사용합니다.

| Track | Context embedding 사용 방식 |
| --- | --- |
| 4-A concat | final CNN feature 앞/뒤로 classifier 직전에 concat |
| 4-B gating | feature/channel gate 생성 |
| 4-C gamma-only FiLM | block-wise gamma 생성 |
| 4-D full FiLM | block-wise gamma와 beta 생성 |

Gate/gamma/beta output head는 4-6에서 설계합니다. 4-6에서는 modulation head를
identity에 가깝게 초기화해서 첫 update가 Stage 2 baseline behavior를 망치지 않게 해야 합니다.

## Overfitting control

`I60/R20` train split은 작습니다.

```text
train rows = 671
validation rows = 287
test rows = 1,441
```

따라서 첫 context branch는 작게 유지합니다.
- raw 60-step context sequence 사용 안 함;
- deep context network 사용 안 함;
- 큰 embedding dimension 사용 안 함;
- 4-A/B/C/D에 news/LLM vector 포함 안 함;
- ablation 간 encoder 공유;
- 모든 context 값을 저장해서 audit 가능하게 함.

추천 첫 hyperparameter:

| Parameter | Value |
| --- | ---: |
| `context_hidden_dim` | `32` |
| `context_embedding_dim` | `32` |
| `context_dropout` | `0.10` |
| `context_clip_low_quantile` | `0.01` |
| `context_clip_high_quantile` | `0.99` |
| `context_epsilon` | `1e-8` |

## 4-5 결정

다음을 고정하고 4-6으로 넘어갑니다.

```text
primary_context_features = [
    fg_value,
    fg_mean_60,
    fg_delta_60,
    fg_std_60,
    bb_percent_b_60,
    bb_bandwidth_60,
    mfi_60,
    rv_60,
]

context_preprocessing =
    feature-specific transform
    -> train-only median imputation
    -> train-only 1/99% clipping
    -> train-only z-score normalization

context_encoder =
    Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU
```
