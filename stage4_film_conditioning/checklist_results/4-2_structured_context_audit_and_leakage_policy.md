# 4-2 Structured Numeric Context Audit and Leakage Policy

## English

Status: revised planning decision. Implementation audit remains in `4-I2`.

Stage 4 uses the selected Stage 2 baseline:

```text
image_window = I60
return_horizon = R20
image_spec = ohlc_ma_vb
```

For every sample:

```text
t = image end date
image window = [t - 59, ..., t]
label window = (t, ..., t + 20]
```

All context features must be available at or before `t`. If a source's
publication time is ambiguous, use the most recent value strictly before `t`.

## Main Decision

The first Stage 4 context policy is now:

```text
context_window = image_window
```

Reason:
- The selected image is an `I60` chart.
- The condition vector should summarize the same 60-day market context that
  the CNN sees visually.
- FiLM then learns how the 60-day context modulates 60-day visual features.

This is more defensible than using `Bollinger 20` or `MFI 14` as the main
setting only because those are common technical-analysis defaults.

For the selected Stage 4 baseline:

```text
I60/R20/ohlc_ma_vb -> 60-day context features
```

If a later experiment uses `I20`, the matched context uses 20-day context
features. If `I5` is used, 5-day technical features are possible for parity,
but they should be marked as high-noise and secondary because Bollinger/MFI are
unstable on very short windows.

## Professor-Direction Interpretation

The advisor direction is not to draw Bollinger bands, F&G, or MFI into the
chart image as extra pixels for the main Stage 4 experiment.

The intended design is:

```text
chart image -> CNN visual feature
market context vector -> MLP condition embedding
condition embedding -> concat / gate / FiLM gamma-beta
```

So Bollinger, MFI, F&G, and volatility enter as **conditional embedding input**,
not as new lines or indicators rendered inside the image.

## Primary Context Vector: Matched Window

Use compact trailing summary features, not a raw 60-step sequence:

| Feature | Source | Window | Definition | Leakage rule |
|:---|:---|:---|:---|:---|
| `fg_value` | external F&G dataset | latest `<= t` | Fear & Greed value at decision date | as-of merge only, no future backfill |
| `fg_mean_60` | external F&G dataset | last 60 days through `t` | full-window F&G regime mean | dates `<= t` only |
| `fg_delta_60` | external F&G dataset | `t` vs `t-60` | full-window F&G regime change | prior available values only |
| `fg_std_60` | external F&G dataset | last 60 days through `t` | F&G instability/regime volatility | dates `<= t` only |
| `bb_percent_b_60` | BTC OHLCV | 60 days through `t` | position of current close within 60-day Bollinger band | close prices `<= t` only |
| `bb_bandwidth_60` | BTC OHLCV | 60 days through `t` | 60-day Bollinger width / middle band | close prices `<= t` only |
| `mfi_60` | BTC OHLCV | 60 days through `t` | full-window Money Flow Index | OHLCV `<= t` only |
| `rv_60` | BTC OHLCV | 60 days through `t` | full-window realized volatility | close prices `<= t` only |

Why compact summaries instead of the full 60 values:
- The first goal is to test context conditioning, not sequence modeling.
- A compact vector is easier to audit for leakage.
- A compact vector is easier to interpret against FiLM gamma/beta.
- A 60-step context sequence would require a separate encoder and would
  confound the first FiLM experiment.

## 1. Fear & Greed

Fear & Greed is not derived from BTC OHLCV. It requires an external sentiment
or regime dataset. The current candidate is:

```text
https://www.kaggle.com/datasets/ashishpatel8736/historical-and-fear-greed-index-datasets
```

Primary `I60` features:
- `fg_value`: current/as-of value at `t`;
- `fg_mean_60`: 60-day sentiment regime average;
- `fg_delta_60`: full-window sentiment change;
- `fg_std_60`: instability of sentiment over the image window.

Do not use `fg_delta_20` as the main setting for `I60`. It can be kept for a
later multi-scale ablation, but the main argument is window-matched context.

The `4-I2` audit must check:
- date coverage against BTC Stage 2 dates;
- column names and value scale;
- whether the index is crypto-specific or broad/equity F&G;
- missing dates;
- timezone/publication timestamp assumptions.

## 2. Bollinger Features

Bollinger features are derived from BTC close prices. No additional dataset is
needed.

Primary `I60` implementation:

```text
sma60_t = mean(Close[t-59:t])
std60_t = std(Close[t-59:t])
upper60_t = sma60_t + 2 * std60_t
lower60_t = sma60_t - 2 * std60_t

bb_percent_b_60 = (Close_t - lower60_t) / (upper60_t - lower60_t)
bb_bandwidth_60 = (upper60_t - lower60_t) / sma60_t
```

Interpretation:
- `%B` tells whether the current close is near the lower band, middle, upper
  band, or outside the band.
- Bandwidth tells whether the 60-day window is compressed or volatile.

Why `BB60` is primary:
- It summarizes the same 60-day evidence window as the chart image.
- It avoids the weak argument that `BB20` is used only because it is a common
  default.
- It gives a cleaner FiLM interpretation: gamma/beta are conditioned on the
  full-window chart regime.

Keep `BB20` only as a later `standard_window` diagnostic.

## 3. Money Flow Index

MFI is derived from BTC OHLCV. No additional dataset is needed.

Primary `I60` implementation:

```text
typical_price_t = (High_t + Low_t + Close_t) / 3
raw_money_flow_t = typical_price_t * Volume_t

positive_flow_t = raw_money_flow_t if typical_price_t > typical_price_{t-1} else 0
negative_flow_t = raw_money_flow_t if typical_price_t < typical_price_{t-1} else 0

money_flow_ratio_60 =
    sum(positive_flow[t-59:t]) / sum(negative_flow[t-59:t])

mfi_60 = 100 - 100 / (1 + money_flow_ratio_60)
```

Why `MFI60` is primary:
- It summarizes volume-adjusted buying/selling pressure over the same 60 days
  shown to the CNN.
- It aligns with the Stage 4 conditioning thesis better than using `MFI14` by
  convention.

Keep `MFI14` only as a later `standard_window` diagnostic.

Implementation notes:
- Use an epsilon when negative flow is zero.
- Inspect the raw distribution before clipping.
- Normalize final context features using train-only statistics.

## 4. Which Features Need External Data?

| Feature family | External dataset? | Reason |
|:---|:---:|:---|
| Fear & Greed | yes | external sentiment/regime signal |
| Bollinger %B / bandwidth | no | derived from BTC close prices |
| MFI | no | derived from BTC OHLCV |
| Realized volatility | no | derived from BTC close/log returns |

So the first Stage 4 implementation needs one extra external dataset only:
Fear & Greed. The remaining features come from the Stage 2 BTC OHLCV file.

## Context-Window Ablation Plan

Primary run:
1. `matched_window`: `BB60`, `MFI60`, `F&G60`, `RV60`.

Secondary diagnostics, only after the main four model families are checked:
2. `standard_window`: `BB20`, `MFI14`, short F&G summaries, `RV20`.
3. `multi_scale`: matched-window plus short-window summaries, for example
   `BB20 + BB60`, `MFI14 + MFI60`, `F&G20 + F&G60`, `RV20 + RV60`.

Recommended run order:
1. Run the four Stage 4 model families with `matched_window`.
2. If one model family is clearly strongest, run `standard_window` and
   `multi_scale` on that model family first.
3. Only expand the full grid if the window policy itself becomes a central
   result.

## Leakage and Normalization Rules

1. Use only information available at or before image end date `t`.
2. External daily context uses as-of merge: latest context date `<= t`.
3. No backward fill from future dates.
4. Rolling features are computed with trailing windows only.
5. Context imputation rules are fitted on train only.
6. Context normalization statistics are fitted on train only.
7. Save the context table used for every prediction with date, raw values,
   normalized values, missing flags, and merge source date.

## 한국어

상태: 수정된 계획 결정 완료. 실제 구현 audit은 `4-I2`에서 진행합니다.

Stage 4는 Stage 2 selected baseline을 기준으로 합니다.

```text
image_window = I60
return_horizon = R20
image_spec = ohlc_ma_vb
```

각 sample에서:

```text
t = image end date
image window = [t - 59, ..., t]
label window = (t, ..., t + 20]
```

모든 context feature는 반드시 `t` 또는 그 이전에 알 수 있어야 합니다. 외부 source의
발표 시간이 애매하면 보수적으로 `t`보다 이전의 가장 최근 값을 씁니다.

## 핵심 수정

첫 Stage 4 context policy는 다음으로 수정합니다.

```text
context_window = image_window
```

이유:
- selected image는 `I60` chart입니다.
- condition vector도 CNN이 보는 동일한 60일 시장 맥락을 요약해야 합니다.
- FiLM은 그 60일 context가 60일 visual feature를 어떻게 조절하는지 학습합니다.

따라서 `BB20`이나 `MFI14`를 단지 기술적 분석에서 흔하다는 이유만으로 main setting에
두는 것보다, `BB60/MFI60/F&G60`이 논리적으로 더 강합니다.

Stage 4 selected baseline에서는:

```text
I60/R20/ohlc_ma_vb -> 60-day context features
```

나중에 `I20`을 쓰면 20일 context를 씁니다. `I5`는 parity를 위해 5일 지표를 만들 수
있지만, Bollinger/MFI가 매우 noisy하므로 secondary로 표시해야 합니다.

## 교수님 방향성과의 연결

교수님 의도는 Bollinger band, F&G, MFI를 chart image 위에 새 선이나 픽셀로 그리라는
뜻이 아닙니다.

의도한 구조는 다음입니다.

```text
chart image -> CNN visual feature
market context vector -> MLP condition embedding
condition embedding -> concat / gate / FiLM gamma-beta
```

즉 Bollinger, MFI, F&G, volatility는 **conditional embedding input**으로 들어갑니다.
main Stage 4에서는 이미지 자체를 더 복잡하게 만들지 않습니다.

## Primary context vector: matched window

60일 raw sequence 전체가 아니라 compact trailing summary를 먼저 씁니다.

| Feature | Source | Window | 정의 | Leakage 규칙 |
|:---|:---|:---|:---|:---|
| `fg_value` | 외부 F&G dataset | latest `<= t` | decision date의 Fear & Greed 값 | as-of merge만 사용, future backfill 금지 |
| `fg_mean_60` | 외부 F&G dataset | `t`까지 60일 | full-window F&G regime 평균 | `<= t` 날짜만 사용 |
| `fg_delta_60` | 외부 F&G dataset | `t` vs `t-60` | full-window F&G 변화 | 이전 이용 가능 값만 사용 |
| `fg_std_60` | 외부 F&G dataset | `t`까지 60일 | F&G instability/regime volatility | `<= t` 날짜만 사용 |
| `bb_percent_b_60` | BTC OHLCV | `t`까지 60일 | 60일 Bollinger band 내 현재 close 위치 | `<= t` close만 사용 |
| `bb_bandwidth_60` | BTC OHLCV | `t`까지 60일 | 60일 Bollinger width / middle band | `<= t` close만 사용 |
| `mfi_60` | BTC OHLCV | `t`까지 60일 | 60일 Money Flow Index | `<= t` OHLCV만 사용 |
| `rv_60` | BTC OHLCV | `t`까지 60일 | 60일 realized volatility | `<= t` close만 사용 |

왜 60일 raw 값을 그대로 넣지 않는가:
- 첫 목표는 context conditioning 검증이지 sequence modeling이 아닙니다.
- compact vector가 leakage audit이 쉽습니다.
- compact vector가 FiLM gamma/beta와 연결해서 해석하기 쉽습니다.
- 60-step context sequence는 별도 encoder가 필요해서 첫 FiLM 실험을 복잡하게 만듭니다.

## 1. Fear & Greed

Fear & Greed는 BTC OHLCV에서 직접 계산하는 지표가 아닙니다. 외부 sentiment 또는
regime dataset이 필요합니다. 현재 후보는 다음입니다.

```text
https://www.kaggle.com/datasets/ashishpatel8736/historical-and-fear-greed-index-datasets
```

Primary `I60` feature:
- `fg_value`: `t`의 현재/as-of 값;
- `fg_mean_60`: 60일 sentiment regime 평균;
- `fg_delta_60`: full-window sentiment 변화;
- `fg_std_60`: image window 동안 sentiment 불안정성.

`I60` main setting에서 `fg_delta_20`을 쓰지 않습니다. `fg_delta_20`은 나중에
multi-scale ablation에서만 유지합니다.

`4-I2` audit에서 확인할 것:
- BTC Stage 2 기간과 date coverage가 맞는지;
- column 이름과 value scale;
- crypto-specific F&G인지 broad/equity F&G인지;
- missing date;
- timezone/publication timestamp 가정.

## 2. Bollinger

Bollinger feature는 BTC close에서 계산합니다. 추가 dataset은 필요 없습니다.

Primary `I60` 구현:

```text
sma60_t = mean(Close[t-59:t])
std60_t = std(Close[t-59:t])
upper60_t = sma60_t + 2 * std60_t
lower60_t = sma60_t - 2 * std60_t

bb_percent_b_60 = (Close_t - lower60_t) / (upper60_t - lower60_t)
bb_bandwidth_60 = (upper60_t - lower60_t) / sma60_t
```

해석:
- `%B`는 현재 종가가 lower/middle/upper band 또는 band 밖 어디에 있는지 보여줍니다.
- bandwidth는 60일 window가 압축되어 있는지, 변동성이 큰지 보여줍니다.

왜 `BB60`이 primary인가:
- chart image가 60일을 보므로 `BB60`이 CNN이 보는 동일한 가격 window를 요약합니다.
- 단순히 `BB20`이 흔한 기본값이라는 논리보다 방어하기 쉽습니다.
- FiLM 해석도 더 깔끔합니다. gamma/beta가 full-window chart regime에 의해
  조절된다고 설명할 수 있습니다.

`BB20`은 나중의 `standard_window` diagnostic으로만 유지합니다.

## 3. MFI

MFI는 BTC OHLCV에서 계산합니다. 추가 dataset은 필요 없습니다.

Primary `I60` 구현:

```text
typical_price_t = (High_t + Low_t + Close_t) / 3
raw_money_flow_t = typical_price_t * Volume_t

positive_flow_t = raw_money_flow_t if typical_price_t > typical_price_{t-1} else 0
negative_flow_t = raw_money_flow_t if typical_price_t < typical_price_{t-1} else 0

money_flow_ratio_60 =
    sum(positive_flow[t-59:t]) / sum(negative_flow[t-59:t])

mfi_60 = 100 - 100 / (1 + money_flow_ratio_60)
```

왜 `MFI60`이 primary인가:
- CNN이 보는 같은 60일 동안의 volume-adjusted buying/selling pressure를 요약합니다.
- `MFI14`를 관행으로 쓰는 것보다 Stage 4의 conditioning thesis와 더 잘 맞습니다.

`MFI14`는 나중의 `standard_window` diagnostic으로만 유지합니다.

구현 주의:
- negative flow가 0일 때 epsilon을 둡니다.
- raw distribution을 본 뒤 clipping 여부를 결정합니다.
- 최종 context feature는 train split 통계로만 normalize합니다.

## 4. 외부 데이터 필요 여부

| Feature family | 외부 dataset 필요? | 이유 |
|:---|:---:|:---|
| Fear & Greed | yes | 외부 sentiment/regime signal |
| Bollinger %B / bandwidth | no | BTC close에서 파생 |
| MFI | no | BTC OHLCV에서 파생 |
| Realized volatility | no | BTC close/log return에서 파생 |

따라서 첫 Stage 4 구현에서 외부 dataset이 필요한 것은 Fear & Greed 하나입니다. 나머지는
Stage 2 BTC OHLCV 파일에서 만들 수 있습니다.

## Context-window ablation 계획

Primary run:
1. `matched_window`: `BB60`, `MFI60`, `F&G60`, `RV60`.

Secondary diagnostic:
2. `standard_window`: `BB20`, `MFI14`, short F&G summary, `RV20`.
3. `multi_scale`: `BB20 + BB60`, `MFI14 + MFI60`, `F&G20 + F&G60`,
   `RV20 + RV60`.

권장 실행 순서:
1. 네 가지 Stage 4 model family를 `matched_window`로 먼저 실행합니다.
2. 가장 강한 model family가 보이면 그 모델부터 `standard_window`와 `multi_scale`을
   비교합니다.
3. window policy 자체가 중요한 결과가 될 때만 full grid로 확장합니다.

## Leakage와 normalization 규칙

1. image end date `t` 또는 그 이전 정보만 사용합니다.
2. 외부 daily context는 latest context date `<= t` 기준으로 as-of merge합니다.
3. 미래 날짜에서 backward fill하지 않습니다.
4. rolling feature는 trailing window만 사용합니다.
5. context imputation rule은 train split 기준으로만 정합니다.
6. context normalization 통계는 train split에서만 fit합니다.
7. prediction마다 사용된 context table을 date, raw value, normalized value,
   missing flag, merge source date와 함께 저장합니다.
