# 4-2 Structured Numeric Context Audit and Leakage Policy

## English

Status: complete as a planning decision; implementation audit remains in `4-I2`.

Stage 4 uses the selected Stage 2 baseline:

```text
image_window = I60
return_horizon = R20
image_spec = ohlc_ma_vb
```

For every sample, define:

```text
t = image end date
image window = [t - 59, ..., t]
label window = (t, ..., t + 20]
```

All context features must be known at or before `t`. If a source's publication
time is ambiguous, use the most recent value strictly before `t` as the
conservative default.

## Recommended First Context Vector

Use a compact daily vector rather than a 60-step sequence:

| Feature | Source | Window | Definition | Leakage rule |
|:---|:---|:---|:---|:---|
| `fg_value` | external F&G dataset | latest `<= t` | Fear & Greed value at decision date | as-of merge only, no backfill |
| `fg_mean_60` | external F&G dataset | last 60 days through `t` | rolling mean of F&G | rolling window uses only dates `<= t` |
| `fg_delta_20` | external F&G dataset | `t` vs `t-20` | short-term sentiment change | use prior available values only |
| `bb_percent_b_20` | BTC OHLCV | 20 days through `t` | Bollinger band position | computed from close prices `<= t` |
| `bb_bandwidth_20` | BTC OHLCV | 20 days through `t` | band width / middle band | computed from close prices `<= t` |
| `mfi_14` | BTC OHLCV | 14 days through `t` | Money Flow Index | computed from OHLCV `<= t` |
| `rv_20` | BTC OHLCV | 20 days through `t` | realized volatility of log returns | computed from close prices `<= t` |
| `rv_60` | BTC OHLCV | 60 days through `t` | slower realized volatility regime | computed from close prices `<= t` |

This keeps the condition vector small, auditable, and compatible with an MLP
condition encoder.

## 1. Fear & Greed

Fear & Greed should not be derived from BTC OHLCV. It is an external sentiment
or market-regime series. The user-provided candidate is:

```text
https://www.kaggle.com/datasets/ashishpatel8736/historical-and-fear-greed-index-datasets
```

Planning decision:
- Do not feed the whole 60-day F&G sequence into the first Stage 4 model.
- Use summary features:
  - current/as-of value: `fg_value`;
  - 60-day regime average: `fg_mean_60`;
  - 20-day change: `fg_delta_20`.

Why this is better for the first experiment:
- It is faster and easier to audit.
- It avoids adding a sequence encoder before proving the value of context.
- It is easier to interpret FiLM gamma/beta against simple regime variables.
- It still captures both the current state and the recent sentiment regime.

If the Kaggle F&G dataset is not crypto-specific, record it as a broad market
sentiment proxy or switch to a crypto-specific Fear & Greed source. The `4-I2`
audit must check:
- date coverage against BTC Stage 2 dates;
- column names and value scale;
- whether the index is crypto F&G or equity/CNN F&G;
- missing dates;
- timezone/publication timestamp assumptions.

## 2. Bollinger Features

Bollinger features are derived directly from BTC close prices, so no extra
dataset is needed.

Default implementation:

```text
sma20_t = mean(Close[t-19:t])
std20_t = std(Close[t-19:t])
upper20_t = sma20_t + 2 * std20_t
lower20_t = sma20_t - 2 * std20_t
bb_percent_b_20 = (Close_t - lower20_t) / (upper20_t - lower20_t)
bb_bandwidth_20 = (upper20_t - lower20_t) / sma20_t
```

Use both `%B` and bandwidth:
- `%B` tells where price sits inside/outside the band.
- bandwidth tells whether the regime is compressed or volatile.

The first implementation should use the standard 20-day Bollinger window even
for `I60` images. The 20-day calculation is fully contained inside the 60-day
image window and matches the selected `R20` evaluation horizon.

## 3. Money Flow Index

MFI is also derived from BTC OHLCV, so no extra dataset is needed.

Default implementation:

```text
typical_price_t = (High_t + Low_t + Close_t) / 3
raw_money_flow_t = typical_price_t * Volume_t

positive_flow_t = raw_money_flow_t if typical_price_t > typical_price_{t-1} else 0
negative_flow_t = raw_money_flow_t if typical_price_t < typical_price_{t-1} else 0

money_flow_ratio_14 =
    sum(positive_flow[t-13:t]) / sum(negative_flow[t-13:t])

mfi_14 = 100 - 100 / (1 + money_flow_ratio_14)
```

Implementation notes:
- Use an epsilon when negative flow is zero.
- Clip or preserve out-of-range values only after checking the raw distribution.
- Normalize the final context features with train-only statistics.

## 4. Which Features Need External Data?

| Feature family | Needs external dataset? | Reason |
|:---|:---:|:---|
| Fear & Greed | yes | external sentiment/regime signal |
| Bollinger %B / bandwidth | no | derived from BTC close prices |
| MFI | no | derived from BTC OHLCV |
| Realized volatility | no | derived from BTC close/log returns |

So the first Stage 4 implementation only needs one extra attached dataset:
Fear & Greed. Everything else can be produced from the Stage 2 BTC OHLCV file.

## Leakage and Normalization Rules

1. Use only information available at or before image end date `t`.
2. External daily context uses as-of merge: latest context date `<= t`.
3. No backward fill from future dates.
4. Rolling features are computed with trailing windows only.
5. Context imputation rules are fitted on train only.
6. Context normalization statistics are fitted on train only.
7. Save the context table used for every prediction with date, raw values,
   normalized values, missing flags, and merge source date.

## Recommended Source Ablation After First Run

The first model should use the full compact vector above. If it improves or
behaves oddly, run a small source ablation:

1. `technical_only`: Bollinger + MFI + realized volatility.
2. `sentiment_only`: F&G features only.
3. `all_context`: F&G + technical indicators.

This separates "FiLM works because it has external sentiment" from "FiLM works
because OHLCV-derived regime summaries help the CNN."

## 한국어

상태: 계획 결정 완료. 실제 구현 audit은 `4-I2`에서 진행합니다.

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
발표 시간이 애매하면, 보수적으로 `t`보다 이전의 가장 최근 값을 씁니다.

## 첫 context vector 권장안

60일짜리 sequence를 그대로 넣기보다 compact daily vector를 먼저 씁니다.

| Feature | Source | Window | 정의 | Leakage 규칙 |
|:---|:---|:---|:---|:---|
| `fg_value` | 외부 F&G dataset | latest `<= t` | decision date의 Fear & Greed 값 | as-of merge만 사용, future backfill 금지 |
| `fg_mean_60` | 외부 F&G dataset | `t`까지 60일 | F&G rolling mean | `<= t` 날짜만 사용 |
| `fg_delta_20` | 외부 F&G dataset | `t` vs `t-20` | 단기 sentiment 변화 | 이전 이용 가능 값만 사용 |
| `bb_percent_b_20` | BTC OHLCV | `t`까지 20일 | Bollinger band 내 위치 | `<= t` close만 사용 |
| `bb_bandwidth_20` | BTC OHLCV | `t`까지 20일 | band width / middle band | `<= t` close만 사용 |
| `mfi_14` | BTC OHLCV | `t`까지 14일 | Money Flow Index | `<= t` OHLCV만 사용 |
| `rv_20` | BTC OHLCV | `t`까지 20일 | log return realized volatility | `<= t` close만 사용 |
| `rv_60` | BTC OHLCV | `t`까지 60일 | 느린 volatility regime | `<= t` close만 사용 |

이 방식은 condition vector가 작고, audit하기 쉽고, MLP condition encoder와 잘 맞습니다.

## 1. Fear & Greed

Fear & Greed는 BTC OHLCV에서 직접 계산하는 지표가 아닙니다. 외부 sentiment 또는
market-regime series입니다. 사용자가 제시한 후보는 다음입니다.

```text
https://www.kaggle.com/datasets/ashishpatel8736/historical-and-fear-greed-index-datasets
```

계획 결정:
- 첫 Stage 4 모델에는 F&G 60일 raw sequence 전체를 넣지 않습니다.
- summary feature를 씁니다:
  - 현재/as-of 값: `fg_value`;
  - 60일 regime 평균: `fg_mean_60`;
  - 20일 변화: `fg_delta_20`.

이게 첫 실험에 더 좋은 이유:
- 빠르고 audit하기 쉽습니다.
- context 효과를 보기 전에 sequence encoder를 추가하지 않아도 됩니다.
- FiLM gamma/beta를 단순 regime 변수와 연결해서 해석하기 쉽습니다.
- 현재 상태와 최근 sentiment regime을 둘 다 담습니다.

만약 Kaggle F&G dataset이 crypto-specific이 아니라면, broad market sentiment
proxy로 기록하거나 crypto-specific Fear & Greed source로 바꿔야 합니다. `4-I2`
audit에서 확인할 것:
- BTC Stage 2 기간과 date coverage가 맞는지;
- column 이름과 value scale;
- crypto F&G인지 equity/CNN F&G인지;
- missing dates;
- timezone/publication timestamp 가정.

## 2. Bollinger

Bollinger feature는 BTC close에서 바로 계산합니다. 추가 dataset은 필요 없습니다.

기본 구현:

```text
sma20_t = mean(Close[t-19:t])
std20_t = std(Close[t-19:t])
upper20_t = sma20_t + 2 * std20_t
lower20_t = sma20_t - 2 * std20_t
bb_percent_b_20 = (Close_t - lower20_t) / (upper20_t - lower20_t)
bb_bandwidth_20 = (upper20_t - lower20_t) / sma20_t
```

둘 다 씁니다.
- `%B`는 가격이 band 안/밖 어디에 있는지 말합니다.
- bandwidth는 시장이 압축되어 있는지, 변동성이 큰지 말합니다.

첫 구현에서는 `I60` image라도 표준 20-day Bollinger를 씁니다. 20일 계산은 60일
image window 안에 완전히 포함되고, 선택한 `R20` 평가 horizon과도 맞습니다.

## 3. MFI

MFI도 BTC OHLCV에서 바로 계산합니다. 추가 dataset은 필요 없습니다.

기본 구현:

```text
typical_price_t = (High_t + Low_t + Close_t) / 3
raw_money_flow_t = typical_price_t * Volume_t

positive_flow_t = raw_money_flow_t if typical_price_t > typical_price_{t-1} else 0
negative_flow_t = raw_money_flow_t if typical_price_t < typical_price_{t-1} else 0

money_flow_ratio_14 =
    sum(positive_flow[t-13:t]) / sum(negative_flow[t-13:t])

mfi_14 = 100 - 100 / (1 + money_flow_ratio_14)
```

구현 주의:
- negative flow가 0일 때 epsilon을 둡니다.
- out-of-range clip 여부는 raw distribution을 본 뒤 결정합니다.
- 최종 context feature는 train split 통계로만 normalize합니다.

## 4. 외부 데이터가 필요한 지표

| Feature family | 외부 dataset 필요? | 이유 |
|:---|:---:|:---|
| Fear & Greed | yes | 외부 sentiment/regime signal |
| Bollinger %B / bandwidth | no | BTC close에서 파생 |
| MFI | no | BTC OHLCV에서 파생 |
| Realized volatility | no | BTC close/log return에서 파생 |

따라서 첫 Stage 4 구현에서 추가로 attach해야 하는 dataset은 Fear & Greed 하나입니다.
나머지는 Stage 2 BTC OHLCV 파일에서 만들 수 있습니다.

## Leakage와 normalization 규칙

1. image end date `t` 또는 그 이전 정보만 사용합니다.
2. 외부 daily context는 latest context date `<= t` 기준으로 as-of merge합니다.
3. 미래 날짜에서 backward fill하지 않습니다.
4. rolling feature는 trailing window만 사용합니다.
5. context imputation rule은 train split 기준으로만 정합니다.
6. context normalization 통계는 train split에서만 fit합니다.
7. prediction마다 사용된 context table을 date, raw value, normalized value,
   missing flag, merge source date와 함께 저장합니다.

## 첫 실행 이후 source ablation 권장

첫 모델은 위 compact vector 전체를 씁니다. 성능이 좋아지거나 이상하게 움직이면
작은 source ablation을 진행합니다.

1. `technical_only`: Bollinger + MFI + realized volatility.
2. `sentiment_only`: F&G features only.
3. `all_context`: F&G + technical indicators.

이렇게 해야 "FiLM이 외부 sentiment 때문에 좋아졌는지"와 "OHLCV에서 파생한 regime
summary가 CNN에 도움이 된 것인지"를 분리해서 설명할 수 있습니다.
