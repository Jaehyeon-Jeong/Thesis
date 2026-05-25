# Stage 4 Context Fusion Ablation Plan

## English

Stage 4 is now defined by **how the market context is fused with the fixed BTC
chart-image CNN**. The context source and the fusion mechanism must be separated
so the thesis can answer both questions:

1. Does market context help?
2. If it helps, is FiLM better than simpler fusion methods?

## Fixed Image Baseline

The fixed primary chart-image baseline is:

```text
I60 / R20 / ohlc_ma_vb
```

Reasons:
- It is the best selected five-seed Stage 2 configuration.
- Five-seed accuracy mean: `0.5793`.
- Five-seed ROC-AUC mean: `0.5849`.
- It gives Stage 4 a stronger and more defensible baseline than the one-seed
  result alone.

The market context is not drawn into the image for the main experiment. The
image stays `ohlc_ma_vb`; the context is a separate vector.

## Primary Context Source

The first Stage 4 run should use structured numeric context:

| Feature | Meaning | Source type | Leakage rule |
|:---|:---|:---|:---|
| Fear & Greed score | Daily crypto sentiment/regime proxy | external daily index | use value known at or before image end date `t` |
| Bollinger %B | close position inside Bollinger band | computed from BTC OHLCV | rolling window ending at `t` only |
| Bollinger bandwidth | volatility/range proxy | computed from BTC OHLCV | rolling window ending at `t` only |
| MFI | volume-aware overbought/oversold proxy | computed from BTC OHLCV | rolling window ending at `t` only |
| realized volatility | recent market risk state | computed from BTC OHLCV | rolling window ending at `t` only |

These values become the first-run model input:

```text
context_vector[t] = [
    fg_value,
    fg_mean_60,
    fg_delta_60,
    fg_std_60,
    bb_percent_b_60,
    bb_bandwidth_60,
    mfi_60,
    rv_60,
]
```

The final vector is deliberately compact but still keeps current sentiment,
60-day sentiment level/change/instability, 60-day band location/range,
volume-aware pressure, and realized volatility. It is normalized using
train-split statistics only, then passed through the shared small MLP context
encoder:

```text
Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU
```

## Four Main Ablations

### 4-A. CNN + context concat

```text
chart image -> CNN -> image feature ----\
                                         concat -> classifier -> Up/Down
context -> MLP -> context embedding -----/
```

Purpose:
- Tests whether simply adding side information is enough.
- The context does not change how the CNN extracts image features.
- It only reaches the classifier after the image feature is already built.

Interpretation:
- If concat works as well as FiLM, FiLM may not be necessary.
- If FiLM beats concat, the result supports the claim that conditional feature
  modulation matters.

### 4-B. CNN + context gating

```text
chart image -> CNN -> image feature ------\
                                           multiply -> classifier -> Up/Down
context -> MLP -> gate vector -------------/
```

Purpose:
- Tests a simple context-dependent modulation method.
- The gate is usually channel-wise or feature-wise.
- A sigmoid gate can keep values between `0` and `1`, but a residual gate such
  as `1 + delta_gate` can also keep the model close to the baseline.

Interpretation:
- Gating answers whether multiplicative context control is already enough.
- It is a stronger comparison than concat but still simpler than full FiLM.

### 4-C. CNN + context FiLM gamma-only

```text
feature map F
context -> MLP -> gamma
F' = gamma * F
```

Purpose:
- Tests FiLM-style scaling without additive shift.
- This isolates the role of `gamma`.

Implementation rule:
- In convolutional blocks, `gamma` is block/channel-wise:
  `(batch, channels)` broadcast to `(batch, channels, height, width)`.
- Initialize near identity with `gamma = 1 + delta_gamma`.

Interpretation:
- Useful because FiLM papers often emphasize gamma analysis.
- If gamma-only performs well, scaling may be enough.

### 4-D. CNN + context FiLM full

```text
feature map F
context -> MLP -> gamma, beta
F' = gamma * F + beta
```

Purpose:
- Main Stage 4 model.
- Tests full feature-wise affine modulation.

Implementation rule:
- Insert FiLM after BatchNorm and before activation:

```text
Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d
```

Interpretation:
- This is the cleanest match to the advisor's direction:
  structured market context explicitly modulates visual feature extraction.
- `gamma` and `beta` can be saved and analyzed by date, regime, layer, channel,
  correctness, and confidence.

## News Context Position

News context is not removed. It is a second-phase extension.

Why not use news first:
- The numeric context ablation is faster and easier to audit for leakage.
- News requires publication-time alignment, daily aggregation, text cleaning,
  and reproducible encoding/caching.
- If numeric context already fails across concat/gating/FiLM, news can still be
  justified as a richer context source, but it should not be mixed before the
  core fusion comparison is understood.

How news can be used later:

```text
daily BTC news -> aggregation/summary/encoder -> news context vector
news context vector + optional numeric context -> concat/gating/FiLM
```

Candidate dataset:
- `edaschau/bitcoin_news` on Hugging Face.
- It contains Yahoo-Finance-sourced Bitcoin/crypto keyword matched articles.
- Public viewer metadata shows about `210k` rows, train split only, date range
  from 2011 to 2025, and columns including `date_time`, `title`, `source`,
  `url`, and `article_text`.

Recommended news order:
1. Headline-only daily aggregation.
2. Non-LLM encoder: TF-IDF/SVD or trainable embedding + GRU.
3. Optional article-summary context after caching rules are fixed.
4. Optional LLM embedding/summary only after reproducibility and runtime are
   defensible.

## 한국어

Stage 4는 이제 **market context를 고정된 BTC chart-image CNN에 어떻게 결합할지**를
검증하는 단계입니다. context source와 fusion mechanism을 분리해야 논문에서 두 가지
질문을 답할 수 있습니다.

1. 시장 맥락 정보가 도움이 되는가?
2. 도움이 된다면, FiLM이 단순 fusion보다 나은가?

## 고정 이미지 baseline

고정 primary chart-image baseline은 다음입니다.

```text
I60 / R20 / ohlc_ma_vb
```

이유:
- Stage 2 selected five-seed에서 가장 좋은 configuration입니다.
- Five-seed accuracy mean: `0.5793`.
- Five-seed ROC-AUC mean: `0.5849`.
- seed 하나 결과보다 방어 가능한 Stage 4 baseline입니다.

Main experiment에서는 market context를 image에 그리지 않습니다. 이미지는
`ohlc_ma_vb` 그대로 두고, context는 별도 vector로 넣습니다.

## 1차 context source

첫 Stage 4 run은 structured numeric context로 시작합니다.

| Feature | 의미 | Source type | Leakage rule |
|:---|:---|:---|:---|
| Fear & Greed score | daily crypto sentiment/regime proxy | external daily index | image end date `t` 또는 그 이전 값만 사용 |
| Bollinger %B | 종가가 Bollinger band 안에서 어디에 있는지 | BTC OHLCV 계산값 | `t`까지의 rolling window만 사용 |
| Bollinger bandwidth | 변동성/range proxy | BTC OHLCV 계산값 | `t`까지의 rolling window만 사용 |
| MFI | volume-aware overbought/oversold proxy | BTC OHLCV 계산값 | `t`까지의 rolling window만 사용 |
| realized volatility | 최근 시장 위험 상태 | BTC OHLCV 계산값 | `t`까지의 rolling window만 사용 |

이 값들은 첫 run의 model input vector가 됩니다.

```text
context_vector[t] = [
    fg_value,
    fg_mean_60,
    fg_delta_60,
    fg_std_60,
    bb_percent_b_60,
    bb_bandwidth_60,
    mfi_60,
    rv_60,
]
```

최종 벡터는 작게 유지하지만 current sentiment, 60일 sentiment level/change/instability,
60일 band location/range, volume-aware pressure, realized volatility를 포함합니다.
context vector는 train split 통계로만 normalize하고, 아래 shared MLP context encoder에
넣습니다.

```text
Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU
```

## 네 가지 main ablation

### 4-A. CNN + context concat

```text
chart image -> CNN -> image feature ----\
                                         concat -> classifier -> Up/Down
context -> MLP -> context embedding -----/
```

목적:
- side information을 그냥 추가하는 것만으로 충분한지 확인합니다.
- context는 CNN이 image feature를 추출하는 과정에는 개입하지 않습니다.
- image feature가 만들어진 뒤 classifier 직전에만 들어갑니다.

해석:
- concat이 FiLM과 비슷하게 좋으면 FiLM이 꼭 필요하지 않을 수 있습니다.
- FiLM이 concat보다 좋으면 conditional feature modulation이 중요하다는 주장을
  할 수 있습니다.

### 4-B. CNN + context gating

```text
chart image -> CNN -> image feature ------\
                                           multiply -> classifier -> Up/Down
context -> MLP -> gate vector -------------/
```

목적:
- 단순한 context-dependent modulation을 테스트합니다.
- gate는 보통 channel-wise 또는 feature-wise입니다.
- sigmoid gate로 `0~1` 값을 만들 수도 있고, baseline에 가깝게 시작하려면
  `1 + delta_gate` 방식도 가능합니다.

해석:
- gating은 multiplicative context control만으로 충분한지 확인합니다.
- concat보다 강한 비교군이고, full FiLM보다는 단순합니다.

### 4-C. CNN + context FiLM gamma-only

```text
feature map F
context -> MLP -> gamma
F' = gamma * F
```

목적:
- additive shift 없이 FiLM-style scaling만 테스트합니다.
- `gamma`의 역할을 분리해서 봅니다.

구현 규칙:
- convolution block에서는 `gamma`가 block/channel-wise입니다:
  `(batch, channels)`를 `(batch, channels, height, width)`로 broadcast합니다.
- `gamma = 1 + delta_gamma`로 identity 근처에서 시작합니다.

해석:
- FiLM 논문에서 gamma 분석이 중요하기 때문에 필요한 비교입니다.
- gamma-only가 좋으면 scaling만으로 충분하다는 해석이 가능합니다.

### 4-D. CNN + context FiLM full

```text
feature map F
context -> MLP -> gamma, beta
F' = gamma * F + beta
```

목적:
- Stage 4 main model입니다.
- full feature-wise affine modulation을 테스트합니다.

구현 규칙:
- FiLM은 BatchNorm 뒤, activation 전에 삽입합니다.

```text
Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d
```

해석:
- 교수님 방향성과 가장 잘 맞습니다:
  structured market context가 visual feature extraction을 명시적으로 조절합니다.
- `gamma`, `beta`를 저장해서 date, regime, layer, channel, correctness,
  confidence별로 분석할 수 있습니다.

## News context 위치

뉴스 context는 제거하지 않습니다. 2차 확장으로 둡니다.

뉴스를 바로 main으로 쓰지 않는 이유:
- numeric context ablation이 더 빠르고 leakage audit이 쉽습니다.
- news는 publication-time alignment, daily aggregation, text cleaning,
  reproducible encoding/cache가 필요합니다.
- numeric context가 concat/gating/FiLM에서 실패하더라도, news는 더 풍부한 context
  source로 여전히 실험 가치가 있습니다. 다만 core fusion 비교 전에 섞지 않는 것이
  안전합니다.

뉴스를 나중에 쓰는 방식:

```text
daily BTC news -> aggregation/summary/encoder -> news context vector
news context vector + optional numeric context -> concat/gating/FiLM
```

후보 dataset:
- Hugging Face의 `edaschau/bitcoin_news`.
- Yahoo Finance 기반 Bitcoin/crypto keyword matched article dataset입니다.
- 공개 viewer metadata 기준 약 `210k` rows, train split only, 2011-2025 date range,
  `date_time`, `title`, `source`, `url`, `article_text` columns를 포함합니다.

추천 news 진행 순서:
1. Headline-only daily aggregation.
2. Non-LLM encoder: TF-IDF/SVD 또는 trainable embedding + GRU.
3. Cache rule이 고정된 뒤 optional article-summary context.
4. 재현성과 runtime이 방어 가능할 때 optional LLM embedding/summary.
