# Stage 4 Pipeline

## English

Stage 4 keeps the Stage 2 BTC chart-image pipeline fixed and adds a second
input: market context. The experiments compare how that context is attached to
the CNN.

## Fixed Inputs

From Stage 2:
- BTC OHLCV loading and chart image generation.
- Label construction: future `R20` return up/down.
- Split, normalization, evaluation, trading metrics, Grad-CAM conventions.
- Primary model family: `I60/R20/ohlc_ma_vb`.

From Stage 3:
- Linear adapter result is a comparison point only.
- It does not change the Stage 4 architecture.

## Stage 4 Data Flow

```text
BTC OHLCV
  -> I60/R20/ohlc_ma_vb chart image
  -> Stock_CNN image branch

BTC OHLCV + external market context
  -> context feature builder
  -> train-only normalization
  -> MLP context encoder

image branch + context branch
  -> concat / gating / gamma-only FiLM / full FiLM
  -> Up/Down logits
  -> predictions, metrics, trading metrics
  -> Grad-CAM + context/gate/gamma/beta logs
```

## Main Numeric Context Flow

```text
image_end_date = t

context_vector[t] = [
    fg_value[t or previous available],
    fg_mean_60[t],
    fg_delta_60[t],
    fg_std_60[t],
    bollinger_percent_b_60[t],
    bollinger_bandwidth_60[t],
    mfi_60[t],
    realized_volatility_60[t],
]
```

All context features must satisfy:
- available at or before `t`;
- no use of `t+1 ... t+R20` information;
- train-only normalization statistics;
- explicit missing-value policy.
- F&G is the only first-run external dataset. Bollinger, MFI, and realized
  volatility are derived from BTC OHLCV.
- The first run uses compact trailing summaries, not a raw 60-step context
  sequence.
- The first run uses `context_window = image_window`. For the selected `I60`
  baseline this means 60-day context features. `BB20` and `MFI14` are kept only
  as later standard-window diagnostics.

## Four Model Flows

### 4-A. Concat

```text
image -> CNN -> final image feature
context -> MLP -> context embedding
[image feature, context embedding] -> classifier
```

### 4-B. Gating

```text
image -> CNN -> image feature
context -> MLP -> gate
image feature * gate -> classifier
```

### 4-C. Gamma-only FiLM

```text
context -> MLP -> block-wise gamma
Conv -> BN -> gamma * feature -> LeakyReLU -> MaxPool
```

### 4-D. Full FiLM

```text
context -> MLP -> block-wise gamma, beta
Conv -> BN -> gamma * feature + beta -> LeakyReLU -> MaxPool
```

## News Context Extension

News is a planned second-phase context source.

```text
BTC news rows
  -> publication-time audit
  -> daily aggregation
  -> headline/article summary or embedding
  -> news context vector
  -> same concat/gating/FiLM heads
```

The news context should not be implemented until:
- the source is downloaded/versioned;
- publication timestamps are aligned to BTC trading dates;
- the aggregation rule is fixed;
- the encoder/cache policy is reproducible.

## Output Requirements

Each Stage 4 run must save:
- predictions;
- classification metrics;
- trading metrics;
- context feature statistics;
- gate/gamma/beta logs where applicable;
- Grad-CAM figures;
- run manifest with config, seed, source version, and Git commit.

## 한국어

Stage 4는 Stage 2 BTC chart-image pipeline을 고정하고, 두 번째 입력으로 market
context를 추가합니다. 실험의 핵심은 이 context를 CNN에 어떻게 붙이는지 비교하는
것입니다.

## 고정 입력

Stage 2에서 고정해서 가져오는 것:
- BTC OHLCV loading과 chart image generation.
- Label construction: future `R20` return up/down.
- Split, normalization, evaluation, trading metrics, Grad-CAM 규칙.
- Primary model family: `I60/R20/ohlc_ma_vb`.

Stage 3에서 가져오는 것:
- Linear adapter 결과는 비교 대상일 뿐입니다.
- Stage 4 architecture를 바꾸는 dependency가 아닙니다.

## Stage 4 데이터 흐름

```text
BTC OHLCV
  -> I60/R20/ohlc_ma_vb chart image
  -> Stock_CNN image branch

BTC OHLCV + external market context
  -> context feature builder
  -> train-only normalization
  -> MLP context encoder

image branch + context branch
  -> concat / gating / gamma-only FiLM / full FiLM
  -> Up/Down logits
  -> predictions, metrics, trading metrics
  -> Grad-CAM + context/gate/gamma/beta logs
```

## Main numeric context 흐름

```text
image_end_date = t

context_vector[t] = [
    fg_value[t 또는 직전 available 값],
    fg_mean_60[t],
    fg_delta_60[t],
    fg_std_60[t],
    bollinger_percent_b_60[t],
    bollinger_bandwidth_60[t],
    mfi_60[t],
    realized_volatility_60[t],
]
```

모든 context feature는 다음을 지켜야 합니다.
- `t` 또는 그 이전에 알 수 있어야 합니다.
- `t+1 ... t+R20` 정보를 쓰면 안 됩니다.
- normalization 통계는 train split에서만 fit합니다.
- missing-value policy를 명시해야 합니다.
- 첫 run에서 외부 dataset이 필요한 것은 F&G뿐입니다. Bollinger, MFI, realized
  volatility는 BTC OHLCV에서 파생합니다.
- 첫 run은 raw 60-step context sequence가 아니라 compact trailing summary를
  사용합니다.
- 첫 run은 `context_window = image_window`를 사용합니다. 선택된 `I60` baseline에서는
  60일 context feature를 뜻합니다. `BB20`, `MFI14`는 나중의 standard-window
  diagnostic으로만 유지합니다.

## 네 가지 model flow

### 4-A. Concat

```text
image -> CNN -> final image feature
context -> MLP -> context embedding
[image feature, context embedding] -> classifier
```

### 4-B. Gating

```text
image -> CNN -> image feature
context -> MLP -> gate
image feature * gate -> classifier
```

### 4-C. Gamma-only FiLM

```text
context -> MLP -> block-wise gamma
Conv -> BN -> gamma * feature -> LeakyReLU -> MaxPool
```

### 4-D. Full FiLM

```text
context -> MLP -> block-wise gamma, beta
Conv -> BN -> gamma * feature + beta -> LeakyReLU -> MaxPool
```

## News context 확장

뉴스는 2차 context source로 계획합니다.

```text
BTC news rows
  -> publication-time audit
  -> daily aggregation
  -> headline/article summary or embedding
  -> news context vector
  -> same concat/gating/FiLM heads
```

뉴스 context는 아래가 정해지기 전까지 구현하지 않습니다.
- source download/version;
- publication timestamp와 BTC trading date 정렬;
- aggregation rule;
- encoder/cache 재현성 정책.

## Output 요구사항

각 Stage 4 run은 다음을 저장해야 합니다.
- predictions;
- classification metrics;
- trading metrics;
- context feature statistics;
- 필요한 경우 gate/gamma/beta logs;
- Grad-CAM figures;
- config, seed, source version, Git commit이 들어간 run manifest.
