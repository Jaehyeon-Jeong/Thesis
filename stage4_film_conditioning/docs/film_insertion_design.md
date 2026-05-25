# Concat/Gating/FiLM Insertion Design

## English

Base Stage 2 block:

```text
Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d
```

Stage 4 FiLM block:

```text
Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d
```

Why this location:
- The FiLM reference notes emphasize applying FiLM after normalization.
- In this Stock_CNN setting, BatchNorm is already inside every convolution
  block.
- Therefore the minimal change is to insert FiLM between BatchNorm and
  LeakyReLU.

Expected tensor rule:
- Input activation after BatchNorm: `(batch_size, channels, height, width)`.
- Gamma: `(batch_size, channels)` or broadcastable
  `(batch_size, channels, 1, 1)`.
- Beta: `(batch_size, channels)` or broadcastable
  `(batch_size, channels, 1, 1)`.
- Output: same shape as input activation.

Initialization rule:
- Use `gamma = 1 + delta_gamma` so the model starts close to the unmodulated
  baseline.
- Initialize beta near zero.
- This keeps the first FiLM model from destroying the baseline feature scale at
  the start of training.

## Fixed I60 Shapes

The first Stage 4 main run uses `I60/R20/ohlc_ma_vb`.

| Location | Shape |
| --- | --- |
| input | `(B, 1, 96, 180)` |
| layer4 output after pool | `(B, 512, 2, 180)` |
| flatten | `(B, 184320)` |
| context embedding | `(B, 32)` |

## Variants to Compare

### 4-A. Concat

```text
image_feature:   (B, 184320)
context_embed:   (B, 32)
concat_feature:  (B, 184352)
classifier:      Dropout(0.5) -> Linear(184352, 2)
```

The convolution blocks remain unchanged. Context only reaches the final
classifier.

### 4-B. Gating

```text
F4:       (B, 512, 2, 180)
raw_gate: (B, 512)
gate = 2 * sigmoid(raw_gate)
F4' = F4 * gate[:, :, None, None]
```

Gate only the final I60 feature map in the first run. The gate head is
zero-initialized so the model starts with `gate = 1`.

### 4-C. Gamma-only FiLM

```text
Conv2d -> BatchNorm2d -> gamma * feature -> LeakyReLU -> MaxPool2d
gamma = 1 + delta_gamma
```

Generate one channel-wise gamma vector for each block:
`64`, `128`, `256`, and `512`.

### 4-D. Full FiLM

```text
Conv2d -> BatchNorm2d -> gamma * feature + beta -> LeakyReLU -> MaxPool2d
gamma = 1 + delta_gamma
beta = beta_delta
```

Generate channel-wise gamma and beta for every block.

## Initialization Guardrail

Zero-initialize the final gate/FiLM heads:
- `gate = 1` for 4-B;
- `gamma = 1` for 4-C/4-D;
- `beta = 0` for 4-D.

This protects the unconditioned feature scale at the start of training.

## 한국어

기존 Stage 2 block:

```text
Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d
```

Stage 4 FiLM block:

```text
Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d
```

이 위치를 쓰는 이유:
- FiLM reference note에서 normalization 뒤 FiLM 적용이 중요하게 다뤄집니다.
- 현재 Stock_CNN 구조에서는 모든 convolution block 안에 BatchNorm이 이미 있습니다.
- 따라서 가장 작은 변경은 BatchNorm과 LeakyReLU 사이에 FiLM을 넣는 것입니다.

예상 tensor 규칙:
- BatchNorm 뒤 activation: `(batch_size, channels, height, width)`.
- Gamma: `(batch_size, channels)` 또는 broadcast 가능한
  `(batch_size, channels, 1, 1)`.
- Beta: `(batch_size, channels)` 또는 broadcast 가능한
  `(batch_size, channels, 1, 1)`.
- Output: input activation과 같은 shape.

초기화 규칙:
- `gamma = 1 + delta_gamma`를 사용해서 model이 modulation 없는 baseline에
  가까운 상태에서 시작하게 합니다.
- beta는 zero 근처에서 시작합니다.
- 이렇게 해야 첫 FiLM 모델이 학습 시작부터 baseline feature scale을 망가뜨리지 않습니다.

## 고정 I60 shape

첫 Stage 4 main run은 `I60/R20/ohlc_ma_vb`를 사용합니다.

| 위치 | Shape |
| --- | --- |
| input | `(B, 1, 96, 180)` |
| layer4 output after pool | `(B, 512, 2, 180)` |
| flatten | `(B, 184320)` |
| context embedding | `(B, 32)` |

## 비교할 variant

### 4-A. Concat

```text
image_feature:   (B, 184320)
context_embed:   (B, 32)
concat_feature:  (B, 184352)
classifier:      Dropout(0.5) -> Linear(184352, 2)
```

Convolution block은 바꾸지 않습니다. Context는 마지막 classifier에서만 사용됩니다.

### 4-B. Gating

```text
F4:       (B, 512, 2, 180)
raw_gate: (B, 512)
gate = 2 * sigmoid(raw_gate)
F4' = F4 * gate[:, :, None, None]
```

첫 run에서는 final I60 feature map만 gate합니다. Gate head는 zero-initialize해서
`gate = 1`에서 시작합니다.

### 4-C. Gamma-only FiLM

```text
Conv2d -> BatchNorm2d -> gamma * feature -> LeakyReLU -> MaxPool2d
gamma = 1 + delta_gamma
```

각 block에 대해 channel-wise gamma vector를 만듭니다:
`64`, `128`, `256`, `512`.

### 4-D. Full FiLM

```text
Conv2d -> BatchNorm2d -> gamma * feature + beta -> LeakyReLU -> MaxPool2d
gamma = 1 + delta_gamma
beta = beta_delta
```

모든 block에 대해 channel-wise gamma와 beta를 만듭니다.

## 초기화 guardrail

최종 gate/FiLM head는 zero-initialize합니다.
- 4-B: `gate = 1`;
- 4-C/4-D: `gamma = 1`;
- 4-D: `beta = 0`.

이렇게 해서 학습 시작 시 unconditioned feature scale을 보호합니다.
