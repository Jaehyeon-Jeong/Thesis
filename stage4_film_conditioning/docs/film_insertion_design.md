# FiLM Insertion Design

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

Variants to compare:
- Gamma-only FiLM: `gamma * feature`.
- Full FiLM: `gamma * feature + beta`.

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

비교할 variant:
- Gamma-only FiLM: `gamma * feature`.
- Full FiLM: `gamma * feature + beta`.
