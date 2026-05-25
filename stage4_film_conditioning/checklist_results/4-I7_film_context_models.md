# 4-I7 FiLM Context Stock_CNN Models

## English

Status: complete.

Implemented the two Stage 4 FiLM model variants:

- `CNN + context FiLM gamma-only`
- `CNN + context FiLM full`

Code:

- `src/stage4_film/models/film_stock_cnn.py`
- `scripts/check_stage4_model_shapes.py`

Model structure:

```text
chart image
  -> Stage 2 Stock_CNN block
       Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d
  -> flatten
  -> Stage 2-style classifier
  -> logits

market context vector (8)
  -> shared context MLP
  -> context embedding (32)
  -> FiLM generator
  -> block-wise gamma/beta
```

This implements the Stage 4 decision from `4-6`: FiLM is inserted after
BatchNorm and before LeakyReLU in every I60 CNN block.

Modes:

- `film_gamma`
  - applies `F' = gamma * F`
  - emits gamma for channels `[64, 128, 256, 512]`
- `film_full`
  - applies `F' = gamma * F + beta`
  - emits gamma and beta for channels `[64, 128, 256, 512]`

Tensor path for the primary I60 model:

- image: `(B, 1, 96, 180)`
- context: `(B, 8)`
- context embedding: `(B, 32)`
- block 1 FiLM: gamma `(B, 64)`, output `(B, 64, 36, 180)`,
  pooled block output `(B, 64, 18, 180)`
- block 2 FiLM: gamma `(B, 128)`, output `(B, 128, 10, 180)`,
  pooled block output `(B, 128, 5, 180)`
- block 3 FiLM: gamma `(B, 256)`, output `(B, 256, 6, 180)`,
  pooled block output `(B, 256, 3, 180)`
- block 4 FiLM: gamma `(B, 512)`, output `(B, 512, 5, 180)`,
  pooled block output `(B, 512, 2, 180)`
- flatten: `(B, 184320)`
- logits: `(B, 2)`

Parameter checks:

| model | parameters | delta vs Stage 2 I60 |
| --- | ---: | ---: |
| Stage 2 I60 baseline | 2,952,962 | 0 |
| Stage 4 `film_gamma` | 2,985,986 | +33,024 |
| Stage 4 `film_full` | 3,017,666 | +64,704 |

Why these deltas:

- Both FiLM models add the shared context encoder: `1,344` parameters.
- `film_gamma` adds the gamma generator: `31,680` parameters.
- `film_full` adds gamma and beta generators: `63,360` parameters.

Initialization:

- Gamma is generated as `1 + delta_gamma`.
- Gamma heads are zero-initialized.
- Beta heads are zero-initialized for full FiLM.
- Therefore the first forward pass starts as identity modulation:
  - gamma min/max: `1.0 / 1.0`
  - beta min/max: `0.0 / 0.0`
  - post-FiLM feature maps equal pre-FiLM feature maps.

Implementation detail:

- Stage 2 uses `LeakyReLU(inplace=True)`.
- For later interpretation/export, `forward_with_modulation_values()` stores
  pre-FiLM and post-FiLM feature maps before that in-place activation mutates
  tensors.

Validation:

- `python -m py_compile` passed for the new model and updated checker.
- `python scripts/check_stage4_model_shapes.py --config configs/env_local.yaml --model film_gamma --batch-size 2`
  passed.
- `python scripts/check_stage4_model_shapes.py --config configs/env_local.yaml --model film_full --batch-size 2`
  passed.
- Both checks used dummy tensors and real normalized context rows from the local
  `4-I2` context table when available.

Next:

- `4-I8`: implement the Stage 4 runner that uses the fixed Stage 2 BTC data
  pipeline and trains/evaluates `concat`, `gating`, `film_gamma`, and
  `film_full`.

## 한국어

상태: 완료.

Stage 4 FiLM model 두 가지를 구현했습니다:

- `CNN + context FiLM gamma-only`
- `CNN + context FiLM full`

코드:

- `src/stage4_film/models/film_stock_cnn.py`
- `scripts/check_stage4_model_shapes.py`

모델 구조:

```text
chart image
  -> Stage 2 Stock_CNN block
       Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d
  -> flatten
  -> Stage 2-style classifier
  -> logits

market context vector (8)
  -> shared context MLP
  -> context embedding (32)
  -> FiLM generator
  -> block-wise gamma/beta
```

즉 `4-6`에서 정한 대로 모든 I60 CNN block 안에서 BatchNorm 뒤,
LeakyReLU 전에 FiLM을 삽입했습니다.

Mode:

- `film_gamma`
  - `F' = gamma * F`
  - channel `[64, 128, 256, 512]`에 대한 gamma를 만듭니다.
- `film_full`
  - `F' = gamma * F + beta`
  - channel `[64, 128, 256, 512]`에 대한 gamma와 beta를 만듭니다.

Primary I60 model의 tensor path:

- image: `(B, 1, 96, 180)`
- context: `(B, 8)`
- context embedding: `(B, 32)`
- block 1 FiLM: gamma `(B, 64)`, output `(B, 64, 36, 180)`,
  pooled block output `(B, 64, 18, 180)`
- block 2 FiLM: gamma `(B, 128)`, output `(B, 128, 10, 180)`,
  pooled block output `(B, 128, 5, 180)`
- block 3 FiLM: gamma `(B, 256)`, output `(B, 256, 6, 180)`,
  pooled block output `(B, 256, 3, 180)`
- block 4 FiLM: gamma `(B, 512)`, output `(B, 512, 5, 180)`,
  pooled block output `(B, 512, 2, 180)`
- flatten: `(B, 184320)`
- logits: `(B, 2)`

Parameter check:

| model | parameters | Stage 2 I60 대비 증가 |
| --- | ---: | ---: |
| Stage 2 I60 baseline | 2,952,962 | 0 |
| Stage 4 `film_gamma` | 2,985,986 | +33,024 |
| Stage 4 `film_full` | 3,017,666 | +64,704 |

왜 이렇게 증가하는가:

- 두 FiLM model 모두 shared context encoder `1,344` parameter를 추가합니다.
- `film_gamma`는 gamma generator `31,680` parameter를 추가합니다.
- `film_full`은 gamma와 beta generator `63,360` parameter를 추가합니다.

초기화:

- Gamma는 `1 + delta_gamma`로 생성합니다.
- Gamma head는 zero-initialized입니다.
- Full FiLM의 beta head도 zero-initialized입니다.
- 그래서 첫 forward pass는 identity modulation에서 시작합니다:
  - gamma min/max: `1.0 / 1.0`
  - beta min/max: `0.0 / 0.0`
  - post-FiLM feature map은 pre-FiLM feature map과 같습니다.

구현상 주의:

- Stage 2 block은 `LeakyReLU(inplace=True)`를 사용합니다.
- 그래서 해석/export용 `forward_with_modulation_values()`에서는 in-place
  activation이 tensor를 바꾸기 전에 pre-FiLM/post-FiLM feature map을 따로
  저장하도록 했습니다.

검증:

- 새 model과 수정된 checker에 대해 `python -m py_compile` 통과.
- `python scripts/check_stage4_model_shapes.py --config configs/env_local.yaml --model film_gamma --batch-size 2`
  통과.
- `python scripts/check_stage4_model_shapes.py --config configs/env_local.yaml --model film_full --batch-size 2`
  통과.
- Dummy tensor와 local `4-I2` context table의 실제 normalized context row 모두에서
  확인했습니다.

다음:

- `4-I8`: 고정된 Stage 2 BTC data pipeline을 사용해서 `concat`, `gating`,
  `film_gamma`, `film_full`을 학습/평가하는 Stage 4 runner 구현.
