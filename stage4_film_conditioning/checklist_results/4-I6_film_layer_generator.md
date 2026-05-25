# 4-I6 FiLM Layer and Generator Modules

## English

Status: complete.

Implemented the FiLM building blocks needed for `4-C` gamma-only FiLM and
`4-D` full FiLM.

What was added:
- `src/stage4_film/layers/film.py`
- `src/stage4_film/layers/__init__.py`
- `src/stage4_film/conditions/film_generator.py`
- `scripts/check_stage4_film_layers.py`

FiLM layer:
- Module: `FeatureWiseAffineModulation`.
- Applies channel-wise modulation to a 4D CNN feature map.
- Gamma-only mode:
  - `F' = gamma * F`
- Full FiLM mode:
  - `F' = gamma * F + beta`
- Tensor rules:
  - feature map: `(B, C, H, W)`
  - gamma: `(B, C)` or `(B, C, 1, 1)`
  - beta: optional `(B, C)` or `(B, C, 1, 1)`

FiLM generator:
- Module: `FilmParameterGenerator`.
- Input: context embedding `(B, 32)`.
- For I60, it emits parameters for all four CNN blocks:
  - block 1: `64` channels
  - block 2: `128` channels
  - block 3: `256` channels
  - block 4: `512` channels
- `film_gamma` emits:
  - `gamma` and `delta_gamma` for each block
  - no `beta`
- `film_full` emits:
  - `gamma`, `delta_gamma`, and `beta` for each block

Identity initialization:
- The generator heads are zero-initialized.
- It predicts `delta_gamma`, then uses `gamma = 1 + delta_gamma`.
- Therefore at initialization:
  - `delta_gamma = 0`
  - `gamma = 1`
  - `beta = 0` for full FiLM
- This matches the practical FiLM initialization idea: start from the
  unconditioned CNN feature path, then learn context-dependent modulation.

Parameter check:
- Context encoder parameters: `1,344`.
- `film_gamma` generator parameters: `31,680`.
- `film_full` generator parameters: `63,360`.
- These counts are generator-head parameters only; full model counts will be
  checked in `4-I7` after the FiLM Stock_CNN wrappers are implemented.

Local validation:
- `python -m py_compile` passed.
- `python scripts/check_stage4_film_layers.py --config configs/env_local.yaml --batch-size 2`
  passed.
- I60 feature map shapes checked:
  - block 1: `(2, 64, 18, 180)`
  - block 2: `(2, 128, 5, 180)`
  - block 3: `(2, 256, 3, 180)`
  - block 4: `(2, 512, 2, 180)`
- `film_gamma` generated:
  - gamma shapes `(2, 64)`, `(2, 128)`, `(2, 256)`, `(2, 512)`
  - gamma min/max `1.0 / 1.0`
  - delta gamma min/max `0.0 / 0.0`
  - identity output for every block
- `film_full` generated:
  - gamma shapes `(2, 64)`, `(2, 128)`, `(2, 256)`, `(2, 512)`
  - beta shapes `(2, 64)`, `(2, 128)`, `(2, 256)`, `(2, 512)`
  - gamma min/max `1.0 / 1.0`
  - beta min/max `0.0 / 0.0`
  - identity output for every block
- Real context check:
  - Read local 4-I2 normalized context rows.
  - Context encoder produced finite `(2, 32)` embeddings.

Interpretation:
- `4-I6` does not yet define the full FiLM CNN models.
- It provides the reusable layer and generator modules.
- `4-I7` will insert these modules into the Stock_CNN blocks after BatchNorm
  and before LeakyReLU.

## 한국어

상태: 완료.

`4-C` gamma-only FiLM과 `4-D` full FiLM에 필요한 FiLM 부품을 구현했습니다.

추가한 파일:
- `src/stage4_film/layers/film.py`
- `src/stage4_film/layers/__init__.py`
- `src/stage4_film/conditions/film_generator.py`
- `scripts/check_stage4_film_layers.py`

FiLM layer:
- Module: `FeatureWiseAffineModulation`.
- 4D CNN feature map에 channel-wise modulation을 적용합니다.
- Gamma-only mode:
  - `F' = gamma * F`
- Full FiLM mode:
  - `F' = gamma * F + beta`
- Tensor 규칙:
  - feature map: `(B, C, H, W)`
  - gamma: `(B, C)` 또는 `(B, C, 1, 1)`
  - beta: optional `(B, C)` 또는 `(B, C, 1, 1)`

FiLM generator:
- Module: `FilmParameterGenerator`.
- 입력: context embedding `(B, 32)`.
- I60 기준 네 CNN block 전체에 대한 parameter를 만듭니다:
  - block 1: `64` channels
  - block 2: `128` channels
  - block 3: `256` channels
  - block 4: `512` channels
- `film_gamma`는 다음을 만듭니다:
  - block별 `gamma`, `delta_gamma`
  - `beta`는 만들지 않음
- `film_full`은 다음을 만듭니다:
  - block별 `gamma`, `delta_gamma`, `beta`

Identity initialization:
- Generator head는 0으로 초기화합니다.
- generator는 `delta_gamma`를 예측하고, 실제 gamma는 `gamma = 1 + delta_gamma`로
  만듭니다.
- 따라서 초기 상태는:
  - `delta_gamma = 0`
  - `gamma = 1`
  - full FiLM의 `beta = 0`
- 즉 처음에는 unconditioned CNN feature path를 그대로 통과시키고, 학습하면서
  context-dependent modulation을 배우게 됩니다.

Parameter check:
- Context encoder parameters: `1,344`.
- `film_gamma` generator parameters: `31,680`.
- `film_full` generator parameters: `63,360`.
- 이 수치는 generator head만의 parameter count입니다. 전체 FiLM CNN model
  parameter count는 `4-I7`에서 model wrapper를 만든 뒤 확인합니다.

Local validation:
- `python -m py_compile` 통과.
- `python scripts/check_stage4_film_layers.py --config configs/env_local.yaml --batch-size 2`
  통과.
- I60 feature map shape 확인:
  - block 1: `(2, 64, 18, 180)`
  - block 2: `(2, 128, 5, 180)`
  - block 3: `(2, 256, 3, 180)`
  - block 4: `(2, 512, 2, 180)`
- `film_gamma` 생성 결과:
  - gamma shapes `(2, 64)`, `(2, 128)`, `(2, 256)`, `(2, 512)`
  - gamma min/max `1.0 / 1.0`
  - delta gamma min/max `0.0 / 0.0`
  - 모든 block에서 identity output
- `film_full` 생성 결과:
  - gamma shapes `(2, 64)`, `(2, 128)`, `(2, 256)`, `(2, 512)`
  - beta shapes `(2, 64)`, `(2, 128)`, `(2, 256)`, `(2, 512)`
  - gamma min/max `1.0 / 1.0`
  - beta min/max `0.0 / 0.0`
  - 모든 block에서 identity output
- Real context check:
  - 4-I2에서 만든 local normalized context row를 읽었습니다.
  - context encoder가 finite `(2, 32)` embedding을 생성했습니다.

해석:
- `4-I6`은 아직 전체 FiLM CNN model을 정의한 단계가 아닙니다.
- 여기서는 재사용 가능한 FiLM layer와 generator module을 만든 것입니다.
- `4-I7`에서 이 module을 Stock_CNN block의 BatchNorm 뒤, LeakyReLU 앞에
  삽입합니다.
