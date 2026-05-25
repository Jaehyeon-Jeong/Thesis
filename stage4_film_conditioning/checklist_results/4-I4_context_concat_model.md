# 4-I4 Context Concat Model

## English

Status: complete.

Implemented the first Stage 4 ablation model: `CNN + context concat`.

What was added:
- `src/stage4_film/models/context_stock_cnn.py`
- `src/stage4_film/models/__init__.py`
- `scripts/check_stage4_model_shapes.py`

Model structure:
- The Stage 2 Stock_CNN convolution blocks are reused unchanged.
- The Stage 2 final classifier is replaced.
- Image branch:
  - Input: `(B, 1, 96, 180)` for primary `I60`.
  - CNN block output: `(B, 512, 2, 180)`.
  - Flattened image feature: `(B, 184320)`.
- Context branch:
  - Input: normalized 8-feature context vector `(B, 8)`.
  - Shared context encoder output: `(B, 32)`.
- Fusion:
  - Concat feature: `(B, 184320 + 32) = (B, 184352)`.
  - Final classifier: `Dropout(0.5) -> Linear(184352, 2)`.
  - Output logits: `(B, 2)`.

Parameter check:
- Stage 2 I60 baseline expected parameters: `2,952,962`.
- Stage 4 concat actual/expected parameters: `2,954,370`.
- Delta vs Stage 2 baseline: `+1,408`.
- The increase is small because only the context encoder and 32 extra final
  classifier inputs are added.

Local validation:
- `python -m py_compile` passed for the new model and checker files.
- `python scripts/check_stage4_model_shapes.py --config configs/env_local.yaml --model concat --batch-size 2`
  passed.
- Dummy check:
  - `image`: `(2, 1, 96, 180)`
  - `flatten`: `(2, 184320)`
  - `context`: `(2, 8)`
  - `context_embedding`: `(2, 32)`
  - `concat_feature`: `(2, 184352)`
  - `logits`: `(2, 2)`
- Real context check:
  - Read local 4-I2 normalized context rows from
    `outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_features.csv`.
  - Full forward pass produced finite `(2, 2)` logits.

Interpretation:
- This is the `4-A` baseline for the advisor-requested ablation.
- It tests whether simply appending market context to the visual CNN feature is
  enough before moving to gating and FiLM.
- It does not yet modulate intermediate CNN feature maps; that starts at `4-I5`
  gating and `4-I6/4-I7` FiLM.

## 한국어

상태: 완료.

첫 Stage 4 ablation 모델인 `CNN + context concat`을 구현했습니다.

추가한 파일:
- `src/stage4_film/models/context_stock_cnn.py`
- `src/stage4_film/models/__init__.py`
- `scripts/check_stage4_model_shapes.py`

모델 구조:
- Stage 2 Stock_CNN convolution block은 그대로 재사용합니다.
- Stage 2의 마지막 classifier만 교체합니다.
- Image branch:
  - 입력: primary `I60` 기준 `(B, 1, 96, 180)`.
  - CNN block output: `(B, 512, 2, 180)`.
  - Flatten image feature: `(B, 184320)`.
- Context branch:
  - 입력: normalize된 8개 context feature `(B, 8)`.
  - shared context encoder output: `(B, 32)`.
- Fusion:
  - Concat feature: `(B, 184320 + 32) = (B, 184352)`.
  - Final classifier: `Dropout(0.5) -> Linear(184352, 2)`.
  - Output logits: `(B, 2)`.

Parameter check:
- Stage 2 I60 baseline expected parameters: `2,952,962`.
- Stage 4 concat actual/expected parameters: `2,954,370`.
- Stage 2 baseline 대비 증가량: `+1,408`.
- 증가량이 작은 이유는 context encoder와 final classifier의 32개 추가 입력만
  늘어났기 때문입니다.

Local validation:
- 새 model/checker 파일의 `python -m py_compile` 통과.
- `python scripts/check_stage4_model_shapes.py --config configs/env_local.yaml --model concat --batch-size 2`
  통과.
- Dummy check:
  - `image`: `(2, 1, 96, 180)`
  - `flatten`: `(2, 184320)`
  - `context`: `(2, 8)`
  - `context_embedding`: `(2, 32)`
  - `concat_feature`: `(2, 184352)`
  - `logits`: `(2, 2)`
- Real context check:
  - 4-I2에서 만든 local normalized context row를 읽었습니다:
    `outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_features.csv`.
  - full forward pass에서 finite `(2, 2)` logits가 나왔습니다.

해석:
- 이 모델은 교수님이 말한 ablation 중 `4-A` baseline입니다.
- 시장 context를 visual CNN feature에 단순히 붙이는 것만으로 충분한지 확인합니다.
- 아직 CNN 중간 feature map을 조절하지는 않습니다. 중간 feature modulation은
  `4-I5` gating, `4-I6/4-I7` FiLM에서 시작합니다.
