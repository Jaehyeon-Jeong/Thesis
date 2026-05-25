# 4-I5 Context Gating Model

## English

Status: complete.

Implemented the second Stage 4 ablation model: `CNN + context gating`.

What was added:
- Extended `src/stage4_film/models/context_stock_cnn.py`.
- Extended `scripts/check_stage4_model_shapes.py` to support `--model gating`.
- Updated `src/stage4_film/models/__init__.py` exports.

Model structure:
- The Stage 2 Stock_CNN convolution blocks are reused unchanged.
- The classifier input dimension stays the same as Stage 2.
- Image branch:
  - Input: `(B, 1, 96, 180)` for primary `I60`.
  - Final CNN feature map: `(B, 512, 2, 180)`.
- Context branch:
  - Input: normalized 8-feature context vector `(B, 8)`.
  - Shared context encoder output: `(B, 32)`.
  - Gate head: `Linear(32, 512)`.
- Gating:
  - Raw gate: `(B, 512)`.
  - Gate formula: `gate = 2 * sigmoid(raw_gate)`.
  - Gate is reshaped to `(B, 512, 1, 1)` and multiplied into the final feature
    map.
  - Gated feature map: `(B, 512, 2, 180)`.
  - Flattened feature: `(B, 184320)`.
  - Final classifier: `Dropout(0.5) -> Linear(184320, 2)`.
  - Output logits: `(B, 2)`.

What the gate does:
- The final I60 CNN feature map has `512` channels.
- Each channel can be understood as one learned visual pattern detector.
- The gate gives one scalar multiplier per channel and per sample:
  `gate[:, channel]`.
- The model applies it as:
  `gated_feature[:, c, :, :] = feature[:, c, :, :] * gate[:, c]`.
- If a gate value is above `1`, that feature channel is amplified.
- If a gate value is below `1`, that feature channel is suppressed.
- If a gate value is exactly `1`, that channel is unchanged.
- Therefore gating tests whether the same chart pattern should be weighted
  differently under different market context states.

Why `gate = 2 * sigmoid(raw_gate)`:
- A sigmoid alone returns values in `(0, 1)`.
- If the gate head is zero-initialized, then `raw_gate = 0`.
- `sigmoid(0) = 0.5`, so using sigmoid alone would shrink every CNN channel by
  half at initialization.
- Multiplying by `2` changes the neutral point:
  `2 * sigmoid(0) = 1`.
- This makes the initial model an identity modulation of the Stage 2 feature
  map, so training starts from the baseline feature path instead of damaging it.
- During training:
  - `raw_gate < 0` gives `gate < 1`, suppressing the channel.
  - `raw_gate = 0` gives `gate = 1`, keeping the channel.
  - `raw_gate > 0` gives `gate > 1`, amplifying the channel.

Identity initialization:
- `gate_head.weight` and `gate_head.bias` are zero-initialized.
- Therefore the first forward pass starts from `raw_gate = 0` and
  `gate = 2 * sigmoid(0) = 1`.
- This means the model starts from the Stage 2 image feature path and learns
  deviations from it.

Parameter check:
- Stage 2 I60 baseline expected parameters: `2,952,962`.
- Stage 4 gating actual/expected parameters: `2,971,202`.
- Delta vs Stage 2 baseline: `+18,240`.
- Added parameters are the context encoder (`1,344`) plus the gate head
  (`32 * 512 + 512 = 16,896`).

Local validation:
- `python -m py_compile` passed.
- `python scripts/check_stage4_model_shapes.py --config configs/env_local.yaml --model concat --batch-size 2`
  still passed.
- `python scripts/check_stage4_model_shapes.py --config configs/env_local.yaml --model gating --batch-size 2`
  passed.
- Gating dummy check:
  - `final_feature_map`: `(2, 512, 2, 180)`
  - `context_embedding`: `(2, 32)`
  - `raw_gate`: `(2, 512)`
  - `gate`: `(2, 512)`
  - `gated_feature_map`: `(2, 512, 2, 180)`
  - `flatten`: `(2, 184320)`
  - `logits`: `(2, 2)`
  - initial gate min/max: `1.0 / 1.0`
- Real context check:
  - Read local 4-I2 normalized context rows from
    `outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_features.csv`.
  - Full forward pass produced finite `(2, 2)` logits.
  - Real-context initial gate min/max: `1.0 / 1.0`.

Interpretation:
- This is the `4-B` baseline for the advisor-requested ablation.
- It tests whether context should modulate the visual feature map
  multiplicatively, rather than merely being appended before the classifier.
- It is still simpler than FiLM because it only gates the final feature map and
  has no additive `beta` term.

## 한국어

상태: 완료.

두 번째 Stage 4 ablation 모델인 `CNN + context gating`을 구현했습니다.

추가/수정한 파일:
- `src/stage4_film/models/context_stock_cnn.py` 확장.
- `scripts/check_stage4_model_shapes.py`에 `--model gating` 지원 추가.
- `src/stage4_film/models/__init__.py` export 업데이트.

모델 구조:
- Stage 2 Stock_CNN convolution block은 그대로 재사용합니다.
- classifier 입력 차원은 Stage 2와 동일하게 유지합니다.
- Image branch:
  - 입력: primary `I60` 기준 `(B, 1, 96, 180)`.
  - 마지막 CNN feature map: `(B, 512, 2, 180)`.
- Context branch:
  - 입력: normalize된 8개 context feature `(B, 8)`.
  - shared context encoder output: `(B, 32)`.
  - Gate head: `Linear(32, 512)`.
- Gating:
  - Raw gate: `(B, 512)`.
  - Gate formula: `gate = 2 * sigmoid(raw_gate)`.
  - Gate를 `(B, 512, 1, 1)`로 reshape해서 마지막 feature map에 곱합니다.
  - Gated feature map: `(B, 512, 2, 180)`.
  - Flattened feature: `(B, 184320)`.
  - Final classifier: `Dropout(0.5) -> Linear(184320, 2)`.
  - Output logits: `(B, 2)`.

gate가 하는 일:
- 마지막 I60 CNN feature map에는 `512`개 channel이 있습니다.
- 각 channel은 CNN이 학습한 하나의 visual pattern detector로 볼 수 있습니다.
- gate는 sample마다, channel마다 scalar multiplier 하나를 줍니다:
  `gate[:, channel]`.
- 적용 방식은 다음과 같습니다:
  `gated_feature[:, c, :, :] = feature[:, c, :, :] * gate[:, c]`.
- gate 값이 `1`보다 크면 해당 feature channel을 키웁니다.
- gate 값이 `1`보다 작으면 해당 feature channel을 줄입니다.
- gate 값이 정확히 `1`이면 해당 channel은 그대로 둡니다.
- 따라서 gating은 같은 chart pattern이라도 market context 상태에 따라 다르게
  가중해야 하는지 테스트합니다.

`gate = 2 * sigmoid(raw_gate)`를 쓰는 이유:
- sigmoid만 쓰면 출력 범위는 `(0, 1)`입니다.
- gate head를 0으로 초기화하면 `raw_gate = 0`입니다.
- `sigmoid(0) = 0.5`이므로 sigmoid만 gate로 쓰면 초기 상태에서 모든 CNN channel을
  절반으로 줄여버립니다.
- 그래서 `2`를 곱해 neutral point를 바꿉니다:
  `2 * sigmoid(0) = 1`.
- 이렇게 하면 초기 모델은 Stage 2 feature map을 그대로 통과시키는 identity
  modulation에서 시작합니다.
- 학습 중에는:
  - `raw_gate < 0`이면 `gate < 1`, channel을 약화합니다.
  - `raw_gate = 0`이면 `gate = 1`, channel을 유지합니다.
  - `raw_gate > 0`이면 `gate > 1`, channel을 강화합니다.

Identity initialization:
- `gate_head.weight`와 `gate_head.bias`는 0으로 초기화합니다.
- 그래서 첫 forward pass는 `raw_gate = 0`,
  `gate = 2 * sigmoid(0) = 1`에서 시작합니다.
- 즉 처음에는 Stage 2 image feature path를 그대로 통과시키고, 학습하면서
  context에 따른 channel 조절을 배우게 됩니다.

Parameter check:
- Stage 2 I60 baseline expected parameters: `2,952,962`.
- Stage 4 gating actual/expected parameters: `2,971,202`.
- Stage 2 baseline 대비 증가량: `+18,240`.
- 추가 parameter는 context encoder (`1,344`)와 gate head
  (`32 * 512 + 512 = 16,896`)입니다.

Local validation:
- `python -m py_compile` 통과.
- `python scripts/check_stage4_model_shapes.py --config configs/env_local.yaml --model concat --batch-size 2`
  재확인 통과.
- `python scripts/check_stage4_model_shapes.py --config configs/env_local.yaml --model gating --batch-size 2`
  통과.
- Gating dummy check:
  - `final_feature_map`: `(2, 512, 2, 180)`
  - `context_embedding`: `(2, 32)`
  - `raw_gate`: `(2, 512)`
  - `gate`: `(2, 512)`
  - `gated_feature_map`: `(2, 512, 2, 180)`
  - `flatten`: `(2, 184320)`
  - `logits`: `(2, 2)`
  - initial gate min/max: `1.0 / 1.0`
- Real context check:
  - 4-I2에서 만든 local normalized context row를 읽었습니다:
    `outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_features.csv`.
  - full forward pass에서 finite `(2, 2)` logits가 나왔습니다.
  - real-context initial gate min/max: `1.0 / 1.0`.

해석:
- 이 모델은 교수님이 말한 ablation 중 `4-B` baseline입니다.
- context를 classifier 앞에 단순히 붙이는 대신, visual feature map을
  multiplicative하게 조절하는 것이 도움이 되는지 확인합니다.
- 다만 아직 FiLM보다 단순합니다. 마지막 feature map에만 gate를 곱하고, additive
  `beta` term은 없습니다.
