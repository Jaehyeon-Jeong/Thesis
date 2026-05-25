# 4-I3 Context MLP Encoder

## English

Status: complete

Purpose:
- Convert the normalized 8-dimensional market-context vector from `4-I2` into
  a shared 32-dimensional condition embedding.
- Reuse this same encoder for the upcoming concat, gating, gamma-only FiLM, and
  full FiLM models.

Implemented files:
- `src/stage4_film/conditions/__init__.py`
- `src/stage4_film/conditions/context_encoder.py`
- `scripts/check_stage4_context_encoder.py`

Architecture:

```text
context vector (B, 8)
  -> Linear(8, 32)
  -> ReLU
  -> Dropout(0.10)
  -> Linear(32, 32)
  -> ReLU
  -> condition embedding (B, 32)
```

Parameter count:
- Total parameters: `1,344`.
- Trainable parameters: `1,344`.

Input columns:
- `fg_value_normalized`
- `fg_mean_60_normalized`
- `fg_delta_60_normalized`
- `fg_std_60_normalized`
- `bb_percent_b_60_normalized`
- `bb_bandwidth_60_normalized`
- `mfi_60_normalized`
- `rv_60_normalized`

Validation:
- `python -m py_compile` passed for the condition package and check script.
- `python scripts/check_stage4_context_encoder.py --config configs/env_local.yaml`
  passed.
- Dummy tensor check:
  - Input: `(4, 8)`
  - Output: `(4, 32)`
  - Finite output: `true`
- Real context row check using the local `4-I2` context table:
  - Rows checked: `4`
  - Input: `(4, 8)`
  - Output: `(4, 32)`
  - Finite output: `true`

Design note:
- The encoder does not see the chart image.
- It only turns structured context values into a condition embedding.
- Later model heads decide how to use that embedding:
  - `4-A`: concatenate after CNN flatten.
  - `4-B`: create a final-block gate.
  - `4-C`: create FiLM gamma values.
  - `4-D`: create FiLM gamma and beta values.

Next:
- `4-I4`: implement `CNN + context concat`.

## 한국어

상태: 완료

목적:
- `4-I2`에서 만든 normalized 8차원 market-context vector를 공통 32차원 condition
  embedding으로 변환합니다.
- 이 encoder는 이후 concat, gating, gamma-only FiLM, full FiLM 모델이 공통으로
  사용합니다.

구현 파일:
- `src/stage4_film/conditions/__init__.py`
- `src/stage4_film/conditions/context_encoder.py`
- `scripts/check_stage4_context_encoder.py`

구조:

```text
context vector (B, 8)
  -> Linear(8, 32)
  -> ReLU
  -> Dropout(0.10)
  -> Linear(32, 32)
  -> ReLU
  -> condition embedding (B, 32)
```

파라미터 수:
- Total parameters: `1,344`.
- Trainable parameters: `1,344`.

입력 column:
- `fg_value_normalized`
- `fg_mean_60_normalized`
- `fg_delta_60_normalized`
- `fg_std_60_normalized`
- `bb_percent_b_60_normalized`
- `bb_bandwidth_60_normalized`
- `mfi_60_normalized`
- `rv_60_normalized`

검증:
- condition package와 check script의 `python -m py_compile` 통과.
- `python scripts/check_stage4_context_encoder.py --config configs/env_local.yaml`
  통과.
- Dummy tensor check:
  - Input: `(4, 8)`
  - Output: `(4, 32)`
  - Finite output: `true`
- Local `4-I2` context table의 실제 context row check:
  - Rows checked: `4`
  - Input: `(4, 8)`
  - Output: `(4, 32)`
  - Finite output: `true`

설계 note:
- 이 encoder는 chart image를 보지 않습니다.
- structured context 값을 condition embedding으로 바꾸는 역할만 합니다.
- 이후 모델 head가 이 embedding을 어떻게 사용할지 결정합니다:
  - `4-A`: CNN flatten 뒤 concatenate.
  - `4-B`: final-block gate 생성.
  - `4-C`: FiLM gamma 생성.
  - `4-D`: FiLM gamma/beta 생성.

다음:
- `4-I4`: `CNN + context concat` 구현.
