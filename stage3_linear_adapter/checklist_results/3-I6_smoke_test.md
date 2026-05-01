# 3-I6 Smoke Test

## English

Status: complete

Validation performed:
- `python -m py_compile` passed for Stage 3 package and scripts.
- Shape/parameter checks passed for I5, I20, I60.
- Tiny local run passed:
  - I5/R5/ohlc
  - seed 42
  - 1 epoch
  - train 32, validation 16, test 16 rows
- Output check passed after:
  - checkpoint
  - train history
  - predictions
  - classification metrics
  - trading metrics
  - Linear Grad-CAM

## 한국어

상태: 완료

검증 내용:
- Stage 3 package와 script `python -m py_compile` 통과.
- I5, I20, I60 shape/parameter check 통과.
- 작은 local run 통과:
  - I5/R5/ohlc
  - seed 42
  - 1 epoch
  - train 32, validation 16, test 16 rows
- 다음 output check 통과:
  - checkpoint
  - train history
  - predictions
  - classification metrics
  - trading metrics
  - Linear Grad-CAM

