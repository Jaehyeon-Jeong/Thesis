# 3-3 Training and Evaluation Comparison Plan

## English

Status: done

Defined the Stage 3 training/evaluation comparison plan.

Key decisions:
- Primary comparison is a single-seed grid with seed `42`.
- Grid size is `36` Linear runs:
  `I5/I20/I60 x R5/R20/R60 x ohlc/ohlc_vb/ohlc_ma/ohlc_ma_vb`.
- Training settings inherit Stage 2 defaults:
  batch `128`, Adam, learning rate `1e-5`, CrossEntropyLoss, early stopping on
  `val_loss` with patience `2`.
- Comparison tables join Stage 2 baseline and Stage 3 Linear by image window,
  return horizon, image spec, and seed.

Output:
- `docs/training_evaluation_comparison_plan.md`

## 한국어

상태: 완료

Stage 3 training/evaluation 비교 계획을 정의했습니다.

핵심 결정:
- Primary comparison은 seed `42` single-seed grid입니다.
- Grid size는 Linear run `36`개입니다:
  `I5/I20/I60 x R5/R20/R60 x ohlc/ohlc_vb/ohlc_ma/ohlc_ma_vb`.
- Training setting은 Stage 2 기본값을 상속합니다:
  batch `128`, Adam, learning rate `1e-5`, CrossEntropyLoss, `val_loss`
  기준 early stopping patience `2`.
- 비교표는 image window, return horizon, image spec, seed 기준으로 Stage 2
  baseline과 Stage 3 Linear를 join합니다.

산출물:
- `docs/training_evaluation_comparison_plan.md`
