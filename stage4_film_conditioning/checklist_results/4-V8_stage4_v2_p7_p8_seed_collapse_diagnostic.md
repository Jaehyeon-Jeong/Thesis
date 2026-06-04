# 4-V8 Stage 4 v2 P7/P8 Seed-Collapse Diagnostic

## English

Status: completed; use V8 as the diagnostic bridge from F&G-only numeric FiLM
to the V9 scale check and then to the news-context track.

Purpose:
- Diagnose seed-dependent prediction collapse before running another FiLM
  gamma/beta grid.
- Compare the two relevant F&G-only experiments:
  - P7: `I60/R20/ohlc_ma_vb + F&G-only + film_full`
  - P8: `I60/R20/ohlc_ma_vb + F&G-only + film_full_bounded_last_block`
- Explain why seeds `43` and `44` collapse in opposite directions:
  - P7 collapses toward mostly Up predictions.
  - P8 collapses toward mostly Down predictions.

Why this comes before a scale grid:
- P8 improved ranking-style metrics over P7:
  - P7 ROC-AUC mean: `0.5465`
  - P8 ROC-AUC mean: `0.5763`
  - P7 average precision mean: `0.5739`
  - P8 average precision mean: `0.6033`
- But P8 reduced accuracy/F1 because default `prob_up >= 0.5` class decisions
  collapsed for seeds `43` and `44`.
- This suggests the next question is not simply “which gamma/beta scale works?”
  but whether the model has ranking signal that is being lost through
  thresholding, checkpoint selection, or class-bias drift.

Added script:
- `scripts/analyze_stage4_seed_collapse.py`

Script behavior:
- Reads existing Stage 4 prediction CSVs for P7/P8.
- Computes default-threshold metrics for validation and test.
- Flags class-collapse from predicted positive rate:
  - mostly Down: predicted positive rate `<= 0.15`
  - mostly Up: predicted positive rate `>= 0.85`
- Selects a validation threshold by balanced accuracy.
- Applies the selected threshold to test predictions.
- Writes P7/P8 paired comparisons by seed and split.

Kaggle wrapper:
- `notebooks/kaggle_stage4_v2_v8_p7_p8_seed_collapse_diagnostic_one_cell.md`

Generated outputs:
- `reports/tables/stage4_v2_v8_p7_p8_seed_collapse_default_metrics.csv`
- `reports/tables/stage4_v2_v8_p7_p8_seed_collapse_probability_quantiles.csv`
- `reports/tables/stage4_v2_v8_p7_p8_seed_collapse_threshold_calibration.csv`
- `reports/tables/stage4_v2_v8_p7_p8_seed_collapse_pairwise_comparison.csv`
- `reports/tables/stage4_v2_v8_p7_p8_seed_collapse_report.md`

Result:
- P8 improved ranking signal versus P7, but class decision collapse remained.
- Validation-threshold calibration did not robustly solve seed `43`/`44`.
- This justified a small targeted V9 scale grid (`0.02`, `0.05`, `0.10`) rather
  than a broad random search.

## 한국어

상태: 완료. V8은 F&G-only numeric FiLM에서 V9 scale check, 이후 news-context
track으로 넘어가기 위한 진단 단계입니다.

목적:
- FiLM gamma/beta scale grid를 다시 돌리기 전에 seed-dependent prediction
  collapse 원인을 진단합니다.
- 비교 대상은 F&G-only 실험 두 개입니다:
  - P7: `I60/R20/ohlc_ma_vb + F&G-only + film_full`
  - P8: `I60/R20/ohlc_ma_vb + F&G-only + film_full_bounded_last_block`
- seed `43`, `44`가 서로 반대 방향으로 무너지는 이유를 봅니다:
  - P7은 대부분 Up 예측으로 무너졌습니다.
  - P8은 대부분 Down 예측으로 무너졌습니다.

왜 scale grid보다 먼저인가:
- P8은 P7보다 ranking 계열 metric이 좋아졌습니다:
  - P7 ROC-AUC mean: `0.5465`
  - P8 ROC-AUC mean: `0.5763`
  - P7 average precision mean: `0.5739`
  - P8 average precision mean: `0.6033`
- 하지만 P8은 seed `43`, `44`에서 기본 `prob_up >= 0.5` class decision이
  무너지면서 accuracy/F1이 낮아졌습니다.
- 따라서 다음 질문은 단순히 “gamma/beta scale을 얼마로 할까?”가 아니라,
  ranking signal이 threshold, checkpoint selection, class-bias drift 때문에
  class decision에서 손실되는지 확인하는 것입니다.

추가 script:
- `scripts/analyze_stage4_seed_collapse.py`

Script 동작:
- 기존 Stage 4 P7/P8 prediction CSV를 읽습니다.
- validation/test의 default-threshold metric을 계산합니다.
- predicted positive rate 기준으로 class collapse를 표시합니다:
  - mostly Down: predicted positive rate `<= 0.15`
  - mostly Up: predicted positive rate `>= 0.85`
- validation balanced accuracy 기준 threshold를 선택합니다.
- 선택된 threshold를 test prediction에 적용합니다.
- seed/split별 P7/P8 paired comparison을 저장합니다.

Kaggle wrapper:
- `notebooks/kaggle_stage4_v2_v8_p7_p8_seed_collapse_diagnostic_one_cell.md`

생성된 출력:
- `reports/tables/stage4_v2_v8_p7_p8_seed_collapse_default_metrics.csv`
- `reports/tables/stage4_v2_v8_p7_p8_seed_collapse_probability_quantiles.csv`
- `reports/tables/stage4_v2_v8_p7_p8_seed_collapse_threshold_calibration.csv`
- `reports/tables/stage4_v2_v8_p7_p8_seed_collapse_pairwise_comparison.csv`
- `reports/tables/stage4_v2_v8_p7_p8_seed_collapse_report.md`

결과:
- P8은 P7보다 ranking signal을 개선했지만 class decision collapse는 남았습니다.
- Validation-threshold calibration은 seed `43`, `44`를 robust하게 해결하지
  못했습니다.
- 그래서 넓은 random search가 아니라 `0.02`, `0.05`, `0.10`의 작은 V9 targeted
  scale grid를 실행하는 것으로 결정했습니다.
