# 1-I7 Evaluation and Prediction-output Implementation

## English

Status:
- Completed on 2026-05-01.

What changed:
- Added the Stage 1 evaluation package.
- Added checkpoint-to-prediction export.
- Added classification and prediction-return correlation metric writers.
- Added seed-probability averaging for full paper-style runs.
- Added local/Kaggle `evaluation` config sections.

Key files:
- `src/stage1_reimage/evaluation/prediction.py`
- `src/stage1_reimage/evaluation/__init__.py`
- `scripts/evaluate_stage1_predictions.py`
- `docs/evaluation_prediction_implementation.md`
- `configs/env_local.yaml`
- `configs/env_kaggle.yaml`

Validation:
- `python -m compileall src scripts`
- `python scripts/check_scaffold.py --config configs/env_local.yaml`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --max-rows 4`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --split validation --average-seed-predictions 42`

Smoke result:
- Horizon: `stage1_i20_r20`
- Split: `validation`
- Rows: `4`
- Accuracy: `0.25`
- Positive rate: `0.50`
- Predicted positive rate: `0.25`

Notes:
- Smoke metrics are not reproduction results.
- `outputs/` prediction and metric files are excluded from GitHub.
- Final `reports/tables/` assembly remains a later report step after full
  Kaggle outputs exist.
- Portfolio/decile H-L metrics remain deferred until the exact paper-style
  convention is rechecked.

## 한국어

상태:
- 2026-05-01 완료.

변경 내용:
- Stage 1 evaluation package를 추가했습니다.
- checkpoint에서 prediction CSV를 export하는 코드를 추가했습니다.
- classification metric과 prediction-return correlation metric writer를
  추가했습니다.
- full paper-style run을 위한 seed probability averaging을 추가했습니다.
- local/Kaggle `evaluation` config section을 추가했습니다.

주요 파일:
- `src/stage1_reimage/evaluation/prediction.py`
- `src/stage1_reimage/evaluation/__init__.py`
- `scripts/evaluate_stage1_predictions.py`
- `docs/evaluation_prediction_implementation.md`
- `configs/env_local.yaml`
- `configs/env_kaggle.yaml`

검증:
- `python -m compileall src scripts`
- `python scripts/check_scaffold.py --config configs/env_local.yaml`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --max-rows 4`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --split validation --average-seed-predictions 42`

Smoke 결과:
- Horizon: `stage1_i20_r20`
- Split: `validation`
- Rows: `4`
- Accuracy: `0.25`
- Positive rate: `0.50`
- Predicted positive rate: `0.25`

메모:
- smoke metric은 reproduction result가 아닙니다.
- `outputs/` 아래 prediction과 metric file은 GitHub에서 제외합니다.
- 최종 `reports/tables/` 조립은 full Kaggle output 이후 report 단계에서
  진행합니다.
- portfolio/decile H-L metric은 정확한 논문식 convention을 재확인하기 전까지
  보류합니다.
