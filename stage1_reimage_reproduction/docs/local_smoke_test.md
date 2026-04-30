# 1-I9 Local Smoke Test

## English

Status:
- Completed on 2026-05-01.

Purpose:
- Verify the Stage 1 local code path on tiny data before Kaggle full runs.
- Confirm data loading, label/split/normalization, model forward/backward,
  checkpoint writing, prediction/metric writing, and Grad-CAM generation.

Important:
- This is not a reproduction result.
- The model trains on only 8 rows and validates on only 4 rows.
- Accuracy and Grad-CAM patterns from this run should not be interpreted as
  paper evidence.

Commands executed:
- `python scripts/check_scaffold.py --config configs/env_local.yaml`
- `python scripts/check_data_loading.py --config configs/env_local.yaml --sample-indices 0 -1`
- `python scripts/check_label_split_normalization.py --config configs/env_local.yaml --normalization-max-images 128`
- `python scripts/check_model.py --config configs/env_local.yaml --batch-size 2`
- `python scripts/check_training_loop.py --config configs/env_local.yaml --max-epochs 2 --train-samples 8 --val-samples 4 --batch-size 2`
- `python scripts/run_stage1_baseline.py --config configs/env_local.yaml --run-mode smoke --horizons stage1_i20_r20 --max-train-rows 8 --max-val-rows 4 --normalization-max-images 128 --max-epochs 1`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --max-rows 4`
- `python scripts/generate_stage1_gradcam.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --year 1993 --samples-per-class 1 --write-report-copy`
- `python -m compileall src scripts`

Saved logs:
- `reports/smoke_tests/1-I9_check_scaffold.json`
- `reports/smoke_tests/1-I9_check_data_loading.json`
- `reports/smoke_tests/1-I9_check_label_split_normalization.json`
- `reports/smoke_tests/1-I9_check_model.json`
- `reports/smoke_tests/1-I9_check_training_loop.json`
- `reports/smoke_tests/1-I9_run_stage1_baseline.json`
- `reports/smoke_tests/1-I9_evaluate_validation.json`
- `reports/smoke_tests/1-I9_generate_gradcam.json`
- `reports/smoke_tests/1-I9_compileall.log`

Key smoke outputs:
- `outputs/checkpoints/stage1_i20_r20/seed_42/best.pt`
- `outputs/predictions/stage1_i20_r20/seed_42/validation_predictions.csv`
- `outputs/metrics/stage1_i20_r20/seed_42/validation_metrics.json`
- `outputs/figures/gradcam/stage1_i20_r20/seed_42/validation/figure13_style_1993_validation.png`
- `reports/figures/gradcam/stage1_i20_r20_seed_42_validation_1993_figure13_style.png`

Result:
- Local smoke path passed.
- Validation smoke prediction created 4 rows.
- Grad-CAM smoke figure created 2 samples: 1 Up prediction and 1 Down prediction.

## 한국어

상태:
- 2026-05-01 완료.

목적:
- Kaggle full run 전에 Stage 1 local code path가 작은 데이터에서 끝까지 도는지
  확인했습니다.
- data loading, label/split/normalization, model forward/backward, checkpoint 저장,
  prediction/metric 저장, Grad-CAM 생성을 확인했습니다.

중요:
- 이 결과는 재현 결과가 아닙니다.
- model은 train row 8개, validation row 4개로만 smoke 학습했습니다.
- 여기서 나온 accuracy나 Grad-CAM 패턴은 논문 성능 근거로 해석하면 안 됩니다.

실행한 명령:
- `python scripts/check_scaffold.py --config configs/env_local.yaml`
- `python scripts/check_data_loading.py --config configs/env_local.yaml --sample-indices 0 -1`
- `python scripts/check_label_split_normalization.py --config configs/env_local.yaml --normalization-max-images 128`
- `python scripts/check_model.py --config configs/env_local.yaml --batch-size 2`
- `python scripts/check_training_loop.py --config configs/env_local.yaml --max-epochs 2 --train-samples 8 --val-samples 4 --batch-size 2`
- `python scripts/run_stage1_baseline.py --config configs/env_local.yaml --run-mode smoke --horizons stage1_i20_r20 --max-train-rows 8 --max-val-rows 4 --normalization-max-images 128 --max-epochs 1`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --max-rows 4`
- `python scripts/generate_stage1_gradcam.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --year 1993 --samples-per-class 1 --write-report-copy`
- `python -m compileall src scripts`

저장한 로그:
- `reports/smoke_tests/1-I9_check_scaffold.json`
- `reports/smoke_tests/1-I9_check_data_loading.json`
- `reports/smoke_tests/1-I9_check_label_split_normalization.json`
- `reports/smoke_tests/1-I9_check_model.json`
- `reports/smoke_tests/1-I9_check_training_loop.json`
- `reports/smoke_tests/1-I9_run_stage1_baseline.json`
- `reports/smoke_tests/1-I9_evaluate_validation.json`
- `reports/smoke_tests/1-I9_generate_gradcam.json`
- `reports/smoke_tests/1-I9_compileall.log`

주요 smoke output:
- `outputs/checkpoints/stage1_i20_r20/seed_42/best.pt`
- `outputs/predictions/stage1_i20_r20/seed_42/validation_predictions.csv`
- `outputs/metrics/stage1_i20_r20/seed_42/validation_metrics.json`
- `outputs/figures/gradcam/stage1_i20_r20/seed_42/validation/figure13_style_1993_validation.png`
- `reports/figures/gradcam/stage1_i20_r20_seed_42_validation_1993_figure13_style.png`

결과:
- local smoke path가 통과했습니다.
- validation smoke prediction은 4개 row를 생성했습니다.
- Grad-CAM smoke figure는 Up 예측 1개, Down 예측 1개, 총 2개 sample로 생성했습니다.
