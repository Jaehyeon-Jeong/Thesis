# 1-I9 Local Smoke Test Result

## English

Status:
- Completed on 2026-05-01.

Summary:
- Ran the full local smoke path through data checks, model checks, synthetic
  training-loop checks, tiny Stage 1 baseline training, validation prediction
  export, and Grad-CAM generation.

Smoke limits:
- `stage1_i20_r20` only.
- 8 train rows.
- 4 validation rows.
- 1 epoch.
- 1 Up and 1 Down Grad-CAM sample from validation year 1993.

Result:
- All smoke commands completed successfully.
- Outputs are non-reproduction artifacts.

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

Key output:
- `reports/figures/gradcam/stage1_i20_r20_seed_42_validation_1993_figure13_style.png`

Next gate:
- `1-I10. Kaggle full single-seed run`.

## 한국어

상태:
- 2026-05-01 완료.

요약:
- data check, model check, synthetic training-loop check, 작은 Stage 1 baseline
  학습, validation prediction export, Grad-CAM 생성까지 local smoke path를
  끝까지 실행했습니다.

Smoke 제한:
- `stage1_i20_r20`만 사용.
- train row 8개.
- validation row 4개.
- epoch 1회.
- validation year 1993에서 Up 1개, Down 1개 Grad-CAM sample.

결과:
- 모든 smoke 명령이 성공했습니다.
- 산출물은 재현 결과가 아니라 local smoke artifact입니다.

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

주요 output:
- `reports/figures/gradcam/stage1_i20_r20_seed_42_validation_1993_figure13_style.png`

다음 gate:
- `1-I10. Kaggle full single-seed run`.
