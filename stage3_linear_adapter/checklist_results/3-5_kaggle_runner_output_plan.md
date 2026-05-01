# 3-5 Kaggle Runner and Output Plan

## English

Status: done

Defined Stage 3 Kaggle execution and output preservation plan.

Key decisions:
- Kaggle remains the default full-run environment.
- Use the Stage 2 one-cell runner pattern:
  copy code snapshot, attach BTC data, run train/eval/trading/Grad-CAM/output
  check, and backup after each step.
- Output root is `outputs/stage3`.
- Backup zips go to `/kaggle/working/stage3_saved_outputs`.
- Planned notebooks:
  - `kaggle_stage3_linear_single_config_one_cell.md`
  - `kaggle_stage3_linear_grid_single_seed_one_cell.md`
  - `kaggle_stage3_results_viewer_one_cell.md`

Output:
- `docs/kaggle_runner_output_plan.md`

## 한국어

상태: 완료

Stage 3 Kaggle 실행과 output 보존 계획을 정의했습니다.

핵심 결정:
- full-run 기본 환경은 Kaggle입니다.
- Stage 2 one-cell runner pattern을 사용합니다:
  code snapshot 복사, BTC data attach, train/eval/trading/Grad-CAM/output check
  실행, 각 단계 뒤 backup.
- Output root는 `outputs/stage3`입니다.
- Backup zip은 `/kaggle/working/stage3_saved_outputs`에 저장합니다.
- 예정 notebook:
  - `kaggle_stage3_linear_single_config_one_cell.md`
  - `kaggle_stage3_linear_grid_single_seed_one_cell.md`
  - `kaggle_stage3_results_viewer_one_cell.md`

산출물:
- `docs/kaggle_runner_output_plan.md`
