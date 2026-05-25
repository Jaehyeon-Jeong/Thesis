# Stage 4 Kaggle Cells

## English

The first Stage 4 Kaggle one-cell runner is now available for the four main
numeric-context ablations.

Available notebooks:
- `kaggle_stage4_four_ablation_single_seed_one_cell.md`
  - Runs `I60/R20/ohlc_ma_vb`, context window `60`, seed `42`.
  - Runs `concat`, `gating`, `film_gamma`, and `film_full`.
  - Backups after context build, training, evaluation, Grad-CAM, output check,
    and summary.

Planned notebooks:
- `kaggle_stage4_single_ablation_one_cell.md`
- `kaggle_stage4_four_ablation_five_seed_one_cell.md`
- `kaggle_stage4_results_viewer_one_cell.md`

Required backup root:
- `/kaggle/working/stage4_saved_outputs`

## 한국어

첫 Stage 4 Kaggle one-cell runner가 네 가지 numeric-context ablation용으로
준비됐습니다.

사용 가능 notebook:
- `kaggle_stage4_four_ablation_single_seed_one_cell.md`
  - `I60/R20/ohlc_ma_vb`, context window `60`, seed `42` 실행.
  - `concat`, `gating`, `film_gamma`, `film_full` 실행.
  - context build, training, evaluation, Grad-CAM, output check, summary 이후
    backup zip을 저장합니다.

예정 notebook:
- `kaggle_stage4_single_ablation_one_cell.md`
- `kaggle_stage4_four_ablation_five_seed_one_cell.md`
- `kaggle_stage4_results_viewer_one_cell.md`

필수 backup root:
- `/kaggle/working/stage4_saved_outputs`
