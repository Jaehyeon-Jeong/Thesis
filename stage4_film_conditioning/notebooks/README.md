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
- `kaggle_stage4_four_ablation_five_seed_one_cell.md`
  - Runs the same selected Stage 4 configuration for seeds
    `42, 43, 44, 45, 46`.
  - Total: `4` context methods x `5` seeds = `20` runs.
  - Uses `MIN_PREDICTIONS=1000` so old smoke-test artifacts cannot be treated
    as completed full runs.
  - Uses `SAVE_BACKUP_ZIPS=False` by default to avoid filling Kaggle disk.
  - If a previous run stopped from disk pressure, set
    `RESUME_EXISTING_PROJECT=True` before rerunning.
- `kaggle_stage4_v1_interpretation_report_one_cell.md`
  - Run below the five-seed output cell.
  - Reads five-seed summary tables, predictions, Grad-CAM samples, and
    modulation exports.
  - Writes a v1 diagnostic interpretation report and a zip package for
    download.

Planned notebooks:
- `kaggle_stage4_single_ablation_one_cell.md`
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
- `kaggle_stage4_four_ablation_five_seed_one_cell.md`
  - 같은 Stage 4 selected configuration을 seeds `42, 43, 44, 45, 46`으로
    실행합니다.
  - 총 `4` context methods x `5` seeds = `20` runs입니다.
  - `MIN_PREDICTIONS=1000`을 사용해서 과거 smoke-test artifact가 full run
    완료로 처리되지 않게 합니다.
  - Kaggle disk가 차지 않도록 `SAVE_BACKUP_ZIPS=False`가 기본값입니다.
  - 이전 run이 disk 문제로 중단됐으면 `RESUME_EXISTING_PROJECT=True`로 바꾸고
    다시 실행합니다.
- `kaggle_stage4_v1_interpretation_report_one_cell.md`
  - five-seed output cell 아래에서 실행합니다.
  - five-seed summary table, prediction, Grad-CAM sample, modulation export를
    읽습니다.
  - v1 diagnostic interpretation report와 download용 zip package를 저장합니다.

예정 notebook:
- `kaggle_stage4_single_ablation_one_cell.md`
- `kaggle_stage4_results_viewer_one_cell.md`

필수 backup root:
- `/kaggle/working/stage4_saved_outputs`
