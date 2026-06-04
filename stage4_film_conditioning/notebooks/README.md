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
- `kaggle_stage4_v2_p1_visual_only_same_split_one_cell.md`
  - Stage 4 v2 priority 1 control.
  - Runs `I60/R20/ohlc_ma_vb` with no context using the Stage 2 visual-only
    runner.
  - Purpose: determine whether Stage 4 v1 underperformance comes from
    context/FiLM or from the selected sample/split/run conditions.
- `kaggle_stage4_v2_p2_ohlc_visual_only_one_cell.md`
  - Stage 4 v2 priority 2 control.
  - Runs `I60/R20/ohlc` with no context using the Stage 2 visual-only runner.
  - Purpose: measure the plain-OHLC visual baseline before re-adding structured
    context through FiLM in `4-V2`.
- `kaggle_stage4_v2_p3_ohlc_all_context_film_full_one_cell.md`
  - Stage 4 v2 priority 3 diagnostic.
  - Runs `I60/R20/ohlc` with all structured context and `film_full`.
  - Purpose: test whether F&G/BB/MFI/RV context can recover information after
    MA/VB are removed from the image.
- `kaggle_stage4_v2_p4_ohlc_fg_only_film_full_five_seed_one_cell.md`
  - Stage 4 v2 priority 4 diagnostic.
  - Runs `I60/R20/ohlc` with F&G-only context and `film_full` over five seeds.
  - Purpose: isolate external sentiment/regime context from OHLCV-derived
    technical context.
- `kaggle_stage4_v2_p5_ohlc_technical_only_film_full_five_seed_one_cell.md`
  - Stage 4 v2 priority 5 diagnostic.
  - Runs `I60/R20/ohlc` with technical-only context and `film_full` over five
    seeds.
  - Purpose: test whether BB/MFI/RV recover information when MA/VB are removed
    from the image.
- `kaggle_stage4_v2_p6_ohlc_all_context_film_full_five_seed_one_cell.md`
  - Stage 4 v2 priority 6 diagnostic.
  - Runs `I60/R20/ohlc` with all structured context and `film_full` over five
    seeds.
  - Purpose: test whether the earlier seed-42 all-context recovery is robust.
- `kaggle_stage4_v2_p7_ohlc_ma_vb_fg_only_film_full_five_seed_one_cell.md`
  - Stage 4 v2 priority 7 diagnostic.
  - Runs `I60/R20/ohlc_ma_vb` with F&G-only context and `film_full` over five
    seeds.
  - Purpose: test whether external F&G context improves the strongest visual
    baseline.
- `kaggle_stage4_v2_p8_ohlc_ma_vb_fg_only_bounded_last_block_film_five_seed_one_cell.md`
  - Stage 4 v2 priority 8 diagnostic.
  - Runs `I60/R20/ohlc_ma_vb` with F&G-only context and
    `film_full_bounded_last_block` over five seeds.
  - Purpose: preserve the strongest visual baseline by applying bounded FiLM
    only to the final/high-level CNN block.
- `kaggle_stage4_v2_v8_p7_p8_seed_collapse_diagnostic_one_cell.md`
  - Stage 4 v2 priority 9 diagnostic.
  - Does not train new models.
  - Reads P7/P8 checkpoints and prediction CSVs, exports missing validation/test
    predictions, then runs seed-collapse and validation-threshold calibration
    analysis.
  - Purpose: explain why P7 seeds `43`/`44` collapse mostly Up while P8 seeds
    `43`/`44` collapse mostly Down before another FiLM scale grid.
- `kaggle_stage4_v2_v9_bounded_last_block_film_scale_grid_one_cell.md`
  - Stage 4 v2 priority 10 diagnostic.
  - Runs `I60/R20/ohlc_ma_vb` with F&G-only context and
    `film_full_bounded_last_block`.
  - Grid: scales `0.02`, `0.05`, `0.10` x seeds
    `42, 43, 44, 45, 46`.
  - Purpose: keep the architecture fixed and test whether bounded FiLM scale
    can reduce seed collapse before moving to news context.

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
- `kaggle_stage4_v2_p1_visual_only_same_split_one_cell.md`
  - Stage 4 v2 우선순위 1 control입니다.
  - Stage 2 visual-only runner로 `I60/R20/ohlc_ma_vb`, context 없음 실험을
    실행합니다.
  - 목적: Stage 4 v1 성능 하락이 context/FiLM 때문인지, 선택된 sample/split/run
    조건 때문인지 분리합니다.
- `kaggle_stage4_v2_p2_ohlc_visual_only_one_cell.md`
  - Stage 4 v2 우선순위 2 control입니다.
  - Stage 2 visual-only runner로 `I60/R20/ohlc`, context 없음 실험을 실행합니다.
  - 목적: `4-V2`에서 structured context를 FiLM으로 다시 넣기 전에 plain-OHLC
    visual baseline을 확인합니다.
- `kaggle_stage4_v2_p3_ohlc_all_context_film_full_one_cell.md`
  - Stage 4 v2 우선순위 3 diagnostic입니다.
  - `I60/R20/ohlc`에 전체 structured context와 `film_full`을 붙여 실행합니다.
  - 목적: 이미지에서 MA/VB를 제거한 뒤 F&G/BB/MFI/RV context가 정보를 보완할
    수 있는지 확인합니다.
- `kaggle_stage4_v2_p4_ohlc_fg_only_film_full_five_seed_one_cell.md`
  - Stage 4 v2 우선순위 4 diagnostic입니다.
  - `I60/R20/ohlc`에 F&G-only context와 `film_full`을 붙여 five-seed로 실행합니다.
  - 목적: OHLCV-derived technical context와 분리해서 외부 sentiment/regime
    context 효과만 확인합니다.
- `kaggle_stage4_v2_p5_ohlc_technical_only_film_full_five_seed_one_cell.md`
  - Stage 4 v2 우선순위 5 diagnostic입니다.
  - `I60/R20/ohlc`에 technical-only context와 `film_full`을 붙여 five-seed로
    실행합니다.
  - 목적: 이미지에서 MA/VB를 제거했을 때 BB/MFI/RV가 정보를 회복하는지
    확인합니다.
- `kaggle_stage4_v2_p6_ohlc_all_context_film_full_five_seed_one_cell.md`
  - Stage 4 v2 우선순위 6 diagnostic입니다.
  - `I60/R20/ohlc`에 all structured context와 `film_full`을 붙여 five-seed로
    실행합니다.
  - 목적: 이전 seed-42 all-context 회복이 robust한지 확인합니다.
- `kaggle_stage4_v2_p7_ohlc_ma_vb_fg_only_film_full_five_seed_one_cell.md`
  - Stage 4 v2 우선순위 7 diagnostic입니다.
  - `I60/R20/ohlc_ma_vb`에 F&G-only context와 `film_full`을 붙여 five-seed로
    실행합니다.
  - 목적: 외부 F&G context가 가장 강한 visual baseline을 개선하는지
    확인합니다.
- `kaggle_stage4_v2_p8_ohlc_ma_vb_fg_only_bounded_last_block_film_five_seed_one_cell.md`
  - Stage 4 v2 우선순위 8 diagnostic입니다.
  - `I60/R20/ohlc_ma_vb`에 F&G-only context와
    `film_full_bounded_last_block`을 붙여 five-seed로 실행합니다.
  - 목적: 가장 강한 visual baseline을 보존하기 위해 마지막/high-level CNN
    block에만 bounded FiLM을 적용합니다.
- `kaggle_stage4_v2_v8_p7_p8_seed_collapse_diagnostic_one_cell.md`
  - Stage 4 v2 우선순위 9 diagnostic입니다.
  - 새 model을 학습하지 않습니다.
  - P7/P8 checkpoint와 prediction CSV를 읽고, 필요한 validation/test prediction을
    export한 뒤 seed-collapse와 validation-threshold calibration 분석을 실행합니다.
  - 목적: 또 다른 FiLM scale grid 전에 P7 seed `43`/`44`는 대부분 Up으로,
    P8 seed `43`/`44`는 대부분 Down으로 무너지는 이유를 설명합니다.
- `kaggle_stage4_v2_v9_bounded_last_block_film_scale_grid_one_cell.md`
  - Stage 4 v2 우선순위 10 diagnostic입니다.
  - `I60/R20/ohlc_ma_vb`에 F&G-only context와
    `film_full_bounded_last_block`을 붙입니다.
  - Grid: scales `0.02`, `0.05`, `0.10` x seeds
    `42, 43, 44, 45, 46`.
  - 목적: architecture를 고정하고 bounded FiLM scale만 바꿔 seed collapse가
    줄어드는지 확인한 뒤, 안 되면 news context로 넘어갑니다.

예정 notebook:
- `kaggle_stage4_single_ablation_one_cell.md`
- `kaggle_stage4_results_viewer_one_cell.md`

필수 backup root:
- `/kaggle/working/stage4_saved_outputs`
