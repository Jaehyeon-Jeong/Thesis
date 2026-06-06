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
- `kaggle_stage4_news_context_n6_baseline_controls_one_cell.md`
  - Stage 4 news-context priority 6 control.
  - Ensures headline-window TF-IDF/SVD artifacts exist, builds the N5
    `102`-dimensional prebuilt news context for five seeds, then runs
    `CNN + news concat`.
  - Runs `I60/R20/ohlc_ma_vb`, headline windows `7/20/60`, seeds
    `42, 43, 44, 45, 46`.
  - Purpose: test whether headline-only news context is useful as side
    information before adding news-conditioned FiLM.
  - Writes one compact download bundle:
    `/kaggle/working/stage4_news_context_n6_result_bundle.zip`.
- `kaggle_stage4_news_context_n6_svd_dim_grid_one_cell.md`
  - Stage 4 news-context priority 6.1 diagnostic.
  - Rebuilds train-only TF-IDF/SVD vectors with SVD dims `16` and `8`,
    builds matching prebuilt news context artifacts, then runs
    `CNN + news concat` over five seeds.
  - Purpose: check whether smaller headline vectors reduce seed collapse before
    moving to N7 news-conditioned bounded FiLM.
  - Writes one compact download bundle:
    `/kaggle/working/stage4_news_context_n6_svd_dim_grid_result_bundle.zip`.
- `kaggle_stage4_news_context_n7_bounded_film_svd8_one_cell.md`
  - Stage 4 news-context priority 7 main FiLM test.
  - Uses the selected SVD8 headline-news vector, context dim `30`, and runs
    `film_full_bounded_last_block` over five seeds.
  - FiLM scale: `0.05`.
  - Grad-CAM/context modulation export is enabled.
  - Writes one compact download bundle:
    `/kaggle/working/stage4_news_context_n7_bounded_film_svd8_result_bundle.zip`.

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
- `kaggle_stage4_news_context_n6_baseline_controls_one_cell.md`
  - Stage 4 news-context 우선순위 6 control입니다.
  - headline-window TF-IDF/SVD artifact가 없으면 다시 만들고, N5의
    `102`차원 prebuilt news context를 five-seed용으로 만든 뒤
    `CNN + news concat`을 실행합니다.
  - `I60/R20/ohlc_ma_vb`, headline window `7/20/60`, seeds
    `42, 43, 44, 45, 46`입니다.
  - 목적: news-conditioned FiLM으로 넘어가기 전에 headline-only news context가
    side information으로 유용한지 먼저 확인합니다.
  - 결과 download bundle:
    `/kaggle/working/stage4_news_context_n6_result_bundle.zip`.
- `kaggle_stage4_news_context_n6_svd_dim_grid_one_cell.md`
  - Stage 4 news-context 우선순위 6.1 diagnostic입니다.
  - train-only TF-IDF/SVD vector를 SVD dim `16`, `8`로 다시 만들고,
    matching prebuilt news context를 만든 뒤 `CNN + news concat` five-seed를
    실행합니다.
  - 목적: N7 news-conditioned bounded FiLM으로 넘어가기 전에 낮은 차원의
    headline vector가 seed collapse를 줄이는지 확인하는 것입니다.
  - 결과 download bundle:
    `/kaggle/working/stage4_news_context_n6_svd_dim_grid_result_bundle.zip`.
- `kaggle_stage4_news_context_n7_bounded_film_svd8_one_cell.md`
  - Stage 4 news-context 우선순위 7 main FiLM test입니다.
  - 선택된 SVD8 headline-news vector, context dim `30`을 사용하고
    `film_full_bounded_last_block` five-seed를 실행합니다.
  - FiLM scale은 `0.05`입니다.
  - Grad-CAM/context modulation export가 켜져 있습니다.
  - 결과 download bundle:
    `/kaggle/working/stage4_news_context_n7_bounded_film_svd8_result_bundle.zip`.
- `kaggle_stage4_n8b_fg_only_pretrained_frozen_bounded_film_one_cell.md`
  - Stage 2 `I60/R20/ohlc_ma_vb` checkpoint를 load하고 CNN/classifier를
    freeze한 뒤 F&G-only context encoder와 bounded final-block FiLM head만
    학습합니다.
  - 목적: context-FiLM이 baseline을 새로 흔드는 것이 아니라 강한 Stage 2
    visual model 위에서 correction으로 작동하는지 확인합니다.
- `kaggle_stage4_n9_news_pretrained_frozen_svd_scale_grid_one_cell.md`
  - N8-B와 같은 frozen Stage 2 setup에 headline TF-IDF/SVD news context를
    붙입니다.
  - Grid: SVD dim `8/16/32`, scale `0.02/0.05` 중 이미 실행한
    `SVD8/0.02`를 제외합니다.
- `kaggle_stage4_n10_selected_news_interpretability_one_cell.md`
  - 선택된 news bounded-FiLM 후보를 다시 실행하면서 Grad-CAM/context
    modulation artifact를 저장합니다.
- `kaggle_stage4_n10_stage2_vs_n10_correction_analysis_one_cell.md`
  - Stage 2 prediction과 N10 prediction을 같은 sample index로 비교해
    correction/regression table을 만듭니다.
- `kaggle_stage4_n10_b_targeted_gradcam_modulation_one_cell.md`
  - N10 correction-analysis에서 선택한 sample에 대해 Stage 2와 N10
    targeted Grad-CAM 및 gamma/beta modulation metadata를 export합니다.
- `kaggle_stage4_n12a_uncertainty_gated_news_film_one_cell.md`
  - N12-A runner입니다.
  - Stage 2 CNN/classifier를 frozen으로 보존하고, news SVD32 context가 만든
    final-block FiLM을 `4 * p_up * (1 - p_up)` uncertainty gate로 조절합니다.
  - Grid: scale `0.02`, `0.05` x seeds `42, 43, 44, 45, 46`.
- `kaggle_stage4_n12b_confidence_gated_news_film_one_cell.md`
  - N12-B runner입니다.
  - Stage 2 CNN/classifier를 frozen으로 보존하고, news SVD32 context가 만든
    final-block FiLM을 `abs(2 * p_up - 1)` confidence gate로 조절합니다.
  - Grid: scale `0.02`, `0.05` x seeds `42, 43, 44, 45, 46`.
- `kaggle_stage4_n12c_technical_only_pretrained_frozen_bounded_film_one_cell.md`
  - N12-C runner입니다.
  - Stage 2 CNN/classifier를 frozen으로 보존하고, OHLCV-derived technical
    context `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60`만
    bounded final-block FiLM condition으로 사용합니다.
  - Grid: scale `0.02`, `0.05` x seeds `42, 43, 44, 45, 46`.
  - 목적: F&G/news처럼 image-external context가 아니라, chart에서 이미 어느
    정도 보이는 technical context가 별도 FiLM 조건으로 유용한지 분리해서
    확인하는 것입니다.
- `kaggle_stage4_n13_1_ofr_fsi_context_features_one_cell.md`
  - N13-1 runner입니다.
  - OFR Financial Stress Index CSV를 읽고, BTC sample date보다 하루 이전
    값만 사용해 source-level `value`, `mean_20`, `mean_60`, `delta_20`,
    `delta_60`, `std_60` feature를 만듭니다.
  - 목적: FSI를 공식 financial-stress/risk-off proxy로 정렬하고, Stage 4
    frozen FiLM이 쓸 수 있는 prebuilt macro context artifact를 준비합니다.
- `kaggle_stage4_n13_2_fsi_only_pretrained_frozen_bounded_film_one_cell.md`
  - N13-2 runner입니다.
  - Stage 2 CNN/classifier를 frozen으로 보존하고, OFR FSI context encoder와
    bounded final-block FiLM head만 학습합니다.
  - Feature-set grid: `fsi_2 = mean_60 + delta_60`,
    `fsi_3 = mean_60 + delta_60 + std_60`, `fsi_all = all six FSI features`.
  - Grid: scale `0.02` x seeds `42, 43, 44, 45, 46`.
  - 목적: 여섯 개 FSI feature를 무조건 모두 쓰지 않고, screening에서 나온
    compact risk-regime feature set이 더 안정적인지 확인합니다.

예정 notebook:
- `kaggle_stage4_single_ablation_one_cell.md`
- `kaggle_stage4_results_viewer_one_cell.md`

필수 backup root:
- `/kaggle/working/stage4_saved_outputs`
