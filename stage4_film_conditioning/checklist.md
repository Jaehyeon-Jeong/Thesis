# Stage 4 Checklist

## English

Stage 4 now tests **how market context should be attached to the fixed BTC
chart-image CNN**. The image pipeline stays fixed; the comparison is between
context fusion/modulation methods.

Fixed baseline:
- Image/model family: Stage 2 `I60/R20/ohlc_ma_vb`.
- Reason: selected five-seed Stage 2 best configuration.
- Baseline metrics: accuracy mean `0.5793`, ROC-AUC mean `0.5849`.
- Stage 3 Linear result is kept as a negative/simple-parameter ablation.

Active work view:
- Completed numeric-context path: `4-A`/`4-B`/`4-C`/`4-D`, v1, and v2
  diagnostics through `4-V9`.
- Current conclusion: structured numeric context, including F&G-only FiLM, has
  useful ranking signal in some seeds but does not robustly beat the Stage 2
  visual baseline.
- Current next track: `4-N0` to `4-N10`, news context with strict `t-1`
  alignment.
- First news version: headline-only, non-LLM, train-only TF-IDF/SVD over
  7/20/60-day trailing news windows.
- Main order now: source audit -> leakage-safe news alignment -> news vector
  builder -> visual-only same sample -> `CNN + news concat` -> bounded
  last-block FiLM -> interpretability report.

Main Stage 4 ablation:
- [x] 4-A. `CNN + context concat`
  - Context is encoded by MLP and appended to the CNN feature before the
    classifier.
  - Question: is simple side-information fusion enough?
- [x] 4-B. `CNN + context gating`
  - Context creates channel/feature gates that multiply CNN features.
  - Question: is simple multiplicative modulation enough?
- [x] 4-C. `CNN + context FiLM gamma-only`
  - Context creates block-wise `gamma`; apply `F' = gamma * F`.
  - Question: is FiLM-style scaling enough without additive shift?
- [x] 4-D. `CNN + context FiLM full`
  - Context creates block-wise `gamma` and `beta`; apply
    `F' = gamma * F + beta`.
  - Question: does full FiLM give the best conditional adaptation and
    interpretability?
  - Result links: [4-I12 seed-42 four-ablation](checklist_results/4-I12_kaggle_four_ablation_runner.md),
    [4-I13 five-seed four-ablation](checklist_results/4-I13_kaggle_five_seed_runner.md),
    [v1 interpretation report](reports/stage4_v1_interpretation/stage4_v1_interpretation_report.md).

Planning phase:
- [x] 4-0. Stage 4 folder, checklist, and workflow scaffold
  - Result: [4-0 Stage 4 scaffold](checklist_results/4-0_stage4_scaffold.md)
- [x] 4-1. Context fusion and news-context plan
  - Result: [4-1 Context fusion and news plan](checklist_results/4-1_context_fusion_and_news_plan.md)
- [x] 4-2. Structured numeric context audit and leakage policy
  - F&G, Bollinger %B, Bollinger bandwidth, MFI, realized volatility.
  - Primary decision: `context_window = image_window`.
  - For the selected `I60/R20/ohlc_ma_vb` baseline, use matched 60-day
    context first: `F&G60`, `BB60`, `MFI60`, `RV60`.
  - Keep `BB20`, `MFI14`, and short F&G summaries only as later
    `standard_window` or `multi_scale` diagnostics.
  - Result: [4-2 Structured context audit and leakage policy](checklist_results/4-2_structured_context_audit_and_leakage_policy.md)
- [x] 4-3. News dataset audit and news-context feasibility decision
  - Candidate: `edaschau/bitcoin_news`.
  - Decision: feasible as a second-phase context source.
  - First news version: headline-only, strict `t-1` alignment, train-fit
    non-LLM encoder.
  - Defer article summaries and LLM embeddings until leakage-safe headline
    context is stable.
  - Result: [4-3 News dataset audit and feasibility decision](checklist_results/4-3_news_dataset_audit_and_feasibility.md)
- [x] 4-4. Stage 2/Stage 3 dependency and baseline-output review
  - Locked primary baseline: Stage 2 `I60/R20/ohlc_ma_vb`.
  - Primary comparison target: five-seed accuracy mean `0.579320`,
    ROC-AUC mean `0.584862`.
  - Stage 3 Linear is a negative/simple-parameter ablation, not a Stage 4 code
    dependency.
  - Result: [4-4 Stage 2/Stage 3 dependency and baseline output review](checklist_results/4-4_stage2_stage3_dependency_and_baseline_output_review.md)
- [x] 4-5. Context encoder and normalization plan
  - Primary context vector: 8 matched-window features.
  - Preprocessing: feature transform, train-only median imputation,
    train-only 1/99% clipping, train-only z-score normalization.
  - Shared encoder: `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
  - Result: [4-5 Context encoder and normalization plan](checklist_results/4-5_context_encoder_and_normalization_plan.md)
- [x] 4-6. Concat/gating/FiLM insertion design
  - 4-A concat attaches the 32-dim context embedding after CNN flatten:
    `184320 + 32 -> Linear(..., 2)`.
  - 4-B gating applies a final-block channel gate to `(B, 512, 2, 180)`.
  - 4-C/4-D FiLM is inserted after BatchNorm and before LeakyReLU in every I60
    block.
  - Result: [4-6 Concat/gating/FiLM insertion design](checklist_results/4-6_concat_gating_film_insertion_design.md)
- [x] 4-7. Grad-CAM plus context/gate/gamma/beta export plan
  - Primary target is the predicted-class pre-softmax logit.
  - Final report figure uses 10 Predicted Up and 10 Predicted Down test samples.
  - Export context values and gate/gamma/beta values beside Grad-CAM samples.
  - Result: [4-7 Grad-CAM plus context/gate/gamma/beta export plan](checklist_results/4-7_gradcam_context_modulation_export_plan.md)
- [x] 4-8. Kaggle runner and output backup plan
  - Runner stages: context build, training, prediction evaluation, trading
    evaluation, Grad-CAM/export, output check, and summary.
  - Backup root: `/kaggle/working/stage4_saved_outputs`.
  - Completion requires the output checker to pass; checkpoint existence alone
    is not enough.
  - Result: [4-8 Kaggle runner and output backup plan](checklist_results/4-8_kaggle_runner_and_output_backup_plan.md)

Advisor confirmation/reporting:
- [x] 4-R1. Professor meeting direction brief
  - Purpose: summarize Stage 1-4 progress, explain why Stage 4 is
    market-context-conditioned feature modulation, and map 4-A/B/C/D to the
    advisor direction file excerpts.
  - Result: [Professor meeting direction brief](docs/professor_meeting_stage4_direction_brief.md)
- [ ] 4-R2. Advisor feedback and final Stage 4 scope lock
  - Confirm whether Stage 4 should proceed with `matched_window` numeric
    context first.
  - Confirm whether news/LLM remains second-phase after numeric context.
  - Confirm whether 4-A concat, 4-B gating, 4-C gamma-only FiLM, and 4-D full
    FiLM are the intended ablation set.

Implementation phase:
- [x] 4-I0. Implementation readiness review
  - Decision: implementation can proceed to `4-I1`.
  - Stage 4 will reuse Stage 2 BTC data/image/split/evaluation helpers through
    a configurable Stage 2 `src` dependency.
  - Local BTC and F&G data are available for local context feature development.
  - Kaggle runs should still attach the public F&G dataset for reproducibility.
  - Result: [4-I0 Implementation readiness review](checklist_results/4-I0_implementation_readiness_review.md)
  - Data update: [4-I0 Fear & Greed local data check](checklist_results/4-I0_fear_greed_local_data_check.md)
- [x] 4-I1. Shared Stage 4 config/code scaffold
  - Added local/Kaggle configs, Stage 4 config/path/runtime/seed helpers, and a
    scaffold checker.
  - The local scaffold check finds BTC, F&G, and Stage 2 `src` successfully.
  - Result: [4-I1 Shared Stage 4 config/code scaffold](checklist_results/4-I1_shared_code_config_scaffold.md)
- [x] 4-I2. Structured context feature builder
  - Added F&G source audit, OHLCV-derived context features, and train-only
    context preprocessing.
  - Local I60/R20/ohlc_ma_vb context build produced 2,399 rows with no primary
    feature missing-rate warnings.
  - Result: [4-I2 Structured context feature builder](checklist_results/4-I2_structured_context_feature_builder.md)
- [x] 4-I3. Context MLP encoder
  - Added shared `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`
    context encoder.
  - Shape check passed on dummy tensors and real normalized rows from the local
    `4-I2` context table.
  - Result: [4-I3 Context MLP encoder](checklist_results/4-I3_context_mlp_encoder.md)
- [x] 4-I4. `CNN + context concat` model
  - Reused the Stage 2 I60 Stock_CNN convolution blocks unchanged.
  - Replaced only the final classifier so `(B, 184320)` image features and
    `(B, 32)` context embeddings become `(B, 184352)` before logits.
  - Parameter check passed: `2,954,370` parameters, `+1,408` vs Stage 2 I60
    baseline.
  - Result: [4-I4 Context concat model](checklist_results/4-I4_context_concat_model.md)
- [x] 4-I5. `CNN + context gating` model
  - Added final-block channel gating with `gate = 2 * sigmoid(raw_gate)`.
  - Context embedding `(B, 32)` generates `(B, 512)` gates for the final I60
    feature map `(B, 512, 2, 180)`.
  - Gate head is zero-initialized, so the model starts from identity
    modulation with gate min/max `1.0 / 1.0`.
  - Parameter check passed: `2,971,202` parameters, `+18,240` vs Stage 2 I60
    baseline.
  - Result: [4-I5 Context gating model](checklist_results/4-I5_context_gating_model.md)
- [x] 4-I6. FiLM layer and FiLM generator modules
  - Added reusable `FeatureWiseAffineModulation`.
  - Added `FilmParameterGenerator` for `film_gamma` and `film_full`.
  - Gamma is initialized as `1 + delta_gamma`; beta is initialized to `0`.
  - Local check passed for all I60 block feature maps.
  - Generator parameter checks passed: `31,680` for `film_gamma`, `63,360` for
    `film_full`.
  - Result: [4-I6 FiLM layer and generator](checklist_results/4-I6_film_layer_generator.md)
- [x] 4-I7. `CNN + FiLM gamma-only` and `CNN + FiLM full` models
  - Added `FilmContextStockCNN`.
  - Inserted FiLM as `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`
    in every I60 block.
  - `film_gamma` parameter check passed: `2,985,986`, `+33,024` vs Stage 2 I60.
  - `film_full` parameter check passed: `3,017,666`, `+64,704` vs Stage 2 I60.
  - Identity initialization check passed for all four I60 FiLM blocks.
  - Result: [4-I7 FiLM context models](checklist_results/4-I7_film_context_models.md)
- [x] 4-I8. BTC Stage 4 runner using fixed Stage 2 data pipeline
  - Added `run_stage4_context_model.py`, Stage 4 runner helpers, and a
    context-aware training loop.
  - The runner reuses Stage 2 BTC data/image/split/pixel-normalization and
    adds normalized context tensors to each batch.
  - Local smoke training passed for `concat` and `film_gamma`.
  - Result: [4-I8 Stage 4 context runner](checklist_results/4-I8_stage4_context_runner.md)
- [x] 4-I9. Prediction, classification metric, and trading metric export
  - Added Stage 4 prediction helper and evaluation scripts.
  - Exports `test_predictions.csv`, `test_metrics.json`, and
    `test_trading_metrics.json`.
  - Reuses Stage 2 classification/trading metric implementations, with
    `model(image, context)` for Stage 4.
  - Local export checks passed for `concat` and `film_gamma` smoke checkpoints.
  - Result: [4-I9 Prediction and trading exports](checklist_results/4-I9_prediction_trading_exports.md)
- [x] 4-I10. Grad-CAM plus context/gate/gamma/beta export
  - Added Stage 4 Grad-CAM helper and export script.
  - Grad-CAM target remains the predicted-class pre-softmax logit, now through
    `model(image, context)`.
  - Exports `samples.csv`, `modulation_summary.csv`, and
    `modulation_values.json` beside the figure.
  - Local Grad-CAM export checks passed for `concat` and `film_gamma` smoke
    checkpoints.
  - Result: [4-I10 Grad-CAM context/modulation export](checklist_results/4-I10_gradcam_context_modulation_export.md)
- [x] 4-I11. Local or small Kaggle smoke test
  - Added `check_stage4_outputs.py`.
  - The checker verifies checkpoint, training metadata, predictions,
    classification metrics, trading metrics, Grad-CAM, samples,
    modulation exports, context artifacts, and manifest.
  - Local output checks passed for `concat` and `film_gamma` smoke runs.
  - Result: [4-I11 Smoke output check](checklist_results/4-I11_smoke_output_check.md)
- [x] 4-I12. Kaggle single-config run for the four main ablations
  - Completed on Kaggle for `I60/R20/ohlc_ma_vb`, context window `60`,
    seed `42`, methods `concat`, `gating`, `film_gamma`, `film_full`.
  - Result: `film_full` was best among Stage 4 methods with accuracy
    `0.584316` and ROC-AUC `0.596811`.
  - Interpretation: promising versus the Stage 2 five-seed mean, but not yet
    better than the same Stage 2 seed-42 run; five-seed robustness is required.
  - Result: [4-I12 Kaggle four-ablation run](checklist_results/4-I12_kaggle_four_ablation_runner.md)
- [x] 4-I13. Kaggle selected grid/five-seed runner
  - Runner:
    `notebooks/kaggle_stage4_four_ablation_five_seed_one_cell.md`.
  - Fixed run: `I60/R20/ohlc_ma_vb`, context window `60`,
    seeds `42, 43, 44, 45, 46`, methods `concat`, `gating`, `film_gamma`,
    `film_full`.
  - Result: completed. Best v1 method was `film_full` with five-seed accuracy
    mean `0.5510` and ROC-AUC mean `0.5677`, below the Stage 2
    `I60/R20/ohlc_ma_vb` baseline.
  - Result prep: [4-I13 Kaggle five-seed runner](checklist_results/4-I13_kaggle_five_seed_runner.md)
- [ ] 4-I14. Stage 4 result report
  - Numeric-context reporting is complete through `4-V9`.
  - Final Stage 4 result report waits for the news-context track.

Stage 4 v2 diagnostic priorities:
- [x] 4-V0. Priority 1: Stage 4 same-split visual-only baseline,
  `I60/R20/ohlc_ma_vb`, no context
  - Purpose: separate context/FiLM effects from the selected image baseline and
    confirm whether the Stage 4 sample universe itself explains the v1 drop.
  - Execution wrapper:
    `notebooks/kaggle_stage4_v2_p1_visual_only_same_split_one_cell.md`.
  - Result prep: [4-V0 Stage 4 v2 visual-only same-split plan](checklist_results/4-V0_stage4_v2_visual_only_same_split.md)
- [x] 4-V1. Priority 2: Stage 4 same-split visual-only baseline,
  `I60/R20/ohlc`, no context
  - Purpose: measure how much the strong `ohlc_ma_vb` image already encodes
    technical information.
  - Execution wrapper:
    `notebooks/kaggle_stage4_v2_p2_ohlc_visual_only_one_cell.md`.
  - Result prep: [4-V1 Stage 4 v2 OHLC visual-only control](checklist_results/4-V1_stage4_v2_ohlc_visual_only.md)
- [x] 4-V2. Priority 3: `I60/R20/ohlc` + all structured context + `film_full`
  - Purpose: test the duplicate-feature hypothesis by removing MA/VB from the
    image while keeping F&G/BB/MFI/RV as context.
  - Execution wrapper:
    `notebooks/kaggle_stage4_v2_p3_ohlc_all_context_film_full_one_cell.md`.
  - Result: seed-42 accuracy `0.5725`, ROC-AUC `0.5573`; it improved over the
    OHLC-only seed-42 control but did not reach the strong `ohlc_ma_vb`
    visual baseline.
  - Follow-up: expanded to five seeds in `4-V5`; the seed-42 gain was not
    robust.
  - Result prep: [4-V2 Stage 4 v2 OHLC all-context FiLM-full](checklist_results/4-V2_stage4_v2_ohlc_all_context_film_full.md)
- [x] 4-V3. Priority 4: `I60/R20/ohlc` + F&G-only + `film_full`
  - Purpose: isolate image-external regime/sentiment context from OHLCV-derived
    technical context.
  - Execution wrapper:
    `notebooks/kaggle_stage4_v2_p4_ohlc_fg_only_film_full_five_seed_one_cell.md`.
  - Result: five-seed mean accuracy `0.5586`, ROC-AUC `0.5523`; F&G-only did
    not materially improve over the Stage 2 OHLC baseline.
  - Result prep: [4-V3 Stage 4 v2 OHLC F&G-only FiLM-full](checklist_results/4-V3_stage4_v2_ohlc_fg_only_film_full.md)
- [x] 4-V4. Priority 5: `I60/R20/ohlc` + technical-only context + `film_full`
  - Purpose: test whether BB/MFI/RV help when they are not already drawn into
    the image through MA/VB-style visual cues.
  - Execution wrapper:
    `notebooks/kaggle_stage4_v2_p5_ohlc_technical_only_film_full_five_seed_one_cell.md`.
  - Result: five-seed mean accuracy `0.5603`, ROC-AUC `0.5546`; technical-only
    context was slightly above F&G-only but did not materially improve over the
    Stage 2 OHLC baseline.
  - Result prep: [4-V4 Stage 4 v2 OHLC technical-only FiLM-full](checklist_results/4-V4_stage4_v2_ohlc_technical_only_film_full.md)
- [x] 4-V5. Priority 6: `I60/R20/ohlc` + all structured context + `film_full`,
  five seeds
  - Purpose: determine whether the earlier seed-42 all-context improvement was
    a real combination effect or a lucky seed.
  - Execution wrapper:
    `notebooks/kaggle_stage4_v2_p6_ohlc_all_context_film_full_five_seed_one_cell.md`.
  - Result: five-seed mean accuracy `0.5574`, ROC-AUC `0.5519`; the seed-42
    all-context gain was not robust.
  - Result prep: [4-V5 Stage 4 v2 OHLC all-context five-seed](checklist_results/4-V5_stage4_v2_ohlc_all_context_five_seed.md)
- [x] 4-V6. Priority 7: `I60/R20/ohlc_ma_vb` + F&G-only + `film_full`,
  five seeds
  - Purpose: test whether external sentiment/regime context adds incremental
    signal on top of the strongest visual baseline.
  - Execution wrapper:
    `notebooks/kaggle_stage4_v2_p7_ohlc_ma_vb_fg_only_film_full_five_seed_one_cell.md`.
  - Result: five-seed mean accuracy `0.5524`, ROC-AUC `0.5465`; seeds
    `42`, `45`, and `46` were close to the Stage 2 visual baseline, but seeds
    `43` and `44` collapsed toward mostly Up predictions.
  - Result prep: [4-V6 Stage 4 v2 OHLC_MA_VB F&G-only five-seed](checklist_results/4-V6_stage4_v2_ohlc_ma_vb_fg_only_five_seed.md)
- [x] 4-V7. Priority 8: bounded/residual last-block FiLM v2
  - Purpose: preserve the Stage 2 visual evidence and reduce seed-dependent
    collapse by limiting modulation strength and applying FiLM only to the
    high-level final block first.
  - Execution wrapper:
    `notebooks/kaggle_stage4_v2_p8_ohlc_ma_vb_fg_only_bounded_last_block_film_five_seed_one_cell.md`.
  - Fixed run: `I60/R20/ohlc_ma_vb`, F&G-only context,
    `film_full_bounded_last_block`, seeds `42, 43, 44, 45, 46`.
  - Result: five-seed mean accuracy `0.5425`, ROC-AUC `0.5763`; ROC-AUC and
    average precision improved over `4-V6` `film_full`, but seeds `43` and
    `44` collapsed toward mostly Down predictions.
  - Result prep: [4-V7 Stage 4 v2 bounded/residual last-block FiLM](checklist_results/4-V7_stage4_v2_bounded_residual_last_block_film.md)
- [x] 4-V8. Priority 9: P7/P8 seed-collapse diagnostic and validation-threshold
  calibration
  - Purpose: analyze why `film_full` seeds `43`/`44` collapse mostly Up while
    bounded last-block FiLM seeds `43`/`44` collapse mostly Down before running
    another gamma/beta scale grid.
  - Diagnostic wrapper:
    `notebooks/kaggle_stage4_v2_v8_p7_p8_seed_collapse_diagnostic_one_cell.md`.
  - Script:
    `scripts/analyze_stage4_seed_collapse.py`.
  - Outputs: default-threshold metrics, validation-calibrated threshold test
    metrics, probability quantiles, and P7/P8 paired prediction comparison.
  - Result: P8 improved ranking signal versus P7 but did not solve the class
    decision collapse; validation-threshold calibration alone was not enough.
  - Result prep: [4-V8 Stage 4 v2 P7/P8 seed-collapse diagnostic](checklist_results/4-V8_stage4_v2_p7_p8_seed_collapse_diagnostic.md)
- [x] 4-V9. Priority 10: bounded last-block FiLM scale stability grid
  - Purpose: before moving to news context, test whether the P8 collapse comes
    from an overly strong bounded FiLM scale rather than the whole FiLM idea.
  - Execution wrapper:
    `notebooks/kaggle_stage4_v2_v9_bounded_last_block_film_scale_grid_one_cell.md`.
  - Fixed run: `I60/R20/ohlc_ma_vb`, F&G-only context,
    `film_full_bounded_last_block`, scales `0.02`, `0.05`, `0.10`, seeds
    `42, 43, 44, 45, 46`.
  - Checkpoint rule is not changed in this step; V9 records validation/test
    collapse metrics but keeps the experiment axis scale-only.
  - Result: scales `0.02`, `0.05`, and `0.10` all stayed below the Stage 2
    visual baseline. Lower scales reduced some collapse, but seed `44`
    collapsed mostly Down for every scale.
  - Decision: structured F&G-only FiLM is not robust enough to keep tuning
    gamma/beta scale. Move to the news-context track as the next external
    regime source.
  - Result prep: [4-V9 Stage 4 v2 bounded last-block FiLM scale grid](checklist_results/4-V9_stage4_v2_bounded_last_block_film_scale_grid.md)

News-context extension:
- [x] 4-N0. Numeric-context handoff and news scope lock
  - Record V9 conclusion: F&G-only numeric FiLM gives some ranking signal but
    does not robustly beat Stage 2 `I60/R20/ohlc_ma_vb`.
  - Lock the first news track as headline-only, non-LLM, strict `t-1`.
  - Do not continue arbitrary gamma/beta scale search before testing richer
    external context.
  - Result prep: [4-N0 Numeric-context handoff and news scope lock](checklist_results/4-N0_numeric_context_handoff_news_scope_lock.md)
- [x] 4-N1. `edaschau/bitcoin_news` source audit
  - Check row count, date range, columns, source distribution, title/article
    coverage, duplicate URL/title rate, and rows per day.
  - Confirm overlap with BTC sample period and five-seed Stage 4 selected
    sample universe.
  - Result: headline-only file `BTC_match_title.csv` has `30,626` rows from
    `2011-06-22` to `2025-06-03`; selected Stage 4 samples run from
    `2018-04-29` to `2024-12-11`.
  - Preliminary source coverage: strict `t-1` sample coverage is `96.04%`;
    trailing 7-day news coverage is `100.00%`. 4-N2 rechecks 7/20/60-day
    windows.
  - Result: [4-N1 News source audit](checklist_results/4-N1_news_source_audit_design.md)
  - Tables:
    [source audit](reports/tables/stage4_news_source_audit.json),
    [sample coverage](reports/tables/stage4_news_sample_coverage_by_split.csv),
    [source distribution](reports/tables/stage4_news_source_distribution.csv)
- [x] 4-N2. Publication-time alignment and no-future-leakage rule
  - Default policy: for chart image ending at date `t`, use news up to calendar
    date `t-1`.
  - Do not use same-day news until BTC close cutoff and news timestamp cutoff
    are explicitly defended.
  - Produce split-level missing-day/news-count audit.
  - Result: strict `t-1` policy locked. Same-day news is explicitly excluded
    for `2,304 / 2,399` samples where same-day headlines exist.
  - Coverage: train `96.57%`, validation `97.21%`, test `95.56%`; trailing
    7/20/60-day coverage is `100.00%` for every split.
  - Text vectorizer fit rule: fit only on train strict-`t-1` 7/20/60-day
    headline-window documents (`671` samples x `3` windows); validation/test
    documents are transform-only.
  - Result: [4-N2 News publication-time alignment](checklist_results/4-N2_news_publication_time_alignment.md)
  - Tables:
    [policy](reports/tables/stage4_news_alignment_policy.json),
    [by split](reports/tables/stage4_news_alignment_by_split.csv),
    [examples](reports/tables/stage4_news_alignment_examples.csv)
- [x] 4-N3. Headline-only headline-window aggregation table
  - Use `title`, `date_time`, `source`, and `url` first.
  - Remove exact duplicate URLs/titles.
  - Build leakage-safe sample-window fields for trailing `7d`, `20d`, and
    `60d`: concatenated headline text, news counts, and optionally source-count
    features.
  - Defer full `article_text` and summaries.
  - Result: raw headline rows `30,626` -> deduped rows `29,208`; duplicate
    normalized-title rows removed `1,418`.
  - Result: 7/20/60-day headline-window coverage is `100.00%` for train,
    validation, and test.
  - Full aggregation tables are stored under
    `outputs/stage4/news/stage4_news_headline_windows_i60_r20/`.
  - Result: [4-N3 Headline-window aggregation](checklist_results/4-N3_headline_window_aggregation.md)
  - Tables:
    [summary](reports/tables/stage4_news_headline_windows_summary.csv),
    [examples](reports/tables/stage4_news_headline_windows_examples.csv),
    [manifest](reports/tables/stage4_news_headline_windows_manifest.json)
- [ ] 4-N4. Train-only TF-IDF/SVD news vectorizer
  - Fit text preprocessing, TF-IDF vocabulary, IDF weights, and SVD only on
    train-period news.
  - First vector size: `news_svd_32` per window, producing `news_svd_7d`,
    `news_svd_20d`, and `news_svd_60d`.
  - No-news days use a zero news vector plus explicit count features.
  - Save vectorizer metadata, vocabulary hash, SVD dimension, and train-period
    fit range.
- [ ] 4-N5. BTC sample-level news context feature builder
  - Merge daily news vectors into each BTC image sample using strict `t-1`.
  - Candidate first context vector:
    `news_svd_7d + news_svd_20d + news_svd_60d + news_count_7d/20d/60d`.
  - Normalize non-text count features with train-only statistics.
  - Keep the chart image unchanged: `I60/R20/ohlc_ma_vb`.
- [ ] 4-N6. News-context baseline controls
  - Confirm Stage 2 visual-only baseline on the same news-aligned sample set.
  - Run `CNN + news concat` five-seed first.
  - Purpose: test whether the news vector is useful as side information before
    claiming FiLM modulation is useful.
- [ ] 4-N7. News-context bounded FiLM main test
  - Run `CNN + news bounded last-block FiLM` five-seed.
  - Start from the conservative V9 lesson: protect the visual path first.
  - Compare against Stage 2 visual baseline and `CNN + news concat`.
- [ ] 4-N8. News + F&G combined-context ablation
  - Run only after news-only shows useful signal.
  - Candidate vector: `news_svd_7d/20d/60d + news_count_7d/20d/60d + F&G-only`.
  - Purpose: test whether F&G helps as a compact regime summary beside richer
    news context.
- [ ] 4-N9. News interpretability report
  - Export Grad-CAM plus news titles, news-count features, and FiLM
    gamma/beta summaries for correct/incorrect Up/Down samples.
  - Add feature sensitivity: zero news vector, remove F&G, or replace news
    vector with train mean.
- [ ] 4-N10. LLM summary/embedding decision
  - Deferred until headline-only no-leakage track is stable.
  - If used, record model name, prompt, version/date, cache hash, and runtime.

Important:
- Do not draw the context values into the chart image for the main Stage 4
  experiment.
- The context enters as a separate vector.
- All context features must be available at or before image end date `t`.
- Train-only statistics must be used for context normalization.
- News is now the next Stage 4 context track after V9 showed that structured
  F&G-only numeric FiLM is not robust enough.

## 한국어

Stage 4는 이제 **market context를 고정된 BTC chart-image CNN에 어떻게 붙일지**를
검증하는 단계입니다. 이미지 파이프라인은 고정하고, context fusion/modulation 방식을
비교합니다.

고정 baseline:
- Image/model family: Stage 2 `I60/R20/ohlc_ma_vb`.
- 이유: Stage 2 selected five-seed best configuration.
- Baseline metrics: accuracy mean `0.5793`, ROC-AUC mean `0.5849`.
- Stage 3 Linear 결과는 단순 parameter 증가 비교의 negative ablation으로 둡니다.

현재 작업 보기:
- 완료된 numeric-context 경로: `4-A`/`4-B`/`4-C`/`4-D`, v1, v2 diagnostic
  `4-V9`까지.
- 현재 결론: F&G-only FiLM을 포함한 structured numeric context는 일부 seed에서
  ranking signal은 있지만, Stage 2 visual baseline을 안정적으로 넘지 못했습니다.
- 현재 다음 track: strict `t-1` alignment를 쓰는 news context `4-N0`부터
  `4-N10`까지.
- 첫 news version: headline-only, non-LLM, train-only TF-IDF/SVD를 7/20/60-day
  trailing news window에 적용합니다.
- 현재 순서: source audit -> leakage-safe news alignment -> news vector builder
  -> 같은 sample의 visual-only control -> `CNN + news concat` -> bounded
  last-block FiLM -> interpretability report.

Stage 4 main ablation:
- [x] 4-A. `CNN + context concat`
  - context를 MLP로 encoding한 뒤 classifier 직전 CNN feature에 붙입니다.
  - 질문: 단순 side information 추가만으로 충분한가?
- [x] 4-B. `CNN + context gating`
  - context가 channel/feature gate를 만들고 CNN feature에 곱합니다.
  - 질문: 단순 multiplicative modulation만으로 충분한가?
- [x] 4-C. `CNN + context FiLM gamma-only`
  - context가 block별 `gamma`를 만들고 `F' = gamma * F`를 적용합니다.
  - 질문: additive shift 없이 FiLM-style scaling만으로 충분한가?
- [x] 4-D. `CNN + context FiLM full`
  - context가 block별 `gamma`, `beta`를 만들고 `F' = gamma * F + beta`를 적용합니다.
  - 질문: full FiLM이 conditional adaptation과 해석력에서 가장 좋은가?
  - 결과 링크: [4-I12 seed-42 four-ablation](checklist_results/4-I12_kaggle_four_ablation_runner.md),
    [4-I13 five-seed four-ablation](checklist_results/4-I13_kaggle_five_seed_runner.md),
    [v1 interpretation report](reports/stage4_v1_interpretation/stage4_v1_interpretation_report.md).

계획 단계:
- [x] 4-0. Stage 4 폴더, checklist, workflow scaffold
  - 결과: [4-0 Stage 4 scaffold](checklist_results/4-0_stage4_scaffold.md)
- [x] 4-1. Context fusion과 news-context 계획
  - 결과: [4-1 Context fusion and news plan](checklist_results/4-1_context_fusion_and_news_plan.md)
- [x] 4-2. Structured numeric context audit와 leakage policy
  - F&G, Bollinger %B, Bollinger bandwidth, MFI, realized volatility.
  - Primary decision: `context_window = image_window`.
  - 선택된 `I60/R20/ohlc_ma_vb` baseline에서는 60일 matched context를 먼저 사용:
    `F&G60`, `BB60`, `MFI60`, `RV60`.
  - `BB20`, `MFI14`, short F&G summary는 나중의 `standard_window` 또는
    `multi_scale` diagnostic으로만 유지합니다.
  - 결과: [4-2 Structured context audit and leakage policy](checklist_results/4-2_structured_context_audit_and_leakage_policy.md)
- [x] 4-3. News dataset audit와 news-context 사용 가능성 결정
  - 후보: `edaschau/bitcoin_news`.
  - 결정: second-phase context source로 사용 가능.
  - 첫 news version: headline-only, strict `t-1` alignment, train-fit
    non-LLM encoder.
  - Article summary와 LLM embedding은 leakage-safe headline context가 안정화된
    뒤로 미룹니다.
  - 결과: [4-3 News dataset audit and feasibility decision](checklist_results/4-3_news_dataset_audit_and_feasibility.md)
- [x] 4-4. Stage 2/Stage 3 dependency와 baseline output 확인
  - Primary baseline 고정: Stage 2 `I60/R20/ohlc_ma_vb`.
  - Primary comparison target: five-seed accuracy mean `0.579320`,
    ROC-AUC mean `0.584862`.
  - Stage 3 Linear는 negative/simple-parameter ablation이며, Stage 4 code
    dependency가 아닙니다.
  - 결과: [4-4 Stage 2/Stage 3 dependency and baseline output review](checklist_results/4-4_stage2_stage3_dependency_and_baseline_output_review.md)
- [x] 4-5. Context encoder와 normalization 계획
  - Primary context vector: matched-window 8개 feature.
  - Preprocessing: feature transform, train-only median imputation,
    train-only 1/99% clipping, train-only z-score normalization.
  - Shared encoder: `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
  - 결과: [4-5 Context encoder and normalization plan](checklist_results/4-5_context_encoder_and_normalization_plan.md)
- [x] 4-6. Concat/gating/FiLM 삽입 설계
  - 4-A concat은 CNN flatten 뒤 32차원 context embedding을 붙입니다:
    `184320 + 32 -> Linear(..., 2)`.
  - 4-B gating은 final block feature map `(B, 512, 2, 180)`에 channel gate를
    적용합니다.
  - 4-C/4-D FiLM은 모든 I60 block에서 BatchNorm 뒤, LeakyReLU 전에 삽입합니다.
  - 결과: [4-6 Concat/gating/FiLM insertion design](checklist_results/4-6_concat_gating_film_insertion_design.md)
- [x] 4-7. Grad-CAM plus context/gate/gamma/beta export 계획
  - Primary target은 predicted-class pre-softmax logit입니다.
  - 최종 보고 figure는 test sample에서 Predicted Up 10개, Predicted Down 10개를
    사용합니다.
  - Grad-CAM sample 옆에 context 값과 gate/gamma/beta 값을 같이 export합니다.
  - 결과: [4-7 Grad-CAM plus context/gate/gamma/beta export plan](checklist_results/4-7_gradcam_context_modulation_export_plan.md)
- [x] 4-8. Kaggle runner와 output backup 계획
  - Runner 단계: context build, training, prediction evaluation, trading
    evaluation, Grad-CAM/export, output check, summary.
  - Backup root: `/kaggle/working/stage4_saved_outputs`.
  - 완료 판정은 output checker 통과 기준입니다. Checkpoint 존재만으로는 완료가
    아닙니다.
  - 결과: [4-8 Kaggle runner and output backup plan](checklist_results/4-8_kaggle_runner_and_output_backup_plan.md)

교수님 확인/보고:
- [x] 4-R1. 교수님 미팅용 방향성 brief
  - 목적: Stage 1-4 진행 현황, Stage 4를 market-context-conditioned feature
    modulation으로 해석한 이유, 4-A/B/C/D가 교수님 방향성 파일의 어떤 발췌와
    연결되는지 정리합니다.
  - 결과: [Professor meeting direction brief](docs/professor_meeting_stage4_direction_brief.md)
- [ ] 4-R2. 교수님 피드백 반영과 Stage 4 최종 scope lock
  - Stage 4를 `matched_window` numeric context부터 진행하는 것이 맞는지 확인합니다.
  - news/LLM을 numeric context 이후 second-phase로 두는 것이 맞는지 확인합니다.
  - 4-A concat, 4-B gating, 4-C gamma-only FiLM, 4-D full FiLM이 의도한
    ablation set인지 확인합니다.

구현 단계:
- [x] 4-I0. 구현 readiness review
  - 결정: `4-I1` 구현으로 진행할 수 있습니다.
  - Stage 4는 configurable Stage 2 `src` dependency를 통해 Stage 2 BTC
    data/image/split/evaluation helper를 재사용합니다.
  - 로컬 BTC와 F&G data가 local context feature 개발에 사용 가능합니다.
  - Kaggle run에서는 재현성을 위해 public F&G dataset을 계속 attach해야 합니다.
  - 결과: [4-I0 Implementation readiness review](checklist_results/4-I0_implementation_readiness_review.md)
  - 데이터 업데이트: [4-I0 Fear & Greed local data check](checklist_results/4-I0_fear_greed_local_data_check.md)
- [x] 4-I1. Stage 4 공통 config/code scaffold
  - Local/Kaggle config, Stage 4 config/path/runtime/seed helper, scaffold
    checker를 추가했습니다.
  - Local scaffold check에서 BTC, F&G, Stage 2 `src`를 정상 확인했습니다.
  - 결과: [4-I1 Shared Stage 4 config/code scaffold](checklist_results/4-I1_shared_code_config_scaffold.md)
- [x] 4-I2. Structured context feature builder
  - F&G source audit, OHLCV-derived context feature, train-only context
    preprocessing을 추가했습니다.
  - Local I60/R20/ohlc_ma_vb context build에서 2,399 row가 생성됐고 primary
    feature missing-rate warning은 없었습니다.
  - 결과: [4-I2 Structured context feature builder](checklist_results/4-I2_structured_context_feature_builder.md)
- [x] 4-I3. Context MLP encoder
  - 공통 `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`
    context encoder를 추가했습니다.
  - Dummy tensor와 local `4-I2` context table의 실제 normalized row에서 shape
    check를 통과했습니다.
  - 결과: [4-I3 Context MLP encoder](checklist_results/4-I3_context_mlp_encoder.md)
- [x] 4-I4. `CNN + context concat` model
  - Stage 2 I60 Stock_CNN convolution block은 그대로 재사용했습니다.
  - 마지막 classifier만 교체해서 `(B, 184320)` image feature와 `(B, 32)`
    context embedding을 `(B, 184352)`로 붙인 뒤 logits를 만듭니다.
  - Parameter check 통과: `2,954,370` parameters, Stage 2 I60 baseline 대비
    `+1,408`.
  - 결과: [4-I4 Context concat model](checklist_results/4-I4_context_concat_model.md)
- [x] 4-I5. `CNN + context gating` model
  - Final-block channel gating을 추가했고 `gate = 2 * sigmoid(raw_gate)`를
    사용합니다.
  - Context embedding `(B, 32)`이 마지막 I60 feature map `(B, 512, 2, 180)`에
    곱할 `(B, 512)` gate를 만듭니다.
  - Gate head는 zero-initialized라서 gate min/max `1.0 / 1.0`의 identity
    modulation에서 시작합니다.
  - Parameter check 통과: `2,971,202` parameters, Stage 2 I60 baseline 대비
    `+18,240`.
  - 결과: [4-I5 Context gating model](checklist_results/4-I5_context_gating_model.md)
- [x] 4-I6. FiLM layer와 FiLM generator module
  - 재사용 가능한 `FeatureWiseAffineModulation`을 추가했습니다.
  - `film_gamma`, `film_full`용 `FilmParameterGenerator`를 추가했습니다.
  - Gamma는 `1 + delta_gamma`로 초기화하고 beta는 `0`으로 초기화합니다.
  - 모든 I60 block feature map에서 local check를 통과했습니다.
  - Generator parameter check 통과: `film_gamma`는 `31,680`, `film_full`은
    `63,360`.
  - 결과: [4-I6 FiLM layer and generator](checklist_results/4-I6_film_layer_generator.md)
- [x] 4-I7. `CNN + FiLM gamma-only`와 `CNN + FiLM full` model
  - `FilmContextStockCNN`을 추가했습니다.
  - 모든 I60 block에 `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`
    순서로 FiLM을 삽입했습니다.
  - `film_gamma` parameter check 통과: `2,985,986`, Stage 2 I60 대비 `+33,024`.
  - `film_full` parameter check 통과: `3,017,666`, Stage 2 I60 대비 `+64,704`.
  - 네 개 I60 FiLM block 모두에서 identity initialization check를 통과했습니다.
  - 결과: [4-I7 FiLM context models](checklist_results/4-I7_film_context_models.md)
- [x] 4-I8. 고정된 Stage 2 data pipeline을 쓰는 BTC Stage 4 runner
  - `run_stage4_context_model.py`, Stage 4 runner helper, context-aware
    training loop를 추가했습니다.
  - Stage 2 BTC data/image/split/pixel-normalization을 그대로 재사용하고,
    각 batch에 normalized context tensor를 붙입니다.
  - `concat`, `film_gamma` local smoke training을 통과했습니다.
  - 결과: [4-I8 Stage 4 context runner](checklist_results/4-I8_stage4_context_runner.md)
- [x] 4-I9. prediction, classification metric, trading metric export
  - Stage 4 prediction helper와 evaluation script를 추가했습니다.
  - `test_predictions.csv`, `test_metrics.json`, `test_trading_metrics.json`를
    저장합니다.
  - Classification/trading metric 구현은 Stage 2를 재사용하고, 모델 호출만
    `model(image, context)`로 바꿨습니다.
  - `concat`, `film_gamma` smoke checkpoint에서 local export check를 통과했습니다.
  - 결과: [4-I9 Prediction and trading exports](checklist_results/4-I9_prediction_trading_exports.md)
- [x] 4-I10. Grad-CAM plus context/gate/gamma/beta export
  - Stage 4 Grad-CAM helper와 export script를 추가했습니다.
  - Grad-CAM target은 계속 predicted-class pre-softmax logit이며, 이제
    `model(image, context)` 경로로 계산합니다.
  - Figure 옆에 `samples.csv`, `modulation_summary.csv`,
    `modulation_values.json`을 저장합니다.
  - `concat`, `film_gamma` smoke checkpoint에서 local Grad-CAM export check를
    통과했습니다.
  - 결과: [4-I10 Grad-CAM context/modulation export](checklist_results/4-I10_gradcam_context_modulation_export.md)
- [x] 4-I11. local 또는 작은 Kaggle smoke test
  - `check_stage4_outputs.py`를 추가했습니다.
  - Checker는 checkpoint, training metadata, prediction, classification metric,
    trading metric, Grad-CAM, samples, modulation export, context artifact,
    manifest를 확인합니다.
  - `concat`, `film_gamma` smoke run에서 local output check를 통과했습니다.
  - 결과: [4-I11 Smoke output check](checklist_results/4-I11_smoke_output_check.md)
- [x] 4-I12. 네 가지 main ablation의 Kaggle single-config run
  - Kaggle에서 `I60/R20/ohlc_ma_vb`, context window `60`, seed `42`,
    methods `concat`, `gating`, `film_gamma`, `film_full` 실행 완료.
  - 결과: `film_full`이 Stage 4 방법 중 가장 좋았고 accuracy `0.584316`,
    ROC-AUC `0.596811`입니다.
  - 해석: Stage 2 five-seed mean과 비교하면 promising하지만, 같은 Stage 2 seed-42
    run보다 높지는 않으므로 five-seed robustness 확인이 필요합니다.
  - 결과: [4-I12 Kaggle four-ablation run](checklist_results/4-I12_kaggle_four_ablation_runner.md)
- [x] 4-I13. Kaggle selected grid/five-seed runner
  - Runner:
    `notebooks/kaggle_stage4_four_ablation_five_seed_one_cell.md`.
  - 고정 run: `I60/R20/ohlc_ma_vb`, context window `60`,
    seeds `42, 43, 44, 45, 46`, methods `concat`, `gating`, `film_gamma`,
    `film_full`.
  - 결과: 완료. v1에서는 `film_full`이 가장 나았지만 five-seed accuracy mean
    `0.5510`, ROC-AUC mean `0.5677`로 Stage 2 `I60/R20/ohlc_ma_vb` baseline보다
    낮았습니다.
  - 준비 결과: [4-I13 Kaggle five-seed runner](checklist_results/4-I13_kaggle_five_seed_runner.md)
- [ ] 4-I14. Stage 4 결과 보고
  - Numeric-context reporting은 `4-V9`까지 완료됐습니다.
  - 최종 Stage 4 결과 보고는 news-context track 이후 작성합니다.

Stage 4 v2 진단 우선순위:
- [x] 4-V0. 우선순위 1: Stage 4 same-split visual-only baseline,
  `I60/R20/ohlc_ma_vb`, context 없음
  - 목적: v1 성능 하락이 context/FiLM 때문인지, 선택된 Stage 4 sample universe
    자체 때문인지 분리합니다.
  - 실행 wrapper:
    `notebooks/kaggle_stage4_v2_p1_visual_only_same_split_one_cell.md`.
  - 준비 결과: [4-V0 Stage 4 v2 visual-only same-split plan](checklist_results/4-V0_stage4_v2_visual_only_same_split.md)
- [x] 4-V1. 우선순위 2: Stage 4 same-split visual-only baseline,
  `I60/R20/ohlc`, context 없음
  - 목적: 강한 `ohlc_ma_vb` 이미지가 technical 정보를 이미 얼마나 담고 있는지
    확인합니다.
  - 실행 wrapper:
    `notebooks/kaggle_stage4_v2_p2_ohlc_visual_only_one_cell.md`.
  - 준비 결과: [4-V1 Stage 4 v2 OHLC visual-only control](checklist_results/4-V1_stage4_v2_ohlc_visual_only.md)
- [x] 4-V2. 우선순위 3: `I60/R20/ohlc` + all structured context + `film_full`
  - 목적: 이미지에서 MA/VB를 덜어냈을 때 F&G/BB/MFI/RV context가 더 도움 되는지
    확인해 duplicate-feature 가설을 검증합니다.
  - 실행 wrapper:
    `notebooks/kaggle_stage4_v2_p3_ohlc_all_context_film_full_one_cell.md`.
  - 결과: seed-42 accuracy `0.5725`, ROC-AUC `0.5573`; OHLC-only seed-42
    control보다는 개선됐지만 강한 `ohlc_ma_vb` visual baseline에는 못 미쳤습니다.
  - 후속 확인: `4-V5`에서 five-seed로 확장했고, seed-42 gain은 robust하지
    않았습니다.
  - 준비 결과: [4-V2 Stage 4 v2 OHLC all-context FiLM-full](checklist_results/4-V2_stage4_v2_ohlc_all_context_film_full.md)
- [x] 4-V3. 우선순위 4: `I60/R20/ohlc` + F&G-only + `film_full`
  - 목적: 이미지 밖 regime/sentiment context만 따로 효과가 있는지 확인합니다.
  - 실행 wrapper:
    `notebooks/kaggle_stage4_v2_p4_ohlc_fg_only_film_full_five_seed_one_cell.md`.
  - 결과: five-seed mean accuracy `0.5586`, ROC-AUC `0.5523`; F&G-only는
    Stage 2 OHLC baseline을 실질적으로 개선하지 못했습니다.
  - 준비 결과: [4-V3 Stage 4 v2 OHLC F&G-only FiLM-full](checklist_results/4-V3_stage4_v2_ohlc_fg_only_film_full.md)
- [x] 4-V4. 우선순위 5: `I60/R20/ohlc` + technical-only context + `film_full`
  - 목적: BB/MFI/RV가 MA/VB 이미지 정보와 분리됐을 때 독립적으로 도움 되는지
    확인합니다.
  - 실행 wrapper:
    `notebooks/kaggle_stage4_v2_p5_ohlc_technical_only_film_full_five_seed_one_cell.md`.
  - 결과: five-seed mean accuracy `0.5603`, ROC-AUC `0.5546`; technical-only
    context는 F&G-only보다 약간 높았지만 Stage 2 OHLC baseline을 실질적으로
    개선하지 못했습니다.
  - 준비 결과: [4-V4 Stage 4 v2 OHLC technical-only FiLM-full](checklist_results/4-V4_stage4_v2_ohlc_technical_only_film_full.md)
- [x] 4-V5. 우선순위 6: `I60/R20/ohlc` + all structured context + `film_full`,
  five seeds
  - 목적: 이전 seed-42 all-context 개선이 실제 조합 효과인지, 좋은 seed였는지
    확인합니다.
  - 실행 wrapper:
    `notebooks/kaggle_stage4_v2_p6_ohlc_all_context_film_full_five_seed_one_cell.md`.
  - 결과: five-seed mean accuracy `0.5574`, ROC-AUC `0.5519`; seed-42
    all-context 개선은 robust하지 않았습니다.
  - 준비 결과: [4-V5 Stage 4 v2 OHLC all-context five-seed](checklist_results/4-V5_stage4_v2_ohlc_all_context_five_seed.md)
- [x] 4-V6. 우선순위 7: `I60/R20/ohlc_ma_vb` + F&G-only + `film_full`,
  five seeds
  - 목적: 가장 강한 visual baseline 위에 외부 sentiment/regime context가
    incremental signal을 주는지 확인합니다.
  - 실행 wrapper:
    `notebooks/kaggle_stage4_v2_p7_ohlc_ma_vb_fg_only_film_full_five_seed_one_cell.md`.
  - 결과: five-seed mean accuracy `0.5524`, ROC-AUC `0.5465`; seed
    `42`, `45`, `46`은 Stage 2 visual baseline에 근접했지만 seed `43`,
    `44`는 대부분 Up 예측으로 무너졌습니다.
  - 준비 결과: [4-V6 Stage 4 v2 OHLC_MA_VB F&G-only five-seed](checklist_results/4-V6_stage4_v2_ohlc_ma_vb_fg_only_five_seed.md)
- [x] 4-V7. 우선순위 8: bounded/residual last-block FiLM v2
  - 목적: Stage 2 visual evidence를 보존하고 modulation strength를 제한해서
    seed-dependent collapse를 줄입니다.
  - 실행 wrapper:
    `notebooks/kaggle_stage4_v2_p8_ohlc_ma_vb_fg_only_bounded_last_block_film_five_seed_one_cell.md`.
  - 고정 run: `I60/R20/ohlc_ma_vb`, F&G-only context,
    `film_full_bounded_last_block`, seeds `42, 43, 44, 45, 46`.
  - 결과: five-seed mean accuracy `0.5425`, ROC-AUC `0.5763`; `4-V6`
    `film_full`보다 ROC-AUC와 average precision은 개선됐지만 seed `43`,
    `44`는 대부분 Down 예측으로 무너졌습니다.
  - 준비 결과: [4-V7 Stage 4 v2 bounded/residual last-block FiLM](checklist_results/4-V7_stage4_v2_bounded_residual_last_block_film.md)
- [x] 4-V8. 우선순위 9: P7/P8 seed-collapse 진단과 validation-threshold
  calibration
  - 목적: 다음 gamma/beta scale grid를 돌리기 전에 `film_full` seed `43`/`44`가
    대부분 Up으로, bounded last-block FiLM seed `43`/`44`가 대부분 Down으로
    무너지는 이유를 분석합니다.
  - 진단 wrapper:
    `notebooks/kaggle_stage4_v2_v8_p7_p8_seed_collapse_diagnostic_one_cell.md`.
  - Script:
    `scripts/analyze_stage4_seed_collapse.py`.
  - 출력: default-threshold metric, validation calibrated threshold test
    metric, probability quantile, P7/P8 paired prediction comparison.
  - 결과: P8은 P7보다 ranking signal은 개선했지만 class decision collapse를
    해결하지 못했습니다. Validation-threshold calibration만으로도 충분하지
    않았습니다.
  - 준비 결과: [4-V8 Stage 4 v2 P7/P8 seed-collapse diagnostic](checklist_results/4-V8_stage4_v2_p7_p8_seed_collapse_diagnostic.md)
- [x] 4-V9. 우선순위 10: bounded last-block FiLM scale stability grid
  - 목적: news context로 넘어가기 전에, P8 collapse가 FiLM 자체의 실패인지
    bounded FiLM scale이 아직 큰 문제인지 분리해서 확인합니다.
  - 실행 wrapper:
    `notebooks/kaggle_stage4_v2_v9_bounded_last_block_film_scale_grid_one_cell.md`.
  - 고정 run: `I60/R20/ohlc_ma_vb`, F&G-only context,
    `film_full_bounded_last_block`, scales `0.02`, `0.05`, `0.10`, seeds
    `42, 43, 44, 45, 46`.
  - 이 단계에서는 checkpoint rule을 바꾸지 않습니다. V9는 validation/test
    collapse metric을 기록하지만, 실험 축은 scale 하나로 고정합니다.
  - 결과: scales `0.02`, `0.05`, `0.10` 모두 Stage 2 visual baseline보다
    낮았습니다. 낮은 scale은 일부 collapse를 줄였지만 seed `44`는 모든 scale에서
    mostly Down으로 무너졌습니다.
  - 판단: structured F&G-only FiLM은 계속 gamma/beta scale을 조정할 만큼
    robust하지 않습니다. 다음 external regime source로 news context를 진행합니다.
  - 준비 결과: [4-V9 Stage 4 v2 bounded last-block FiLM scale grid](checklist_results/4-V9_stage4_v2_bounded_last_block_film_scale_grid.md)

News-context 확장:
- [x] 4-N0. Numeric-context handoff와 news scope lock
  - V9 결론을 기록합니다: F&G-only numeric FiLM은 일부 ranking signal은 있지만
    Stage 2 `I60/R20/ohlc_ma_vb`를 robust하게 넘지 못했습니다.
  - 첫 news track은 headline-only, non-LLM, strict `t-1`로 고정합니다.
  - 더 풍부한 external context를 테스트하기 전까지 gamma/beta scale을 임의로
    계속 찾지 않습니다.
  - 준비 결과: [4-N0 Numeric-context handoff and news scope lock](checklist_results/4-N0_numeric_context_handoff_news_scope_lock.md)
- [x] 4-N1. `edaschau/bitcoin_news` source audit
  - Row count, date range, column, source distribution, title/article coverage,
    duplicate URL/title rate, day별 기사 수를 확인합니다.
  - BTC sample period와 five-seed Stage 4 selected sample universe와 겹치는지
    확인합니다.
  - 결과: headline-only 파일 `BTC_match_title.csv`는 `30,626` rows,
    `2011-06-22`부터 `2025-06-03`까지입니다. Selected Stage 4 sample은
    `2018-04-29`부터 `2024-12-11`까지입니다.
  - Preliminary source coverage: strict `t-1` sample coverage는 `96.04%`,
    trailing 7-day news coverage는 `100.00%`입니다. 4-N2에서 7/20/60-day
    window를 다시 확인합니다.
  - 결과: [4-N1 News source audit](checklist_results/4-N1_news_source_audit_design.md)
  - Table:
    [source audit](reports/tables/stage4_news_source_audit.json),
    [sample coverage](reports/tables/stage4_news_sample_coverage_by_split.csv),
    [source distribution](reports/tables/stage4_news_source_distribution.csv)
- [x] 4-N2. Publication-time alignment와 no-future-leakage rule
  - 기본 정책: chart image end date가 `t`이면 calendar date `t-1`까지의 뉴스만
    사용합니다.
  - BTC close cutoff와 news timestamp cutoff를 명확히 방어하기 전까지 same-day
    news는 쓰지 않습니다.
  - Split별 missing day/news count audit을 만듭니다.
  - 결과: strict `t-1` policy를 고정했습니다. Same-day headline이 존재하는
    `2,304 / 2,399`개 sample에서도 same-day news는 명시적으로 제외됩니다.
  - Coverage: train `96.57%`, validation `97.21%`, test `95.56%`; trailing
    7/20/60-day coverage는 모든 split에서 `100.00%`입니다.
  - Text vectorizer fit rule: train strict-`t-1` 7/20/60-day headline-window
    document에만 fit합니다 (`671` samples x `3` windows). Validation/test
    document는 transform-only입니다.
  - 결과: [4-N2 News publication-time alignment](checklist_results/4-N2_news_publication_time_alignment.md)
  - Table:
    [policy](reports/tables/stage4_news_alignment_policy.json),
    [by split](reports/tables/stage4_news_alignment_by_split.csv),
    [examples](reports/tables/stage4_news_alignment_examples.csv)
- [x] 4-N3. Headline-only headline-window aggregation table
  - 먼저 `title`, `date_time`, `source`, `url`만 사용합니다.
  - Exact duplicate URL/title을 제거합니다.
  - Trailing `7d`, `20d`, `60d`별로 leakage-safe sample-window field를 만듭니다:
    concatenated headline text, news count, optional source-count feature.
  - Full `article_text`와 summary는 뒤로 미룹니다.
  - 결과: raw headline rows `30,626` -> deduped rows `29,208`; duplicate
    normalized-title rows `1,418`개 제거.
  - 결과: train/validation/test 모두 7/20/60-day headline-window coverage
    `100.00%`.
  - Full aggregation table은
    `outputs/stage4/news/stage4_news_headline_windows_i60_r20/` 아래에 저장했습니다.
  - 결과: [4-N3 Headline-window aggregation](checklist_results/4-N3_headline_window_aggregation.md)
  - Table:
    [summary](reports/tables/stage4_news_headline_windows_summary.csv),
    [examples](reports/tables/stage4_news_headline_windows_examples.csv),
    [manifest](reports/tables/stage4_news_headline_windows_manifest.json)
- [ ] 4-N4. Train-only TF-IDF/SVD news vectorizer
  - Text preprocessing, TF-IDF vocabulary, IDF weight, SVD는 train-period
    news에만 fit합니다.
  - 첫 vector size는 window별 `news_svd_32`입니다:
    `news_svd_7d`, `news_svd_20d`, `news_svd_60d`.
  - 뉴스가 없는 날은 zero news vector와 명시적인 count feature를 사용합니다.
  - Vectorizer metadata, vocabulary hash, SVD dimension, train-period fit
    range를 저장합니다.
- [ ] 4-N5. BTC sample-level news context feature builder
  - Strict `t-1`로 daily news vector를 각 BTC image sample에 merge합니다.
  - 첫 context vector 후보:
    `news_svd_7d + news_svd_20d + news_svd_60d + news_count_7d/20d/60d`.
  - Text 외 count feature는 train-only statistics로 normalize합니다.
  - Chart image는 `I60/R20/ohlc_ma_vb` 그대로 유지합니다.
- [ ] 4-N6. News-context baseline controls
  - 같은 news-aligned sample set에서 Stage 2 visual-only baseline을 확인합니다.
  - 먼저 `CNN + news concat` five-seed를 실행합니다.
  - 목적: FiLM modulation을 주장하기 전에 news vector가 side information으로
    유용한지 확인합니다.
- [ ] 4-N7. News-context bounded FiLM main test
  - `CNN + news bounded last-block FiLM` five-seed를 실행합니다.
  - V9 교훈대로 visual path를 먼저 보호합니다.
  - Stage 2 visual baseline, `CNN + news concat`과 비교합니다.
- [ ] 4-N8. News + F&G combined-context ablation
  - News-only가 유용한 signal을 보일 때만 실행합니다.
  - 후보 vector: `news_svd_7d/20d/60d + news_count_7d/20d/60d + F&G-only`.
  - 목적: 더 풍부한 news context 옆에서 F&G가 compact regime summary로
    도움이 되는지 확인합니다.
- [ ] 4-N9. News interpretability report
  - Correct/incorrect Up/Down sample에 대해 Grad-CAM, news title,
    news-count feature, FiLM gamma/beta summary를 함께 export합니다.
  - Feature sensitivity: zero news vector, F&G 제거, news vector를 train mean으로
    대체하는 분석을 추가합니다.
- [ ] 4-N10. LLM summary/embedding decision
  - Headline-only no-leakage track이 안정화된 뒤로 미룹니다.
  - 사용한다면 model name, prompt, version/date, cache hash, runtime을
    기록해야 합니다.

중요:
- Main Stage 4 실험에서 context 값을 chart image 위에 추가로 그리지 않습니다.
- context는 별도 vector로 들어갑니다.
- 모든 context feature는 image end date `t` 또는 그 이전에 알 수 있어야 합니다.
- context normalization은 train split 통계로만 fit합니다.
- 뉴스는 이제 V9 이후 다음 Stage 4 context track입니다. V9에서 structured
  F&G-only numeric FiLM이 robust하지 않다는 결론이 나왔기 때문입니다.
