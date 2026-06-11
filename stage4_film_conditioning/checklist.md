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
- Current conclusion: structured numeric context and headline-news context can
  show useful signal in some seeds, but scratch-trained context/FiLM models do
  not robustly beat the Stage 2 visual baseline.
- Completed current track: `4-N8-A/B` through N13 context-source comparison and
  FiLM scale/freeze-policy ablations. The most defensible protocol reloads the
  selected Stage 2 checkpoint, reproduces the baseline, freezes the visual
  CNN/classifier, and trains only the context encoder plus bounded final-block
  FiLM heads.
- Completed current track: N15 image-spec context-complement checks. Technical
  replacement context mostly preserved same-image baselines, while F&G showed
  only tiny positive deltas on volume-aware image specs.
- Completed current track: N16 derivatives/leverage regime context. The
  strongest same-image positive case is `ohlc_vb + funding_plus_cftc_oi`,
  which improves the `ohlc_vb` Stage 2 baseline by `+0.002082` accuracy and
  net `+15` corrections.
- First news version: headline-only, non-LLM, train-only TF-IDF/SVD over
  7/20/60-day trailing news windows.
- Main order now: Stage 4 modeling/context-source experiments are complete
  through FSI/RORO, derivatives/leverage, N14 final interpretation, and N14-B
  conditional-regime analysis. N14-B is used as supporting evidence for the
  N16 same-image derivatives/leverage case, not as a new model branch.

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
- [x] 4-I14. Stage 4 result report
  - Numeric-context reporting is complete through `4-V9`.
  - Final reporting is merged into:
    [4-N14 final Stage 4 interpretability report](checklist_results/4-N14_final_stage4_interpretability_report.md).

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
- [x] 4-N4. Train-only TF-IDF/SVD news vectorizer
  - Fit text preprocessing, TF-IDF vocabulary, IDF weights, and SVD only on
    train-period news.
  - First vector size: `news_svd_32` per window, producing `news_svd_7d`,
    `news_svd_20d`, and `news_svd_60d`.
  - No-news days use a zero news vector plus explicit count features.
  - Save vectorizer metadata, vocabulary hash, SVD dimension, and train-period
    fit range.
  - Result: TF-IDF/SVD fit on `2,013` train documents (`671` samples x
    `7/20/60` windows), vocabulary size `10,000`, SVD dim `32`, explained
    variance ratio sum `0.5856`.
  - Full vector artifact:
    `outputs/stage4/news/stage4_news_tfidf_svd_i60_r20/news_tfidf_svd_features.parquet`.
  - Result: [4-N4 Train-only TF-IDF/SVD news vectorizer](checklist_results/4-N4_news_tfidf_svd_vectorizer.md)
  - Tables:
    [manifest](reports/tables/stage4_news_tfidf_svd_manifest.json),
    [summary](reports/tables/stage4_news_tfidf_svd_summary.csv),
    [top terms](reports/tables/stage4_news_tfidf_svd_top_terms.csv)
- [x] 4-N5. BTC sample-level news context feature builder
  - Merge daily news vectors into each BTC image sample using strict `t-1`.
  - First context vector:
    `news_svd_7d + news_svd_20d + news_svd_60d` plus log-count features.
  - Result: `102` normalized features = `96` SVD features plus `6` log-count
    features.
  - Normalization uses train-only median imputation, train quantile clipping,
    and train z-score scaling.
  - Keep the chart image unchanged: `I60/R20/ohlc_ma_vb`.
  - Result: [4-N5 News context feature builder](checklist_results/4-N5_news_context_feature_builder.md)
  - Tables:
    [audit](reports/tables/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60_seed42_news_context_feature_audit.json),
    [summary](reports/tables/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60_seed42_news_context_feature_summary.csv),
    [manifest](reports/tables/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60_seed42_news_context_manifest.json)
- [x] 4-N6. News-context baseline controls
  - Reference visual-only baseline: Stage 2 selected five-seed
    `I60/R20/ohlc_ma_vb`.
  - N5 showed the news-aligned context table keeps the same sample universe:
    train `671`, validation `287`, test `1,441`.
  - Run `CNN + news concat` five-seed first.
  - Purpose: test whether the news vector is useful as side information before
    claiming FiLM modulation is useful.
  - Prepared notebook:
    [kaggle_stage4_news_context_n6_baseline_controls_one_cell.md](notebooks/kaggle_stage4_news_context_n6_baseline_controls_one_cell.md)
  - Result note:
    [4-N6 News-context baseline controls](checklist_results/4-N6_news_context_baseline_controls.md)
  - Kaggle result: accuracy mean `0.5478`, ROC-AUC mean `0.5644`.
  - Diagnosis: seeds `43` and `45` collapsed to near one-sided predictions, so
    the `102`-dimensional news context is not stable enough for direct N7.
- [x] 4-N6.1. News SVD-dimension stability grid
  - Test smaller train-only TF-IDF/SVD dimensions before adding FiLM.
  - Grid: SVD dim `16` and `8`, which produce context dims `54` and `30`.
  - Fixed model: `I60/R20/ohlc_ma_vb` + `CNN + news concat`.
  - Seeds: `42, 43, 44, 45, 46`.
  - Purpose: check whether lower-dimensional headline vectors reduce seed
    collapse while preserving the useful news ranking signal.
  - Prepared notebook:
    [kaggle_stage4_news_context_n6_svd_dim_grid_one_cell.md](notebooks/kaggle_stage4_news_context_n6_svd_dim_grid_one_cell.md)
  - Prepared result note:
    [4-N6.1 News SVD-dim grid](checklist_results/4-N6.1_news_svd_dim_grid.md)
  - Kaggle result: SVD8 accuracy mean `0.5407`, ROC-AUC mean `0.5817`;
    SVD16 accuracy mean `0.5348`, ROC-AUC mean `0.5608`.
  - Decision: use SVD8 for N7 because it preserves the strongest news ranking
    signal and keeps the FiLM input small.
- [x] 4-N7. News-context bounded FiLM main test
  - Run `CNN + news bounded last-block FiLM` five-seed with SVD8 news context.
  - Start from the conservative V9 lesson: protect the visual path first.
  - Use `modulation_scale=0.05`:
    `gamma = 1 + 0.05 * tanh(raw_gamma)`,
    `beta = 0.05 * tanh(raw_beta)`.
  - Compare against Stage 2 visual baseline and `CNN + news concat`.
  - Result: five-seed accuracy mean `0.5591`, ROC-AUC mean `0.5642`,
    predicted-Up-rate mean `0.6952`.
  - Interpretation: N7 reduced seed collapse versus news concat SVD8 but did
    not beat the Stage 2 visual baseline. The model still trains the Stage 2
    architecture from scratch, so it does not yet test the intended
    pretrained-baseline-preserving FiLM idea.
  - Prepared notebook:
    [kaggle_stage4_news_context_n7_bounded_film_svd8_one_cell.md](notebooks/kaggle_stage4_news_context_n7_bounded_film_svd8_one_cell.md)
  - Prepared result note:
    [4-N7 News bounded FiLM SVD8](checklist_results/4-N7_news_bounded_film_svd8.md)
- [x] 4-N8. Stage 2 pretrained baseline-preserving FiLM
  - First substep: Stage 2 checkpoint reload sanity. Load the selected Stage 2
    `I60/R20/ohlc_ma_vb` learned weights inside the Stage 4 code path and
    verify that context-free predictions reproduce the Stage 2 baseline.
  - 4-N8-A1 reload sanity passed locally with the rebuilt Stage 2 checkpoint
    bundle. Stage4-side reload reproduced the five-seed Stage 2 baseline:
    accuracy mean `0.579320`, ROC-AUC mean `0.584863`; classification metrics
    matched the bundle within tolerance.
  - Second substep: freeze the Stage 2 visual CNN and train only the context
    encoder plus bounded last-block FiLM heads. Start with F&G-only and news
    SVD8-only before combining context sources.
  - 4-N8-B implementation completed: Stage 2 checkpoint load, visual-backbone
    freeze, classifier freeze, frozen BatchNorm/dropout eval mode, and
    trainable context encoder plus bounded final-block FiLM heads.
  - Local smoke passed for F&G-only, seed42, scale `0.05`, 64-row train/val/test.
    Loaded Stage 2 keys: `30`; frozen parameters: `2,952,962`; trainable
    parameters: `35,008`.
  - Full N8-B F&G-only Kaggle five-seed run completed for scales `0.02` and
    `0.05`.
  - Result: scale `0.02` accuracy mean `0.580291`, ROC-AUC mean `0.584930`;
    scale `0.05` accuracy mean `0.579320`, ROC-AUC mean `0.584921`.
  - Interpretation: N8-B does not materially beat the Stage 2 baseline, but it
    preserves the baseline and avoids the severe scratch-FiLM seed collapse.
  - Optional substep if needed: keep the CNN frozen but allow the classifier to
    train; only then consider unfreezing the final CNN block.
  - Purpose: test whether market/news context can improve an already strong
    visual model by bounded correction instead of retraining a new CNN from
    scratch.
  - Prepared next-step note:
    [4-N8 Pretrained baseline-preserving FiLM](checklist_results/4-N8_pretrained_baseline_preserving_film.md)
  - Reload sanity script:
    [check_stage4_n8_stage2_checkpoint_reload.py](scripts/check_stage4_n8_stage2_checkpoint_reload.py)
  - N8-B Kaggle runner:
    [kaggle_stage4_n8b_fg_only_pretrained_frozen_bounded_film_one_cell.md](notebooks/kaggle_stage4_n8b_fg_only_pretrained_frozen_bounded_film_one_cell.md)
- [x] 4-N9. News-only and News + F&G pretrained/frozen ablation
  - Run after `4-N8` showed that baseline-preserving context-FiLM can reproduce
    and safely modify the Stage 2 baseline.
  - Gamma/beta rule: do not manually set gamma/beta per sample. The model learns
    `context -> MLP -> gamma/beta`; the experimenter only controls the context
    vector, freeze policy, insertion point, and bounded modulation scale.
  - N9-A weak correction completed: news SVD8-only, CNN frozen, classifier
    frozen, scale `0.02`, five seeds. Result: accuracy mean `0.579459`,
    ROC-AUC mean `0.585670`; stable but only a very small correction versus the
    Stage 2 baseline.
  - N9-B weak correction: news SVD8-only, CNN frozen, classifier frozen, scale
    `0.05`, only if N9-A is stable but too conservative.
  - N9-C medium correction: news SVD8-only, CNN frozen, classifier trainable,
    scale `0.02`.
  - N9-D medium correction: news SVD8-only, CNN frozen, classifier trainable,
    scale `0.05`, only if N9-C is stable but too weak.
  - N9-E combined context: `news_svd_7d/20d/60d + news_count_7d/20d/60d +
    F&G-only`, scale `0.02`, only if news-only is promising or needed for the
    advisor-facing final comparison.
  - Purpose: test whether richer external news context provides incremental
    signal under the stable N8-B structure.
  - Kaggle runner prepared:
    [kaggle_stage4_n9_news_pretrained_frozen_bounded_film_one_cell.md](notebooks/kaggle_stage4_n9_news_pretrained_frozen_bounded_film_one_cell.md)
  - Default run: `N9A`, news SVD8-only, Stage 2 CNN/classifier frozen,
    bounded last-block FiLM scale `0.02`, five seeds.
  - SVD/scale grid runner prepared:
    [kaggle_stage4_n9_news_pretrained_frozen_svd_scale_grid_one_cell.md](notebooks/kaggle_stage4_n9_news_pretrained_frozen_svd_scale_grid_one_cell.md)
  - Grid points: `SVD8/0.05`, `SVD16/0.02`, `SVD16/0.05`, `SVD32/0.02`,
    `SVD32/0.05`. `SVD8/0.02` is excluded because N9-A already ran it.
  - Grid purpose: test whether N9-A was too conservative because FiLM scale was
    too small or SVD8 compressed the headline context too aggressively.
  - To run older single-variant follow-ups, change `N9_VARIANT` at the top of
    the bounded-FiLM cell: `N9B`, `N9C`, or `N9D`.
  - Design note:
    [4-N9 News pretrained/frozen FiLM design](checklist_results/4-N9_news_pretrained_frozen_film_design.md)
  - Closeout: news-only pretrained/frozen runs and grid diagnostics are
    complete for the current Stage 4 scope. `news + F&G` remains explicitly
    not-run/deferred and is not claimed as a result.
- [x] 4-N10. News interpretability report
  - First export Stage 2 baseline vs N8-B F&G-only Grad-CAM on matched samples.
  - Then export news titles, news-count features, and FiLM gamma/beta summaries
    for correct/incorrect Up/Down samples if 4-N9 is run.
  - Add feature sensitivity: zero news vector, remove F&G, or replace context
    vector with train mean.
  - Primary interpretability target: `Stage 2 wrong -> N8/N9 correct` samples,
    because these show whether context-FiLM corrected a visual-baseline error.
  - Initial report completed from available artifacts:
    [4-N10 News interpretability report](checklist_results/4-N10_news_interpretability_report.md)
  - Selected N9 grid-best Grad-CAM export cell prepared:
    [kaggle_stage4_n10_selected_news_interpretability_one_cell.md](notebooks/kaggle_stage4_n10_selected_news_interpretability_one_cell.md)
  - Current finding: N8/N9 bounded FiLM preserves the Stage 2 baseline and
    slightly improves ROC-AUC/calibration/reduces Up-bias, but it does not yet
    produce a defensible accuracy gain.
  - Limitation: the N9 SVD/scale grid bundle is metric-only. Targeted
    `Stage 2 wrong -> N9 correct` Grad-CAM requires an additional selected
    export for the best grid candidate, primarily `SVD32/scale0.02`.
- [x] 4-N10-A. Stage 2 vs N10 correction-analysis code
  - Added a prediction-level comparison script:
    [analyze_stage4_stage2_context_corrections.py](scripts/analyze_stage4_stage2_context_corrections.py)
  - Added a Kaggle one-cell runner:
    [kaggle_stage4_n10_stage2_vs_n10_correction_analysis_one_cell.md](notebooks/kaggle_stage4_n10_stage2_vs_n10_correction_analysis_one_cell.md)
  - Purpose: export `Stage2 wrong -> N10 correct`,
    `Stage2 correct -> N10 wrong`, transition summaries, and selected
    sample-index lists for targeted Grad-CAM/gamma-beta/news interpretation.
- [x] 4-N10-B. Targeted Grad-CAM + gamma/beta modulation export code
  - Added targeted sample support to Stage 2 and Stage 4 Grad-CAM exporters.
  - Added a Kaggle one-cell runner:
    [kaggle_stage4_n10_b_targeted_gradcam_modulation_one_cell.md](notebooks/kaggle_stage4_n10_b_targeted_gradcam_modulation_one_cell.md)
  - Purpose: use the N10-A selected `sample_index` list to export matched
    Stage 2 vs N10 Grad-CAM and N10 FiLM gamma/beta modulation metadata.
  - Design note:
    [4-N10-B targeted Grad-CAM modulation export](checklist_results/4-N10-B_targeted_gradcam_modulation_export.md)
- [x] 4-N11. LLM summary/embedding decision
  - Deferred until headline-only no-leakage track is stable.
  - If used, record model name, prompt, version/date, cache hash, and runtime.
  - Closeout decision:
    [4-N11 LLM embedding deferred](checklist_results/4-N11_llm_embedding_deferred.md).
- [x] 4-N12. Optional uncertainty-gated FiLM follow-up
  - Run only after N9/N10 interpretation shows that context helps mainly when
    the Stage 2 chart model is uncertain.
  - Idea: let context-FiLM correction become stronger for ambiguous Stage 2
    decisions and weaker for high-confidence chart decisions.
  - Candidate uncertainty:
    `uncertainty = 4 * prob_up_stage2 * (1 - prob_up_stage2)`.
  - Candidate formula:
    `gamma = 1 + uncertainty * scale * tanh(raw_gamma)`,
    `beta = uncertainty * scale * tanh(raw_beta)`.
  - Purpose: test the thesis-friendly claim that news/F&G context is most useful
    as a correction signal when the visual chart evidence is ambiguous.
- [x] 4-N12-A. Uncertainty-gated news FiLM implementation and runner
  - Added `film_full_uncertainty_gated_last_block`.
  - Formula:
    `uncertainty = 4 * prob_up_stage2 * (1 - prob_up_stage2)`.
  - Same N8/N9/N10 preservation rule: load the Stage 2
    `I60/R20/ohlc_ma_vb` checkpoint, freeze CNN/classifier, and train only the
    news context encoder plus final-block FiLM heads.
  - Kaggle runner:
    [kaggle_stage4_n12a_uncertainty_gated_news_film_one_cell.md](notebooks/kaggle_stage4_n12a_uncertainty_gated_news_film_one_cell.md)
  - Default grid: news SVD32, scales `0.02` and `0.05`, five seeds.
  - Local shape check passed; result note:
    [4-N12-A uncertainty-gated news FiLM](checklist_results/4-N12-A_uncertainty_gated_news_film.md)
- [x] 4-N12-B. Confidence-gated news FiLM implementation and runner
  - Added `film_full_confidence_gated_last_block`.
  - Formula:
    `confidence = abs(2 * stage2_prob_up - 1)`.
  - Same baseline-preserving rule: load the Stage 2 `I60/R20/ohlc_ma_vb`
    checkpoint, freeze CNN/classifier, and train only the news context encoder
    plus final-block FiLM heads.
  - Kaggle runner:
    [kaggle_stage4_n12b_confidence_gated_news_film_one_cell.md](notebooks/kaggle_stage4_n12b_confidence_gated_news_film_one_cell.md)
  - Default grid: news SVD32, scales `0.02` and `0.05`, five seeds.
  - Local shape check passed; result note:
    [4-N12-B confidence-gated news FiLM](checklist_results/4-N12-B_confidence_gated_news_film.md)
- [x] 4-N12-C. Stage 2 frozen + technical-only bounded FiLM
  - Purpose: separate image-derived technical context from external/news context
    under the same baseline-preserving Stage 2 frozen protocol.
  - Candidate features: `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`,
    `rv_60`.
  - Method: `film_full_bounded_last_block` first; gated variants only if the
    bounded result shows useful signal.
  - Required comparison: Stage 2 baseline, N8-B F&G-only, N9/N10 news-only,
    N12-A/B gated news.
  - Kaggle runner:
    [kaggle_stage4_n12c_technical_only_pretrained_frozen_bounded_film_one_cell.md](notebooks/kaggle_stage4_n12c_technical_only_pretrained_frozen_bounded_film_one_cell.md)
  - Result: scale `0.02` accuracy mean `0.579736`, ROC-AUC mean `0.584778`.
    This is effectively tied with the Stage 2 frozen baseline rather than a
    meaningful improvement.
  - Result note:
    [4-N12-C technical-only pretrained frozen bounded FiLM](checklist_results/4-N12-C_technical_only_pretrained_frozen_bounded_film.md)
- [x] 4-N12-D. Context-source comparison under the frozen Stage 2 protocol
  - Purpose: decide which context source is thesis-defensible rather than
    continuing one-off variants.
  - Compare with the same image, split, checkpoint loading, freeze policy, and
    bounded/gated FiLM protocol:
    `F&G-only`, `news-only`, `technical-only`, `news + F&G`.
  - Required metrics: accuracy, ROC-AUC, Brier score, F1, predicted-Up rate,
    correction count, regression count, net correction.
  - Output: compact comparison table plus recommendation for the final Stage 4
    model.
  - Result: completed for existing context sources. F&G-only scale `0.02` is the
    best compact accuracy candidate (`0.580291` vs Stage 2 `0.579320`), news
    gives the clearest ROC-AUC/Brier signal but weaker hard decisions, and
    technical-only is effectively tied with Stage 2.
  - Caveat: `news + F&G` combined context is recorded as planned/not-run in the
    comparison table; do not claim a result for it until a five-seed run exists.
  - Result note:
    [4-N12-D context-source comparison](checklist_results/4-N12-D_context_source_comparison.md)
- [x] 4-N13. Macro/RORO context extension
  - Purpose: test image-external macro risk-regime context after F&G/news and
    technical context produced only tiny gains.
  - Thesis question: can official financial stress or risk-on/risk-off regime
    condition the frozen Stage 2 BTC chart features more meaningfully than
    context derived from BTC OHLCV?
  - Shared protocol: `I60/R20/ohlc_ma_vb`, five seeds `42-46`, Stage 2
    checkpoints loaded/frozen, classifier frozen, bounded last-block FiLM first
    with conservative scale `0.02`.
- [x] 4-N13-0. Macro/RORO source audit and terminology lock
  - Distinguish `OFR FSI` from `RORO`: OFR FSI is not a direct RORO index; it is
    an official financial-stress/risk-off proxy.
  - Record source links, date coverage, CSV load path, missing-date policy, and
    whether each feature is available at or before image end date `t`.
  - Candidate source 1: OFR Financial Stress Index CSV, covering 2000-present.
  - Candidate source 2: public-data RORO proxy inspired by KC Fed methodology,
    built from FRED/Cboe indicators rather than proprietary/refinitiv series.
  - Closeout: terminology/source audit is documented in
    [4-N13-1 OFR FSI feature builder](checklist_results/4-N13-1_ofr_fsi_feature_builder.md)
    and
    [4-N13-3 public RORO proxy builder](checklist_results/4-N13-3_public_roro_proxy_builder.md).
- [x] 4-N13-1. OFR FSI feature builder
  - Raw source: `https://www.financialresearch.gov/financial-stress-index/data/fsi.csv`.
  - Interpretation: higher `OFR FSI` = stronger financial stress = risk-off
    proxy. Do not hard-code that BTC must fall; let FiLM learn the relation.
  - Candidate features: `ofr_fsi_value`, `ofr_fsi_mean_20`,
    `ofr_fsi_mean_60`, `ofr_fsi_delta_20`, `ofr_fsi_delta_60`,
    `ofr_fsi_std_60`, plus optional category values `Credit`,
    `Equity valuation`, `Funding`, `Safe assets`, `Volatility`.
  - Normalize with train-only imputation, clipping, and z-score statistics.
  - Prepared script:
    [build_stage4_fsi_context_features.py](scripts/build_stage4_fsi_context_features.py)
  - Kaggle one-cell:
    [kaggle_stage4_n13_1_ofr_fsi_context_features_one_cell.md](notebooks/kaggle_stage4_n13_1_ofr_fsi_context_features_one_cell.md)
  - Latest upload zip:
    `/Users/jaehyeonjeong/Desktop/논문/stage4_film_conditioning_n13_1_latest.zip`.
  - Prep note:
    [4-N13-1 OFR FSI feature builder](checklist_results/4-N13-1_ofr_fsi_feature_builder.md)
  - Result: completed with six FSI features, `context_dim=6`,
    train/validation/test split counts `671/287/1441`, and zero missing rate
    across all six FSI features after source-level rolling feature generation.
  - Feature screening result:
    [4-N13-1 OFR FSI feature screening](checklist_results/4-N13-1_fsi_feature_screening.md).
    Use `FSI-2 = mean_60 + delta_60`, `FSI-3 = mean_60 + delta_60 + std_60`,
    and `FSI-all` in the next frozen-FiLM run instead of assuming all six
    features are optimal.
- [x] 4-N13-2. FSI-only frozen bounded FiLM five-seed run
  - Context source: OFR FSI features only.
  - Kaggle one-cell:
    [kaggle_stage4_n13_2_fsi_only_pretrained_frozen_bounded_film_one_cell.md](notebooks/kaggle_stage4_n13_2_fsi_only_pretrained_frozen_bounded_film_one_cell.md)
  - Feature-set grid: `fsi_2`, `fsi_3`, `fsi_all`.
  - Main comparison: Stage 2 frozen baseline, N8-B F&G-only, N10/N12 news-only,
    and N12-C technical-only.
  - Required metrics: accuracy, ROC-AUC, Brier, F1, predicted-Up rate,
    correction/regression/net correction, seed-level collapse check.
  - Result:
    [4-N13-2 FSI-only frozen bounded FiLM](checklist_results/4-N13-2_fsi_only_pretrained_frozen_bounded_film.md).
    Best FSI row is `fsi_all`, accuracy `0.579875`, ROC-AUC `0.584859`,
    net correction `+4` total over five seeds, and zero collapse warnings.
    This is stable but not materially better than Stage 2 or N8-B F&G-only.
- [x] 4-N13-3. KC Fed-inspired public-data RORO proxy builder
  - Raw sources: VIX, S&P500/NASDAQ returns, Broad Dollar Index, US 10Y yield,
    optional high-yield OAS and gold.
  - Direction rule: align features so positive values mean risk-off pressure.
  - Candidate proxy: train-fit PCA first component on risk-off-aligned daily
    changes/returns; keep raw components beside the synthetic score.
  - Document explicitly that this is a public-data RORO proxy, not a replication
    of the KC Fed proprietary/full input set.
  - Implemented:
    [build_stage4_roro_context_features.py](scripts/build_stage4_roro_context_features.py)
    and
    [kaggle_stage4_n13_3_public_roro_context_features_one_cell.md](notebooks/kaggle_stage4_n13_3_public_roro_context_features_one_cell.md).
  - Source audit:
    [4-N13-3 public RORO proxy builder](checklist_results/4-N13-3_public_roro_proxy_builder.md).
    KC Fed official daily/weekly files are cached for documentation, but the
    downloaded files start in June 2023 and do not cover Stage 4 train dates;
    the trainable proxy therefore uses longer-history cached public inputs.
  - Result: local N13-3 artifact was created successfully with VIX, S&P500,
    DXY, and US 10Y components. `context_dim=10`, PCA explained variance ratio
    `0.554831`, split counts `671/287/1441`, and missing warnings `0`.
  - Implemented formula:
    `PC1_train_only(z(VIX_t - VIX_{t-20}), z(-log(SP500_t/SP500_{t-20})), z(log(DXY_t/DXY_{t-20})), z(-(DGS10_t - DGS10_{t-20})))`.
    The sign is fixed so a larger value means stronger risk-off pressure.
  - Cached raw inputs:
    `data_inventory/roro_public/raw/VIXCLS.csv`,
    `data_inventory/roro_public/raw/SP500.csv`,
    `data_inventory/roro_public/raw/DXY_yahoo_DX-Y.NYB.csv`,
    `data_inventory/roro_public/raw/DGS10.csv`.
    `BAMLH0A0HYM2.csv` is cached but excluded from PCA because it lacks
    train-period coverage.
  - Exclusion note: HYG/high-yield ETF price is not used in this N13-3 version
    because it is not HY OAS and would mix ETF price dynamics into the
    credit-risk proxy.
- [x] 4-N13-4. RORO-proxy-only frozen bounded FiLM five-seed run
  - Context source: public-data RORO proxy features only.
  - Same protocol and metrics as 4-N13-2.
  - Prepared Kaggle runner:
    `notebooks/kaggle_stage4_n13_4_roro_only_pretrained_frozen_bounded_film_one_cell.md`.
  - Recommended upload bundle:
    `stage4_film_conditioning_n13_4_with_stage2_bundle.zip`, which embeds the
    Stage 2 I60/R20/ohlc_ma_vb seed 42-46 checkpoint bundle to avoid Kaggle
    reset/path issues.
  - Result: completed over 3 RORO feature sets x 5 seeds. No collapse warning.
    Best accuracy row is `roro_3`, accuracy `0.579320`, ROC-AUC `0.584748`,
    Brier `0.274278`, F1 `0.650924`; this effectively ties Stage 2 and is
    weaker than the best F&G-only row.
  - Review:
    [4-N13-4 RORO-only frozen bounded FiLM](checklist_results/4-N13-4_roro_proxy_only_pretrained_frozen_bounded_film.md).
- [x] 4-N13-5. Macro context-source comparison
  - Compare `FSI-only`, `RORO-proxy-only`, `F&G-only`, `news-only`,
    `technical-only`.
  - Select one candidate for final Stage 4 interpretation only if it improves
    either accuracy or ROC/Brier without class-collapse.
  - Result: completed. F&G-only scale `0.02` remains the best compact accuracy
    candidate (`0.580291`, +`0.000972` vs Stage 2). News SVD32 scale `0.02`
    remains the best interpretability/calibration candidate. FSI/RORO are stable
    but not strong enough alone.
  - Review:
    [4-N13-5 macro context-source comparison](checklist_results/4-N13-5_macro_context_source_comparison.md).
- [x] 4-N13-5A. Cross-context feature audit
  - Merge already-built context features on the same sample/date index:
    F&G, news SVD/count, technical BB/MFI/RV, OFR FSI, public RORO, label,
    future return, Stage 2 `prob_up`, and Stage 2 `correct`.
  - Use train split only for feature selection diagnostics. Validation/test are
    for confirmation only.
  - Audit: missing rate, feature-label correlation, feature-future-return
    correlation, feature-Stage2-error correlation, feature-feature correlation,
    and redundancy clusters.
  - Result: completed on 2,399 aligned samples and 126 context features. News
    SVD dimensions show the strongest train-only signal, F&G remains the
    cleanest compact regime source, and FSI/RORO are stable but weaker alone.
    Stage 2 error-rate correlations are weak across families, so selected-combo
    FiLM should stay small and conservative.
  - Review:
    [4-N13-5A cross-context feature audit](checklist_results/4-N13-5A_cross_context_feature_audit.md).
- [x] 4-N13-5B. Selected-combo context FiLM
  - Run one controlled selected-combo context experiment only if 4-N13-5A finds
    a non-redundant feature set.
  - Candidate size: roughly 6 features.
  - Primary candidate: `news_svd_60d_09`, `news_svd_20d_18`, `fg_mean_60`,
    `fg_delta_60`, `ofr_fsi_std_60`, `riskoff_dollar_return_20`.
  - Optional technical add-on: `bb_bandwidth_60`.
  - Stage 2 frozen protocol, bounded last-block FiLM, five seeds, scale `0.02`
    and optionally `0.05`.
  - Decision rule: keep only if it improves or clarifies the source comparison
    without reducing interpretability.
  - Result: completed as a metric-only five-seed run. Mean accuracy tied the
    Stage 2 frozen baseline (`0.579320`), ROC-AUC changed only by `+0.000004`,
    Brier improved slightly by `-0.000202`, and net correction was `0.0`.
    The raw table has `status=failed` only because Grad-CAM was disabled while
    the previous output checker still required Grad-CAM artifacts.
  - Conclusion: stable but not a final performance candidate. The six-feature
    selected combo mostly preserves Stage 2 decisions, so larger all-context
    stacking is not justified.
  - Review:
    [4-N13-5B selected-combo context FiLM](checklist_results/4-N13-5B_selected_combo_context_film.md).
- [x] 4-N13-6. Macro interpretability export
  - Focus on the strongest compact candidates rather than the selected combo:
    F&G-only `s0.02` and news SVD32 `s0.02`.
  - Target samples: Stage 2 wrong -> context-FiLM correct, Stage 2 correct ->
    context-FiLM wrong, and extreme regime/news windows.
  - Export targeted Grad-CAM, context values, gamma/beta summaries,
    modulation gate if used, and `prob_up` changes.
  - Preparation: N13-6 Kaggle one-cell runner is ready. It generates
    candidate-specific correction/regression tables, augments them with
    extreme-context panels, exports matched Stage 2 vs context-FiLM Grad-CAM,
    and writes one downloadable bundle.
  - Review:
    [4-N13-6 interpretability export](checklist_results/4-N13-6_interpretability_export.md).
- [x] 4-N13-7. Final FiLM constraint and scale ablation on the selected context
  source
  - Run only after 4-N13-5/6 select the strongest stable context source
    (`F&G`, `news`, `FSI`, or `RORO`).
  - Purpose: test whether the current bounded FiLM was too conservative under
    the Stage 2 frozen protocol.
  - Baseline-preserving fixed setup: Stage 2 I60/R20/ohlc_ma_vb visual CNN and
    classifier frozen, same split, same seeds, same selected context features.
  - A. Same bounded equation, larger scale, no new model code required:
    bounded last-block FiLM scale `0.02`, `0.05`, `0.10`, `0.20`.
    - N8-B already covers `0.02` and `0.05`.
    - N13-7A runner is prepared for `0.10` and `0.20`:
      [kaggle_stage4_n13_7a_fg_bounded_scale_grid_one_cell.md](notebooks/kaggle_stage4_n13_7a_fg_bounded_scale_grid_one_cell.md).
    - Prep note:
      [4-N13-7A F&G bounded FiLM scale grid](checklist_results/4-N13-7A_fg_bounded_scale_grid.md).
    - Result: completed. `0.10` accuracy mean `0.579042`, ROC-AUC
      `0.584811`, net correction `-2`; `0.20` accuracy mean `0.578487`,
      ROC-AUC `0.584539`, net correction `-6`. Larger scale did not improve
      N8-B `0.02`, so keep small-scale F&G as the stronger bounded setting.
  - B. Relax gamma/beta constraint:
    unbounded or weakly regularized last-block FiLM with zero-init
    `gamma/beta` heads.
  - C. Alternative gamma/beta equation:
    compare the current `1 + scale * tanh(raw)` rule against a second
    baseline-preserving parameterization such as positive-gamma sigmoid/softplus
    or regularized residual-linear FiLM.
  - D. Classifier-unfreeze variant:
    keep the Stage 2 visual CNN frozen but unfreeze the final classifier,
    training only classifier plus context encoder/FiLM heads.
    - N13-7D runner is prepared with F&G-only `scale=0.02`:
      [kaggle_stage4_n13_7d_fg_classifier_unfreeze_one_cell.md](notebooks/kaggle_stage4_n13_7d_fg_classifier_unfreeze_one_cell.md).
    - Prep note:
      [4-N13-7D F&G classifier-unfreeze FiLM](checklist_results/4-N13-7D_fg_classifier_unfreeze.md).
    - Result: completed. Classifier-unfreeze accuracy mean `0.574323`,
      ROC-AUC `0.584220`, Brier `0.280218`, net correction `-36`.
      The classifier became too flexible and produced more regressions than
      corrections, so N8-B `scale=0.02` remains the best F&G setting.
  - Implementation rule: B/C/D require separate implementation and must be
    reported as FiLM/freeze-policy ablations, not as new context sources.
  - Required metrics: accuracy, ROC-AUC, Brier score, predicted-positive rate,
    collapse warning, correction/regression vs Stage 2, net correction, and
    gamma/beta magnitude summaries.
  - Decision rule: keep the larger-scale or relaxed-constraint model only if it
    improves at least one main metric without class-collapse or a large
    regression increase.
  - If none improves the overall metrics, use the same outputs for conditional
    regime analysis: extreme context regimes, high-volatility/high-stress
    regimes, and Stage 2 wrong -> FiLM correct samples.
  - Closeout: A and D were executed and both were weaker than the conservative
    frozen bounded setup. B/C are deferred because the executed relaxation tests
    already showed larger FiLM freedom increasing regressions rather than
    improving the baseline.
- [x] 4-N13-B. Optional sentiment/event feature extension
  - Decision: moved out of Stage 4 and into Stage 5.
  - Reason: Stage 4 has closed the non-LLM context cycle through numeric,
    TF-IDF/SVD news, FSI/RORO, derivatives/leverage, and image-spec complement
    checks. FinBERT/GPT/Claude sentiment/event features are now tracked in
    `stage5_llm_news_embedding`.
  - Stage 4 status: no longer a blocker for final Stage 4 reporting.
- [x] 4-N15. I60/R20 image-spec context-complement ablation
  - Purpose: test whether context-FiLM helps more when the chart image is
    missing information that is present in richer image specs.
  - Fixed setup: `I60/R20`, image specs `ohlc`, `ohlc_ma`, `ohlc_vb`,
    `ohlc_ma_vb`, seed-matched Stage 2 checkpoint for each image spec, frozen
    visual CNN/classifier, bounded final-block FiLM, scale `0.02`, seeds
    `42,43,44,45,46`.
  - Stage 2 reference: `ohlc_ma_vb` accuracy `0.579320`, `ohlc_vb`
    `0.567384`, `ohlc` `0.558085`, `ohlc_ma` `0.557529`.
  - N15-A. Same-image Stage 2 reload for all four image specs.
    - Establish the exact frozen baseline for every image spec before adding
      context.
    - Important: each image spec must use its own Stage 2 checkpoint. Do not
      reuse the `ohlc_ma_vb` checkpoint for other image specs.
    - Result check: completed. The N15-A mean/std table exactly reproduces the
      existing Stage 2 I60/R20 four-image five-seed results; observed deltas are
      only floating-point noise (`~1e-16`).
    - Local reusable bundle:
      `/Users/jaehyeonjeong/Desktop/논문/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15`.
    - Runner:
      [kaggle_stage4_n15a_rebuild_i60_r20_four_image_stage2_checkpoints_one_cell.md](notebooks/kaggle_stage4_n15a_rebuild_i60_r20_four_image_stage2_checkpoints_one_cell.md).
    - Prep note:
      [4-N15-A I60/R20 four-image Stage 2 checkpoint bundle](checklist_results/4-N15-A_i60_r20_stage2_four_image_checkpoint_bundle.md).
  - N15-B. Image-missing-feature complement FiLM.
    - This comes before F&G-across-images.
    - `ohlc + technical_all`: `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`,
      `rv_60`.
    - `ohlc_ma + volume_volatility`: `mfi_60`, `rv_60`.
    - `ohlc_vb + bb_trend`: `bb_percent_b_60`, `bb_bandwidth_60`.
    - `ohlc_ma_vb + technical_all_control`: same technical vector, expected to
      be neutral if technical context is redundant with the full image.
    - Question: can context-FiLM partially replace visual information that was
      not drawn into the image?
    - Result: completed. All 20 runs finished, but changes were tiny. Accuracy
      deltas versus same-image Stage 2 were `0.000000` for `ohlc`,
      `+0.000139` for `ohlc_ma`, `+0.000139` for `ohlc_vb`, and `+0.000416`
      for `ohlc_ma_vb`. Changed-decision rates were only about `0.2%`-`0.4%`,
      so the technical context vectors did not materially replace missing
      visual information.
    - Runner:
      [kaggle_stage4_n15b_image_missing_feature_complement_film_one_cell.md](notebooks/kaggle_stage4_n15b_image_missing_feature_complement_film_one_cell.md).
    - Result note:
      [4-N15-B image-missing-feature complement FiLM](checklist_results/4-N15-B_image_missing_feature_complement_film.md).
  - N15-C. F&G-only across all image specs.
    - `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`.
    - Question: does external market-regime information help regardless of
      chart-image information richness?
    - This is different from N15-B: F&G is external market-regime information,
      not an OHLCV-derived technical summary.
    - Runner:
      [kaggle_stage4_n15c_fg_only_across_image_specs_one_cell.md](notebooks/kaggle_stage4_n15c_fg_only_across_image_specs_one_cell.md).
    - Prep note:
      [4-N15-C F&G-only across image specs](checklist_results/4-N15-C_fg_only_across_image_specs.md).
  - N15-D. Selected hybrid only if N15-B/C show signal.
    - Candidate rows: `ohlc + technical_all + F&G`,
      `ohlc_ma + volume_volatility + F&G`, `ohlc_vb + bb_trend + F&G`.
    - Decision: skip for now. N15-B technical context only preserved the
      same-image baselines, and N15-C F&G-only produced only tiny positive
      deltas on volume-aware image specs. A hybrid would be hard to justify
      before testing a more genuinely external derivatives/leverage context.
  - Required metrics: accuracy, ROC-AUC, Brier score, predicted-positive rate,
    same-image Stage 2 delta, gap-to-`ohlc_ma_vb` delta, correction/regression
    counts, and net correction.
  - Integrated result summary:
    [4-N15 integrated result summary](checklist_results/4-N15_integrated_result_summary.md).
  - Plan:
    [4-N15 I60/R20 image-spec context-complement plan](checklist_results/4-N15_i60_r20_image_spec_context_complement_plan.md).
- [x] 4-N16. Derivatives/leverage-regime context
  - Purpose: test context sources that are less redundant with OHLC/MA/VB chart
    images than technical indicators, FSI/RORO, or headline TF-IDF/SVD.
  - Fixed setup: `I60/R20`, Stage 2 checkpoint reload, frozen visual
    CNN/classifier, bounded final-block FiLM first, scale `0.02`, seeds
    `42,43,44,45,46`.
  - Primary image spec: `ohlc_ma_vb`. If a feature set shows signal, repeat the
    best candidate on `ohlc_vb` because N15-C showed F&G helped only on
    volume-aware image specs.
  - Available local sources:
    - BitMEX XBTUSD funding rate, `2018-01-01` to `2024-12-31`, daily aggregate
      with Stage 4 missing rate `0%`.
    - BitMEX XBTUSD derivatives activity/futures volume, `2018-01-01` to
      `2024-12-31`, Stage 4 missing rate `0%`.
    - CFTC/CME Bitcoin futures COT open interest and positioning, weekly source
      from official CFTC files, Stage 4 missing rate `0%` after release-lagged
      daily forward-fill.
  - CFTC leakage rule:
    - Do not attach a COT report on its Tuesday report date.
    - Use `available_date = report_date + 3 calendar days`, then forward-fill
      until the next available report.
    - Keep `cot_source_report_date` and `cot_age_days` in the feature table for
      audit.
  - 4-N16-0. Source inventory and coverage lock.
    - Result: local data downloaded and documented under
      `data_inventory/crypto_derivatives/`.
    - Source note:
      [4-N16 derivatives/leverage context plan](checklist_results/4-N16_derivatives_leverage_context_plan.md).
  - [x] 4-N16-1. Derivatives context feature builder.
    - Candidate BitMEX features: `funding_rate_mean/sum/abs_mean/min/max`,
      derivatives `trades`, `volume`, `turnover`, `homeNotional`,
      `foreignNotional`.
    - Candidate CFTC/CME features: `Open_Interest_All`,
      `Lev_Money_Net_Ratio_All`, `Asset_Mgr_Net_Ratio_All`,
      `Dealer_Net_Ratio_All`, and open-interest changes/z-scores.
    - Windows: `7/20/60` for daily BitMEX features; `20/60` and current value
      for release-lagged weekly CFTC context.
    - Normalize with train-only imputation/clipping/z-score.
    - Result: completed locally. Built `derivatives_all`, `funding_only`,
      `funding_plus_cftc_oi`, `funding_plus_activity`, and
      `funding_plus_activity_plus_cftc_oi` prebuilt context artifacts for
      `I60/R20/ohlc_ma_vb`, seeds `42-46`. Context dimensions are `46`, `15`,
      `33`, `28`, and `46`, respectively. Funding/activity raw missing rate is
      `0%`; CFTC rolling/change features have early train/validation raw
      missing below `5%` and are train-median imputed.
    - Result note:
      [4-N16-1 derivatives context feature builder](checklist_results/4-N16-1_derivatives_context_feature_builder.md).
    - Runner:
      [kaggle_stage4_n16_1_derivatives_context_features_one_cell.md](notebooks/kaggle_stage4_n16_1_derivatives_context_features_one_cell.md).
  - [x] 4-N16-2. Train-only derivatives feature audit.
    - Check missing rates, feature-label/future-return correlations,
      redundancy with F&G and technical context, and Stage 2 error correlation.
    - Use this audit to avoid throwing every derivatives column into FiLM.
    - Result: completed locally. BitMEX funding is the strongest train-only
      derivatives signal (`funding_rate_max_7` score `0.4414`;
      `funding_rate_max_20` score `0.4041`). BitMEX activity is secondary
      (`bitmex_foreignnotional_mean_60` score `0.2993`) and CFTC/CME
      positioning is weaker (`cot_open_interest` score `0.2293`) with stronger
      time-trend risk. Prior-context redundancy median max absolute correlation
      is `0.7268`, so N16-3 should test lean feature sets rather than only the
      full derivatives vector.
    - Result note:
      [4-N16-2 derivatives feature audit](checklist_results/4-N16-2_derivatives_feature_audit.md).
    - Report:
      [stage4_n16_2_derivatives_feature_audit_report.md](reports/tables/stage4_n16_2_derivatives_feature_audit_report.md).
  - [x] 4-N16-3. Frozen bounded FiLM feature-set grid on `ohlc_ma_vb`.
    - Priority from N16-2: `funding_only`, `funding_plus_activity`,
      `funding_plus_cftc_oi`, `funding_plus_activity_plus_cftc_oi`.
    - Required metrics: accuracy, ROC-AUC, Brier, F1, predicted-Up rate,
      correction/regression/net correction, changed-decision rate, collapse
      warning.
    - Runner prepared:
      [kaggle_stage4_n16_3_derivatives_feature_set_grid_one_cell.md](notebooks/kaggle_stage4_n16_3_derivatives_feature_set_grid_one_cell.md).
    - Result: completed. Best row, `funding_plus_cftc_oi`, tied the same-image
      Stage 2 baseline accuracy (`0.579320`) but did not improve net
      corrections; `funding_only` was similarly stable but slightly below
      baseline. No seed-collapse issue appeared.
    - Result note:
      [4-N16-3 derivatives feature-set grid](checklist_results/4-N16-3_derivatives_feature_set_grid_prepared.md).
    - Report:
      [stage4_n16_3_derivatives_feature_set_grid_mean_std_results.csv](reports/tables/stage4_n16_3_derivatives_feature_set_grid_mean_std_results.csv).
  - [x] 4-N16-4. Selected volume-aware repeat.
    - Run the selected N16-3 candidates on `ohlc_vb`.
    - Purpose: test whether derivatives context complements volume-aware but
      visually weaker images better than the already strongest `ohlc_ma_vb`.
    - Selected feature sets: `funding_plus_cftc_oi` and `funding_only`.
    - Runner prepared:
      [kaggle_stage4_n16_4_ohlc_vb_derivatives_repeat_one_cell.md](notebooks/kaggle_stage4_n16_4_ohlc_vb_derivatives_repeat_one_cell.md).
    - Result: completed. `ohlc_vb + funding_plus_cftc_oi` improved the
      same-image Stage 2 `ohlc_vb` baseline from `0.567384` to `0.569466`
      (`+0.002082`) with net correction `+15`; `funding_only` was nearly tied
      (`+0.000278`, net correction `+2`). This is a small same-image positive
      result, not an overall-best result over `ohlc_ma_vb`.
    - Result note:
      [4-N16-4 OHLC-VB derivatives repeat](checklist_results/4-N16-4_ohlc_vb_derivatives_repeat.md).
    - Report:
      [stage4_n16_4_ohlc_vb_derivatives_repeat_mean_std_results.csv](reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_mean_std_results.csv).
  - [x] 4-N16-5. Interpretability export.
    - Target panels: extreme funding, high/low CFTC OI, high leveraged-money
      short/long imbalance, Stage 2 wrong -> FiLM correct, and Stage 2 correct
      -> FiLM wrong.
    - Export Grad-CAM, context values, gamma/beta summaries, and `prob_up`
      changes.
    - Result: completed for tabular interpretation. The selected
      `ohlc_vb + funding_plus_cftc_oi` FiLM model improves the same-image
      Stage 2 `ohlc_vb` baseline by `+0.002082` accuracy and net `+15`
      corrections, mainly by reducing weak bullish predictions in higher
      derivatives/leverage regimes. Five-seed targeted Stage2/Stage4
      Grad-CAM and Stage4 gamma/beta export completed locally; Kaggle
      reproduction cell is prepared.
    - Result note:
      [4-N16-5 derivatives interpretability export](checklist_results/4-N16-5_derivatives_interpretability_export.md).
    - Report:
      [stage4_n16_5_ohlc_vb_derivatives_interpretability_report.md](reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_report.md).
- [x] 4-N14. Final Stage 4 interpretability report
  - Purpose: turn the selected Stage 4 model into thesis-ready evidence, not
    just another metric table.
  - Required content: Stage 2 baseline vs selected context-FiLM metrics,
    correction/regression table, predicted-Up distribution, targeted Grad-CAM,
    gamma/beta/modulation-gate summaries, and representative
    `Stage2 wrong -> Stage4 correct` plus `Stage2 correct -> Stage4 wrong`
    samples.
  - Output: compact report for GitHub and professor update; large bundles stay
    local/Kaggle dataset only.
  - Result:
    [4-N14 final Stage 4 interpretability report](checklist_results/4-N14_final_stage4_interpretability_report.md).
- [x] 4-N14-B. Conditional regime analysis
  - Purpose: analyze whether context-FiLM helps in predefined market regimes
    even when average full-test accuracy does not substantially improve.
  - This is a post-training analysis track. Do not train a new model here.
  - Primary first case: same-image N16 comparison,
    `stage2_i60_ohlc_vb_r20` vs
    `stage4_film_full_bounded_last_block_i60_ohlc_vb_r20_c60_n16d_funding_plus_cftc_oi_pretrained_frozen_s0p02`.
  - Candidate regimes: F&G extreme fear/greed, high volatility, high
    derivatives/leverage, high news/macro intensity, and Stage2 uncertainty.
  - Use predefined bucket rules only; do not choose buckets after seeing which
    ones look good.
  - Minimum report threshold: at least `100` seed-decisions and `30` unique
    samples per bucket; smaller buckets are diagnostic only.
  - Plan:
    [4-N14-B conditional regime analysis plan](checklist_results/4-N14-B_conditional_regime_analysis_plan.md).
  - [x] 4-N14-B1. Prediction/context merge table.
    - Build one long decision-level table with Stage2 predictions, Stage4
      predictions, context features, `prob_up_delta`, `true_prob_delta`, and
      transition type.
    - Script:
      [build_stage4_n14b_conditional_merge_table.py](scripts/build_stage4_n14b_conditional_merge_table.py).
    - Kaggle runner:
      [kaggle_stage4_n14b1_conditional_merge_table_one_cell.md](notebooks/kaggle_stage4_n14b1_conditional_merge_table_one_cell.md).
    - Result summary:
      [stage4_n14b1_n16_derivatives_conditional_merge_report.md](reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_report.md).
    - Note: the full merged-decision CSV is large and remains local/Kaggle-only;
      compact summaries are tracked.
  - [x] 4-N14-B2. Predefined regime bucket builder.
    - Build F&G, volatility, derivatives/leverage, news/macro, and Stage2
      uncertainty buckets from the N14-B1 merged table.
  - [x] 4-N14-B3. Bucket-level metrics.
    - For each bucket, compute Stage2 accuracy, Stage4 accuracy, delta,
      correction/regression/net correction, changed-decision rate, and
      probability deltas.
  - [x] 4-N14-B4. Representative correction/regression samples.
    - Select Stage2 wrong -> Stage4 correct, Stage2 correct -> Stage4 wrong,
      high-confidence wrong, and large `prob_up_delta` samples per useful
      bucket.
  - [x] 4-N14-B5. Targeted Grad-CAM/gamma-beta linkage.
    - Link selected samples to Stage2/Stage4 Grad-CAM, context values,
      gamma/beta summaries, and probability shifts.
  - [x] 4-N14-B6. Conditional-regime report.
    - Write the thesis-ready conclusion: whether the context-FiLM correction is
      regime-specific, interpretable, and strong enough to claim conditional
      improvement.
    - Result:
      [stage4_n14b2_b6_n16_derivatives_conditional_buckets_report.md](reports/tables/stage4_n14b2_b6_n16_derivatives_conditional_buckets_report.md).
    - Key thesis row: uncertain chart + high funding has `+0.039604` accuracy
      delta with `24` corrections and `12` regressions.

Important:
- Do not draw the context values into the chart image for the main Stage 4
  experiment.
- The context enters as a separate vector.
- All context features must be available at or before image end date `t`.
- Train-only statistics must be used for context normalization.
- After N7, the main Stage 4 risk is no longer only context quality. The next
  key risk is that previous Stage 4 runs reused the Stage 2 architecture but
  not the learned Stage 2 weights. N8 addresses this before adding more
  context sources.

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
- 현재 결론: structured numeric context와 headline-news context는 일부 seed에서
  signal을 보였지만, scratch-trained context/FiLM 모델은 Stage 2 visual
  baseline을 안정적으로 넘지 못했습니다.
- 완료된 현재 track: `4-N8-A/B` 이후 N13 context-source comparison과
  FiLM scale/freeze-policy ablation까지 진행했습니다. Stage 4 code path에서
  선택된 Stage 2 checkpoint를 불러와 baseline을 재현했고,
  visual CNN/classifier를 freeze한 protocol이 가장 방어 가능했습니다.
- 완료된 현재 track: N15 image-spec context-complement check입니다. Technical
  replacement context는 대부분 same-image baseline을 보존하는 수준이었고,
  F&G는 volume-aware image spec에서만 아주 작은 positive delta를 보였습니다.
- 완료된 현재 track: N16 derivatives/leverage regime context입니다. 가장 쓸 수
  있는 same-image positive case는 `ohlc_vb + funding_plus_cftc_oi`이며,
  `ohlc_vb` Stage 2 baseline 대비 accuracy `+0.002082`, net correction `+15`를
  보였습니다.
- 첫 news version: headline-only, non-LLM, train-only TF-IDF/SVD를 7/20/60-day
  trailing news window에 적용합니다.
- 현재 순서: Stage 4 modeling/context-source 실험은 FSI/RORO,
  derivatives/leverage, N14 최종 해석 보고서, N14-B conditional-regime
  분석까지 완료했습니다. N14-B는 N16 same-image derivatives/leverage case를
  뒷받침하는 근거로 사용하고, 새로운 model branch로 주장하지 않습니다.

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
- [x] 4-I14. Stage 4 결과 보고
  - Numeric-context reporting은 `4-V9`까지 완료됐습니다.
  - 최종 보고는 다음 문서로 통합했습니다:
    [4-N14 final Stage 4 interpretability report](checklist_results/4-N14_final_stage4_interpretability_report.md).

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
- [x] 4-N4. Train-only TF-IDF/SVD news vectorizer
  - Text preprocessing, TF-IDF vocabulary, IDF weight, SVD는 train-period
    news에만 fit합니다.
  - 첫 vector size는 window별 `news_svd_32`입니다:
    `news_svd_7d`, `news_svd_20d`, `news_svd_60d`.
  - 뉴스가 없는 날은 zero news vector와 명시적인 count feature를 사용합니다.
  - Vectorizer metadata, vocabulary hash, SVD dimension, train-period fit
    range를 저장합니다.
  - 결과: train document `2,013`개(`671` samples x `7/20/60` windows)에만
    TF-IDF/SVD를 fit했고, vocabulary size `10,000`, SVD dim `32`, explained
    variance ratio sum `0.5856`입니다.
  - Full vector artifact:
    `outputs/stage4/news/stage4_news_tfidf_svd_i60_r20/news_tfidf_svd_features.parquet`.
  - 결과: [4-N4 Train-only TF-IDF/SVD news vectorizer](checklist_results/4-N4_news_tfidf_svd_vectorizer.md)
  - Table:
    [manifest](reports/tables/stage4_news_tfidf_svd_manifest.json),
    [summary](reports/tables/stage4_news_tfidf_svd_summary.csv),
    [top terms](reports/tables/stage4_news_tfidf_svd_top_terms.csv)
- [x] 4-N5. BTC sample-level news context feature builder
  - Strict `t-1`로 daily news vector를 각 BTC image sample에 merge합니다.
  - 첫 context vector:
    `news_svd_7d + news_svd_20d + news_svd_60d`와 log-count feature입니다.
  - 결과: `102`개 normalized feature = `96`개 SVD feature + `6`개 log-count
    feature입니다.
  - Normalization은 train-only median imputation, train quantile clipping,
    train z-score scaling입니다.
  - Chart image는 `I60/R20/ohlc_ma_vb` 그대로 유지합니다.
  - 결과: [4-N5 News context feature builder](checklist_results/4-N5_news_context_feature_builder.md)
  - Table:
    [audit](reports/tables/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60_seed42_news_context_feature_audit.json),
    [summary](reports/tables/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60_seed42_news_context_feature_summary.csv),
    [manifest](reports/tables/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60_seed42_news_context_manifest.json)
- [x] 4-N6. News-context baseline controls
  - Visual-only reference baseline은 Stage 2 selected five-seed
    `I60/R20/ohlc_ma_vb`입니다.
  - N5에서 news-aligned context table이 같은 sample universe를 유지하는 것을
    확인했습니다: train `671`, validation `287`, test `1,441`.
  - 먼저 `CNN + news concat` five-seed를 실행합니다.
  - 목적: FiLM modulation을 주장하기 전에 news vector가 side information으로
    유용한지 확인합니다.
  - 준비된 notebook:
    [kaggle_stage4_news_context_n6_baseline_controls_one_cell.md](notebooks/kaggle_stage4_news_context_n6_baseline_controls_one_cell.md)
  - 결과:
    [4-N6 News-context baseline controls](checklist_results/4-N6_news_context_baseline_controls.md)
  - Kaggle 결과: accuracy mean `0.5478`, ROC-AUC mean `0.5644`.
  - 진단: seed `43`, `45`가 거의 한쪽 class로 collapse했습니다. 따라서
    `102`차원 news context를 그대로 N7 FiLM에 넣기 전에 차원 안정성 확인이
    필요합니다.
- [x] 4-N6.1. News SVD-dimension stability grid
  - FiLM을 추가하기 전에 train-only TF-IDF/SVD 차원을 줄여 봅니다.
  - Grid: SVD dim `16`, `8`; 최종 context dim은 각각 `54`, `30`입니다.
  - 고정 모델: `I60/R20/ohlc_ma_vb` + `CNN + news concat`.
  - Seeds: `42, 43, 44, 45, 46`.
  - 목적: 낮은 차원의 headline vector가 news ranking signal을 유지하면서
    seed collapse를 줄이는지 확인합니다.
  - 준비된 notebook:
    [kaggle_stage4_news_context_n6_svd_dim_grid_one_cell.md](notebooks/kaggle_stage4_news_context_n6_svd_dim_grid_one_cell.md)
  - 준비 결과:
    [4-N6.1 News SVD-dim grid](checklist_results/4-N6.1_news_svd_dim_grid.md)
  - Kaggle 결과: SVD8 accuracy mean `0.5407`, ROC-AUC mean `0.5817`;
    SVD16 accuracy mean `0.5348`, ROC-AUC mean `0.5608`.
  - 결정: SVD8은 가장 강한 news ranking signal을 유지하고 FiLM input을 작게
    만들기 때문에 N7에 사용합니다.
- [x] 4-N7. News-context bounded FiLM main test
  - SVD8 news context로 `CNN + news bounded last-block FiLM` five-seed를
    실행합니다.
  - V9 교훈대로 visual path를 먼저 보호합니다.
  - `modulation_scale=0.05`를 사용합니다:
    `gamma = 1 + 0.05 * tanh(raw_gamma)`,
    `beta = 0.05 * tanh(raw_beta)`.
  - Stage 2 visual baseline, `CNN + news concat`과 비교합니다.
  - 결과: five-seed accuracy mean `0.5591`, ROC-AUC mean `0.5642`,
    predicted-Up-rate mean `0.6952`.
  - 해석: N7은 news concat SVD8보다 seed collapse를 줄였지만 Stage 2 visual
    baseline을 넘지 못했습니다. 또한 여전히 Stage 2 architecture를 scratch로
    학습한 실험이므로, 의도한 pretrained-baseline-preserving FiLM을 아직 검증한
    것은 아닙니다.
  - 준비된 notebook:
    [kaggle_stage4_news_context_n7_bounded_film_svd8_one_cell.md](notebooks/kaggle_stage4_news_context_n7_bounded_film_svd8_one_cell.md)
  - 준비 결과:
    [4-N7 News bounded FiLM SVD8](checklist_results/4-N7_news_bounded_film_svd8.md)
- [x] 4-N8. Stage 2 pretrained baseline-preserving FiLM
  - 첫 substep: Stage 2 checkpoint reload sanity. 선택된 Stage 2
    `I60/R20/ohlc_ma_vb` learned weight를 Stage 4 code path 안에서 불러와,
    context 없이 Stage 2 baseline 예측이 재현되는지 확인합니다.
  - 4-N8-A1 reload sanity는 rebuilt Stage 2 checkpoint bundle로 local 통과했습니다.
    Stage4-side reload 결과가 five-seed Stage 2 baseline을 재현했습니다:
    accuracy mean `0.579320`, ROC-AUC mean `0.584863`; classification metric은
    bundle 결과와 tolerance 안에서 일치했습니다.
  - 두 번째 substep: Stage 2 visual CNN을 freeze하고 context encoder와 bounded
    last-block FiLM head만 학습합니다. Context는 F&G-only, news SVD8-only를
    먼저 보고 그 다음 combined로 확장합니다.
  - 4-N8-B 구현 완료: Stage 2 checkpoint load, visual-backbone freeze,
    classifier freeze, frozen BatchNorm/dropout eval mode, context encoder와
    bounded final-block FiLM head만 trainable로 두는 경로를 추가했습니다.
  - Local smoke 통과: F&G-only, seed42, scale `0.05`, 64-row train/val/test.
    Loaded Stage 2 key `30`, frozen parameter `2,952,962`, trainable parameter
    `35,008`.
  - Full N8-B F&G-only Kaggle five-seed run 완료: scale `0.02`, `0.05`.
  - 결과: scale `0.02` accuracy mean `0.580291`, ROC-AUC mean `0.584930`;
    scale `0.05` accuracy mean `0.579320`, ROC-AUC mean `0.584921`.
  - 해석: N8-B는 Stage 2 baseline을 크게 이기지는 않았지만, baseline을 보존했고
    scratch-FiLM에서 보였던 심한 seed collapse를 피했습니다.
  - 필요할 경우 optional substep: CNN은 freeze하되 classifier만 열고, 그 다음에만
    final CNN block partial-unfreeze를 고려합니다.
  - 목적: 새 CNN을 scratch로 다시 학습하는 것이 아니라, 이미 강한 visual model을
    bounded correction으로 개선할 수 있는지 검증합니다.
  - 준비 결과:
    [4-N8 Pretrained baseline-preserving FiLM](checklist_results/4-N8_pretrained_baseline_preserving_film.md)
  - Reload sanity script:
    [check_stage4_n8_stage2_checkpoint_reload.py](scripts/check_stage4_n8_stage2_checkpoint_reload.py)
  - N8-B Kaggle runner:
    [kaggle_stage4_n8b_fg_only_pretrained_frozen_bounded_film_one_cell.md](notebooks/kaggle_stage4_n8b_fg_only_pretrained_frozen_bounded_film_one_cell.md)
- [x] 4-N9. News-only와 News + F&G pretrained/frozen ablation
  - `4-N8`에서 baseline-preserving context-FiLM이 Stage 2 baseline을 재현하고
    안전하게 수정할 수 있음을 확인했으므로 실행 후보입니다.
  - Gamma/beta 원칙: sample별 gamma/beta를 사람이 직접 정하지 않습니다. 모델이
    `context -> MLP -> gamma/beta` mapping을 학습하고, 실험자는 context vector,
    freeze policy, FiLM 위치, bounded modulation scale만 통제합니다.
  - N9-A weak correction 완료: news SVD8-only, CNN frozen, classifier frozen,
    scale `0.02`, five seeds. 결과: accuracy mean `0.579459`, ROC-AUC mean
    `0.585670`; 안정적이지만 Stage 2 baseline 대비 correction 크기는 작았습니다.
  - N9-B weak correction: news SVD8-only, CNN frozen, classifier frozen, scale
    `0.05`. N9-A가 안정적이지만 너무 보수적일 때만 실행합니다.
  - N9-C medium correction: news SVD8-only, CNN frozen, classifier trainable,
    scale `0.02`.
  - N9-D medium correction: news SVD8-only, CNN frozen, classifier trainable,
    scale `0.05`. N9-C가 안정적이지만 너무 약할 때만 실행합니다.
  - N9-E combined context: `news_svd_7d/20d/60d + news_count_7d/20d/60d +
    F&G-only`, scale `0.02`. news-only가 유망하거나 교수님 보고용 final
    comparison이 필요할 때 실행합니다.
  - 목적: 안정화된 N8-B 구조에서 richer external news context가 incremental signal을
    주는지 확인합니다.
  - Kaggle runner 준비:
    [kaggle_stage4_n9_news_pretrained_frozen_bounded_film_one_cell.md](notebooks/kaggle_stage4_n9_news_pretrained_frozen_bounded_film_one_cell.md)
  - 기본 실행: `N9A`, news SVD8-only, Stage 2 CNN/classifier frozen,
    bounded last-block FiLM scale `0.02`, five seeds.
  - SVD/scale grid runner 준비:
    [kaggle_stage4_n9_news_pretrained_frozen_svd_scale_grid_one_cell.md](notebooks/kaggle_stage4_n9_news_pretrained_frozen_svd_scale_grid_one_cell.md)
  - Grid points: `SVD8/0.05`, `SVD16/0.02`, `SVD16/0.05`, `SVD32/0.02`,
    `SVD32/0.05`. `SVD8/0.02`는 N9-A에서 이미 실행했기 때문에 제외합니다.
  - Grid 목적: N9-A의 correction이 너무 보수적이었는지, 또는 SVD8이 headline
    context를 너무 강하게 압축했는지 확인합니다.
  - 기존 single-variant 후속 실험은 bounded-FiLM cell 상단의 `N9_VARIANT`만
    `N9B`, `N9C`, `N9D`로 바꿔서 실행합니다.
  - 설계 노트:
    [4-N9 News pretrained/frozen FiLM design](checklist_results/4-N9_news_pretrained_frozen_film_design.md)
  - Closeout: 현재 Stage 4 scope에서는 news-only pretrained/frozen run과 grid
    diagnostic까지 완료로 닫습니다. `news + F&G`는 명시적으로 not-run/deferred이며
    결과로 주장하지 않습니다.
- [x] 4-N10. News interpretability report
  - 먼저 같은 sample에 대해 Stage 2 baseline vs N8-B F&G-only Grad-CAM을
    비교합니다.
  - 4-N9를 실행하면 correct/incorrect Up/Down sample에 대해 news title,
    news-count feature, FiLM gamma/beta summary를 함께 export합니다.
  - Feature sensitivity: zero news vector, F&G 제거, context vector를 train mean으로
    대체하는 분석을 추가합니다.
  - 핵심 해석 target은 `Stage 2 wrong -> N8/N9 correct` sample입니다. 이 케이스가
    context-FiLM이 visual-baseline error를 수정했는지 보여줍니다.
  - 현재 가진 artifact 기준 initial report 완료:
    [4-N10 News interpretability report](checklist_results/4-N10_news_interpretability_report.md)
  - 선택된 N9 grid-best Grad-CAM export cell 준비:
    [kaggle_stage4_n10_selected_news_interpretability_one_cell.md](notebooks/kaggle_stage4_n10_selected_news_interpretability_one_cell.md)
  - 현재 결론: N8/N9 bounded FiLM은 Stage 2 baseline을 보존하고
    ROC-AUC/calibration/Up-bias를 약간 개선하지만, accuracy gain은 강하게
    주장하기 어렵습니다.
  - 한계: N9 SVD/scale grid bundle은 metric-only입니다. 핵심 target인
    `Stage 2 wrong -> N9 correct` Grad-CAM은 best grid 후보인
    `SVD32/scale0.02`에 대해 추가 export가 필요합니다.
- [x] 4-N10-A. Stage 2 vs N10 correction-analysis code
  - Prediction-level 비교 script 추가:
    [analyze_stage4_stage2_context_corrections.py](scripts/analyze_stage4_stage2_context_corrections.py)
  - Kaggle one-cell runner 추가:
    [kaggle_stage4_n10_stage2_vs_n10_correction_analysis_one_cell.md](notebooks/kaggle_stage4_n10_stage2_vs_n10_correction_analysis_one_cell.md)
  - 목적: `Stage2 wrong -> N10 correct`, `Stage2 correct -> N10 wrong`,
    transition summary, targeted Grad-CAM/gamma-beta/news 해석용
    sample-index list를 export합니다.
- [x] 4-N10-B. Targeted Grad-CAM + gamma/beta modulation export code
  - Stage 2와 Stage 4 Grad-CAM exporter에 targeted sample mode를 추가했습니다.
  - Kaggle one-cell runner 추가:
    [kaggle_stage4_n10_b_targeted_gradcam_modulation_one_cell.md](notebooks/kaggle_stage4_n10_b_targeted_gradcam_modulation_one_cell.md)
  - 목적: N10-A에서 고른 동일한 `sample_index`에 대해 Stage 2 vs N10
    Grad-CAM을 비교하고, N10 FiLM gamma/beta modulation metadata를 함께
    export합니다.
  - 설계 노트:
    [4-N10-B targeted Grad-CAM modulation export](checklist_results/4-N10-B_targeted_gradcam_modulation_export.md)
- [x] 4-N11. LLM summary/embedding decision
  - Headline-only no-leakage track이 안정화된 뒤로 미룹니다.
  - 사용한다면 model name, prompt, version/date, cache hash, runtime을
    기록해야 합니다.
  - Closeout decision:
    [4-N11 LLM embedding deferred](checklist_results/4-N11_llm_embedding_deferred.md).
- [x] 4-N12. Optional uncertainty-gated FiLM follow-up
  - N9/N10 해석에서 context가 주로 Stage 2 chart model이 애매한 sample에서
    도움이 된다는 근거가 보일 때만 실행합니다.
  - 아이디어: Stage 2 chart 판단이 애매할수록 context-FiLM correction을 더 허용하고,
    chart 판단이 확신에 가까울수록 correction을 약하게 둡니다.
  - 후보 uncertainty:
    `uncertainty = 4 * prob_up_stage2 * (1 - prob_up_stage2)`.
  - 후보 formula:
    `gamma = 1 + uncertainty * scale * tanh(raw_gamma)`,
    `beta = uncertainty * scale * tanh(raw_beta)`.
  - 목적: news/F&G context가 visual chart evidence가 애매한 구간에서 correction
    signal로 가장 유용하다는 thesis-friendly claim을 검증합니다.
- [x] 4-N12-A. Uncertainty-gated news FiLM 구현과 runner 준비
  - `film_full_uncertainty_gated_last_block`을 추가했습니다.
  - 공식:
    `uncertainty = 4 * prob_up_stage2 * (1 - prob_up_stage2)`.
  - N8/N9/N10과 같은 baseline 보존 규칙을 사용합니다. Stage 2
    `I60/R20/ohlc_ma_vb` checkpoint를 load하고 CNN/classifier는 freeze한 뒤,
    news context encoder와 final-block FiLM head만 학습합니다.
  - Kaggle runner:
    [kaggle_stage4_n12a_uncertainty_gated_news_film_one_cell.md](notebooks/kaggle_stage4_n12a_uncertainty_gated_news_film_one_cell.md)
  - 기본 grid: news SVD32, scale `0.02`, `0.05`, five seeds.
  - Local shape check 통과; 결과 노트:
    [4-N12-A uncertainty-gated news FiLM](checklist_results/4-N12-A_uncertainty_gated_news_film.md)
- [x] 4-N12-B. Confidence-gated news FiLM 구현과 runner 준비
  - `film_full_confidence_gated_last_block`을 추가했습니다.
  - 공식:
    `confidence = abs(2 * stage2_prob_up - 1)`.
  - 같은 baseline 보존 규칙을 사용합니다. Stage 2 `I60/R20/ohlc_ma_vb`
    checkpoint를 load하고 CNN/classifier는 freeze한 뒤, news context encoder와
    final-block FiLM head만 학습합니다.
  - Kaggle runner:
    [kaggle_stage4_n12b_confidence_gated_news_film_one_cell.md](notebooks/kaggle_stage4_n12b_confidence_gated_news_film_one_cell.md)
  - 기본 grid: news SVD32, scale `0.02`, `0.05`, five seeds.
  - Local shape check 통과; 결과 노트:
    [4-N12-B confidence-gated news FiLM](checklist_results/4-N12-B_confidence_gated_news_film.md)
- [x] 4-N12-C. Stage 2 frozen + technical-only bounded FiLM
  - 목적: image에서 파생되는 technical context와 외부/news context를 같은 Stage 2
    frozen protocol 안에서 분리해서 확인합니다.
  - 후보 feature: `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60`.
  - method는 우선 `film_full_bounded_last_block`으로 고정합니다. bounded 결과에서
    signal이 보일 때만 gated variant로 확장합니다.
  - 비교 대상: Stage 2 baseline, N8-B F&G-only, N9/N10 news-only, N12-A/B
    gated news.
  - Kaggle runner:
    [kaggle_stage4_n12c_technical_only_pretrained_frozen_bounded_film_one_cell.md](notebooks/kaggle_stage4_n12c_technical_only_pretrained_frozen_bounded_film_one_cell.md)
  - 결과: scale `0.02` accuracy mean `0.579736`, ROC-AUC mean `0.584778`.
    Stage 2 frozen baseline과 사실상 동률이며 의미 있는 개선은 아닙니다.
  - 결과 노트:
    [4-N12-C technical-only pretrained frozen bounded FiLM](checklist_results/4-N12-C_technical_only_pretrained_frozen_bounded_film.md)
- [x] 4-N12-D. Frozen Stage 2 protocol 안에서 context-source comparison
  - 목적: one-off variant를 계속 늘리지 않고, 어떤 context source가 thesis에서
    방어 가능한지 결정합니다.
  - 같은 image, split, checkpoint loading, freeze policy, bounded/gated FiLM
    protocol 아래에서 `F&G-only`, `news-only`, `technical-only`, `news + F&G`를
    비교합니다.
  - 필수 metric: accuracy, ROC-AUC, Brier score, F1, predicted-Up rate,
    correction count, regression count, net correction.
  - output: compact comparison table과 final Stage 4 model 추천.
  - 결과: existing context source 기준 완료. F&G-only scale `0.02`가 가장
    compact한 accuracy 후보입니다(`0.580291` vs Stage 2 `0.579320`). News는
    ROC-AUC/Brier signal이 가장 뚜렷하지만 hard decision은 약하고,
    technical-only는 Stage 2와 사실상 동률입니다.
  - 주의: `news + F&G` combined context는 comparison table에서 planned/not-run으로
    기록합니다. five-seed run 전에는 결과로 주장하지 않습니다.
  - 결과 노트:
    [4-N12-D context-source comparison](checklist_results/4-N12-D_context_source_comparison.md)
- [x] 4-N13. Macro/RORO context extension
  - 목적: F&G/news/technical context가 작은 개선만 보였기 때문에, 이미지 밖
    macro risk-regime context를 frozen Stage 2 FiLM 구조에 넣어봅니다.
  - thesis question: 공식 financial stress 또는 risk-on/risk-off regime이 BTC
    OHLCV에서 파생한 context보다 Stage 2 visual feature를 더 의미 있게
    condition할 수 있는지 확인합니다.
  - shared protocol: `I60/R20/ohlc_ma_vb`, seeds `42-46`, Stage 2 checkpoint
    loaded/frozen, classifier frozen, bounded last-block FiLM, 우선 conservative
    scale `0.02`.
- [x] 4-N13-0. Macro/RORO source audit and terminology lock
  - `OFR FSI`와 `RORO`를 명확히 구분합니다. OFR FSI는 직접 RORO가 아니라
    공식 financial-stress/risk-off proxy입니다.
  - source link, date coverage, CSV load path, missing-date policy, image end
    date `t` 기준 사용 가능성을 기록합니다.
  - source 1: OFR Financial Stress Index CSV, 2000-present coverage.
  - source 2: KC Fed methodology를 참고한 public-data RORO proxy. KC Fed의
    proprietary/full input을 복제한다고 쓰지 않습니다.
  - Closeout: terminology/source audit는
    [4-N13-1 OFR FSI feature builder](checklist_results/4-N13-1_ofr_fsi_feature_builder.md)
    와
    [4-N13-3 public RORO proxy builder](checklist_results/4-N13-3_public_roro_proxy_builder.md)
    에 기록했습니다.
- [x] 4-N13-1. OFR FSI feature builder
  - raw source: `https://www.financialresearch.gov/financial-stress-index/data/fsi.csv`.
  - 해석: 높은 `OFR FSI` = 높은 financial stress = risk-off proxy. BTC가 반드시
    하락한다고 hard-code하지 않고 FiLM이 관계를 학습하게 둡니다.
  - 후보 feature: `ofr_fsi_value`, `ofr_fsi_mean_20`,
    `ofr_fsi_mean_60`, `ofr_fsi_delta_20`, `ofr_fsi_delta_60`,
    `ofr_fsi_std_60`, optional category values `Credit`,
    `Equity valuation`, `Funding`, `Safe assets`, `Volatility`.
  - train-only imputation, clipping, z-score normalization을 사용합니다.
  - 준비된 script:
    [build_stage4_fsi_context_features.py](scripts/build_stage4_fsi_context_features.py)
  - Kaggle one-cell:
    [kaggle_stage4_n13_1_ofr_fsi_context_features_one_cell.md](notebooks/kaggle_stage4_n13_1_ofr_fsi_context_features_one_cell.md)
  - 최신 업로드 zip:
    `/Users/jaehyeonjeong/Desktop/논문/stage4_film_conditioning_n13_1_latest.zip`.
  - 준비 노트:
    [4-N13-1 OFR FSI feature builder](checklist_results/4-N13-1_ofr_fsi_feature_builder.md)
  - 결과: six FSI features, `context_dim=6`,
    train/validation/test split counts `671/287/1441`, source-level rolling
    feature generation 적용 후 six FSI features 모두 missing rate 0으로
    완료했습니다.
  - feature screening 결과:
    [4-N13-1 OFR FSI feature screening](checklist_results/4-N13-1_fsi_feature_screening.md).
    다음 frozen-FiLM run에서는 여섯 개 전체가 정답이라고 가정하지 않고,
    `FSI-2 = mean_60 + delta_60`, `FSI-3 = mean_60 + delta_60 + std_60`,
    `FSI-all`을 비교합니다.
- [x] 4-N13-2. FSI-only frozen bounded FiLM five-seed run
  - context source: OFR FSI features only.
  - Kaggle one-cell:
    [kaggle_stage4_n13_2_fsi_only_pretrained_frozen_bounded_film_one_cell.md](notebooks/kaggle_stage4_n13_2_fsi_only_pretrained_frozen_bounded_film_one_cell.md)
  - feature-set grid: `fsi_2`, `fsi_3`, `fsi_all`.
  - comparison: Stage 2 frozen baseline, N8-B F&G-only, N10/N12 news-only,
    N12-C technical-only.
  - metric: accuracy, ROC-AUC, Brier, F1, predicted-Up rate,
    correction/regression/net correction, seed-level collapse check.
  - 결과:
    [4-N13-2 FSI-only frozen bounded FiLM](checklist_results/4-N13-2_fsi_only_pretrained_frozen_bounded_film.md).
    best FSI row는 `fsi_all`, accuracy `0.579875`, ROC-AUC `0.584859`,
    five-seed total net correction `+4`, collapse warning `0`입니다.
    안정적이지만 Stage 2 또는 N8-B F&G-only보다 의미 있게 강하지는 않습니다.
- [x] 4-N13-3. KC Fed-inspired public-data RORO proxy builder
  - raw sources: VIX, S&P500/NASDAQ returns, Broad Dollar Index, US 10Y yield,
    optional high-yield OAS and gold.
  - 방향: positive value가 risk-off pressure를 의미하도록 부호를 정렬합니다.
  - candidate proxy: risk-off-aligned daily changes/returns에 대해 train-fit PCA
    first component를 만들고, raw components도 같이 저장합니다.
  - 명시: KC Fed full/proprietary input replication이 아니라 public-data RORO
    proxy입니다.
  - 구현:
    [build_stage4_roro_context_features.py](scripts/build_stage4_roro_context_features.py),
    [kaggle_stage4_n13_3_public_roro_context_features_one_cell.md](notebooks/kaggle_stage4_n13_3_public_roro_context_features_one_cell.md).
  - source audit:
    [4-N13-3 public RORO proxy builder](checklist_results/4-N13-3_public_roro_proxy_builder.md).
    KC Fed official daily/weekly 파일은 documentation용으로 cache했지만
    2023년 6월부터 시작해 Stage 4 train period를 커버하지 못합니다.
    따라서 학습용 proxy는 longer-history public input cache를 사용합니다.
  - 결과: VIX, S&P500, DXY, US 10Y component로 local N13-3 artifact 생성까지
    완료했습니다. `context_dim=10`, PCA explained variance ratio `0.554831`,
    split counts `671/287/1441`, missing warning `0`입니다.
  - 구현 공식:
    `PC1_train_only(z(VIX_t - VIX_{t-20}), z(-log(SP500_t/SP500_{t-20})), z(log(DXY_t/DXY_{t-20})), z(-(DGS10_t - DGS10_{t-20})))`.
    값이 클수록 stronger risk-off pressure가 되도록 sign을 고정했습니다.
  - cached raw input:
    `data_inventory/roro_public/raw/VIXCLS.csv`,
    `data_inventory/roro_public/raw/SP500.csv`,
    `data_inventory/roro_public/raw/DXY_yahoo_DX-Y.NYB.csv`,
    `data_inventory/roro_public/raw/DGS10.csv`.
    `BAMLH0A0HYM2.csv`도 cache했지만 train-period coverage가 없어 PCA에서는
    제외됩니다.
  - 제외 메모: HYG/high-yield ETF price는 HY OAS가 아니고 ETF price dynamics가
    섞이므로 이번 N13-3에서는 사용하지 않습니다.
- [x] 4-N13-4. RORO-proxy-only frozen bounded FiLM five-seed run
  - context source: public-data RORO proxy features only.
  - 4-N13-2와 같은 protocol/metric을 사용합니다.
  - 준비된 Kaggle runner:
    `notebooks/kaggle_stage4_n13_4_roro_only_pretrained_frozen_bounded_film_one_cell.md`.
  - 권장 업로드 bundle:
    `stage4_film_conditioning_n13_4_with_stage2_bundle.zip`. Stage 2
    I60/R20/ohlc_ma_vb seed 42-46 checkpoint bundle을 포함해 Kaggle reset/path
    문제를 줄입니다.
  - 결과: 3개 RORO feature set x 5 seeds 완료. collapse warning은 없습니다.
    best accuracy row는 `roro_3`이며 accuracy `0.579320`, ROC-AUC `0.584748`,
    Brier `0.274278`, F1 `0.650924`입니다. Stage 2와 사실상 동률이고,
    best F&G-only row보다는 약합니다.
  - 리뷰:
    [4-N13-4 RORO-only frozen bounded FiLM](checklist_results/4-N13-4_roro_proxy_only_pretrained_frozen_bounded_film.md).
- [x] 4-N13-5. Macro context-source comparison
  - `FSI-only`, `RORO-proxy-only`, `F&G-only`, `news-only`, `technical-only`,
    를 비교합니다.
  - accuracy 또는 ROC/Brier가 개선되고 class-collapse가 없을 때만 final Stage 4
    interpretation 후보로 선택합니다.
  - 결과: 완료. F&G-only scale `0.02`가 compact accuracy 후보 1위입니다
    (`0.580291`, Stage 2 대비 +`0.000972`). News SVD32 scale `0.02`는
    interpretability/calibration 후보로 유지합니다. FSI/RORO는 안정적이지만
    단독 최종 모델로는 약합니다.
  - 리뷰:
    [4-N13-5 macro context-source comparison](checklist_results/4-N13-5_macro_context_source_comparison.md).
- [x] 4-N13-5A. Cross-context feature audit
  - 이미 만든 context feature들을 같은 sample/date index 기준으로 merge합니다:
    F&G, news SVD/count, technical BB/MFI/RV, OFR FSI, public RORO, label,
    future return, Stage 2 `prob_up`, Stage 2 `correct`.
  - feature selection diagnostic은 train split만 사용합니다. validation/test는
    확인용으로만 둡니다.
  - audit 항목: missing rate, feature-label correlation, feature-future-return
    correlation, feature-Stage2-error correlation, feature-feature correlation,
    redundancy cluster.
  - 결과: 2,399개 aligned sample과 126개 context feature로 완료했습니다.
    News SVD가 train-only signal이 가장 강하고, F&G는 가장 compact한 regime
    source입니다. FSI/RORO는 안정적이지만 단독 signal은 약합니다. Stage 2
    error-rate와의 상관은 전반적으로 약하므로 selected-combo FiLM은 작고
    보수적으로 가야 합니다.
  - 리뷰:
    [4-N13-5A cross-context feature audit](checklist_results/4-N13-5A_cross_context_feature_audit.md).
- [x] 4-N13-5B. Selected-combo context FiLM
  - 4-N13-5A에서 중복이 적은 feature set이 보일 때만 selected-combo context
    실험을 한 번 실행합니다.
  - 후보 크기: 대략 6개 feature.
  - primary candidate: `news_svd_60d_09`, `news_svd_20d_18`, `fg_mean_60`,
    `fg_delta_60`, `ofr_fsi_std_60`, `riskoff_dollar_return_20`.
  - optional technical add-on: `bb_bandwidth_60`.
  - Stage 2 frozen protocol, bounded last-block FiLM, five seeds, scale `0.02`
    그리고 필요하면 `0.05`.
  - 결정 기준: 성능을 높이거나 source comparison 해석을 명확하게 만들 때만
    유지합니다.
  - 결과: metric-only five-seed run으로 완료했습니다. 평균 accuracy는 Stage 2
    frozen baseline과 동일한 `0.579320`, ROC-AUC 변화는 `+0.000004`, Brier는
    `-0.000202`만큼 소폭 개선, net correction은 `0.0`입니다. raw table의
    `status=failed`는 Grad-CAM을 끈 상태에서 이전 output checker가 Grad-CAM
    artifact를 요구해서 생긴 check mismatch입니다.
  - 결론: 안정적이지만 final performance candidate는 아닙니다. 6개 feature
    selected combo는 Stage 2 decision을 거의 보존했으므로, 더 큰 all-context
    stacking은 정당화하기 어렵습니다.
  - 리뷰:
    [4-N13-5B selected-combo context FiLM](checklist_results/4-N13-5B_selected_combo_context_film.md).
- [x] 4-N13-6. Macro interpretability export
  - selected combo가 아니라 가장 강한 compact 후보인 F&G-only `s0.02`와
    news SVD32 `s0.02`를 중심으로 봅니다.
  - target sample: Stage 2 wrong -> context-FiLM correct, Stage 2 correct ->
    context-FiLM wrong, extreme regime/news window.
  - targeted Grad-CAM, context value, gamma/beta summary, modulation gate,
    `prob_up` change를 export합니다.
  - 준비 상태: N13-6 Kaggle one-cell runner를 준비했습니다. 후보별
    correction/regression table을 만들고, extreme-context panel을 추가한 뒤,
    같은 sample index에 대해 Stage 2 vs context-FiLM Grad-CAM/gamma-beta를
    export하고 bundle로 묶습니다.
  - 리뷰:
    [4-N13-6 interpretability export](checklist_results/4-N13-6_interpretability_export.md).
- [x] 4-N13-7. 선택된 context source에 대한 최종 FiLM constraint/scale ablation
  - 4-N13-5/6에서 가장 안정적인 context source(`F&G`, `news`, `FSI`, `RORO`)를
    고른 뒤에만 실행합니다.
  - 목적: Stage 2 frozen protocol에서는 현재 bounded FiLM이 너무 보수적이었는지
    확인합니다.
  - 고정 조건: Stage 2 I60/R20/ohlc_ma_vb visual CNN과 classifier freeze, 같은
    split, 같은 seeds, 같은 selected context feature.
  - A. 같은 bounded equation에서 scale만 키우는 grid, 새 모델 코드 불필요:
    bounded last-block FiLM scale `0.02`, `0.05`, `0.10`, `0.20`.
    - N8-B에서 이미 `0.02`, `0.05`를 수행했습니다.
    - N13-7A runner는 `0.10`, `0.20` 실행용으로 준비했습니다:
      [kaggle_stage4_n13_7a_fg_bounded_scale_grid_one_cell.md](notebooks/kaggle_stage4_n13_7a_fg_bounded_scale_grid_one_cell.md).
    - 준비 메모:
      [4-N13-7A F&G bounded FiLM scale grid](checklist_results/4-N13-7A_fg_bounded_scale_grid.md).
    - 결과: 완료. `0.10` accuracy mean `0.579042`, ROC-AUC `0.584811`,
      net correction `-2`; `0.20` accuracy mean `0.578487`, ROC-AUC
      `0.584539`, net correction `-6`. 큰 scale은 N8-B `0.02`를 이기지
      못했으므로 small-scale F&G 설정을 더 강한 bounded setting으로 유지합니다.
  - B. gamma/beta constraint 완화:
    zero-init `gamma/beta` head를 쓰는 unbounded 또는 weakly regularized
    last-block FiLM.
  - C. gamma/beta equation 변경:
    현재 `1 + scale * tanh(raw)` 규칙을 positive-gamma sigmoid/softplus 또는
    regularized residual-linear FiLM 같은 다른 baseline-preserving 방식과
    비교합니다.
  - D. classifier-unfreeze variant:
    Stage 2 visual CNN은 freeze하고 final classifier만 열어서 classifier +
    context encoder/FiLM head만 학습합니다.
    - N13-7D runner는 F&G-only `scale=0.02`로 준비했습니다:
      [kaggle_stage4_n13_7d_fg_classifier_unfreeze_one_cell.md](notebooks/kaggle_stage4_n13_7d_fg_classifier_unfreeze_one_cell.md).
    - 준비 메모:
      [4-N13-7D F&G classifier-unfreeze FiLM](checklist_results/4-N13-7D_fg_classifier_unfreeze.md).
    - 결과: 완료. Classifier-unfreeze accuracy mean `0.574323`,
      ROC-AUC `0.584220`, Brier `0.280218`, net correction `-36`.
      classifier가 너무 자유로워져 regression이 correction보다 많아졌으므로
      N8-B `scale=0.02`를 가장 좋은 F&G setting으로 유지합니다.
  - 구현 규칙: B/C/D는 별도 구현이 필요하며, 새 context source가 아니라
    FiLM/freeze-policy ablation으로 보고합니다.
  - 필수 metric: accuracy, ROC-AUC, Brier score, predicted-positive rate,
    collapse warning, Stage 2 대비 correction/regression, net correction,
    gamma/beta magnitude summary.
  - 결정 기준: 큰 scale 또는 constraint 완화 모델은 class-collapse나 regression
    증가 없이 주요 metric 중 하나라도 개선될 때만 유지합니다.
  - 전체 metric 개선이 없으면 같은 output으로 조건부 regime 분석을 수행합니다:
    extreme context regime, high-volatility/high-stress regime, Stage 2 wrong ->
    FiLM correct sample.
  - Closeout: A와 D를 실행했고 둘 다 conservative frozen bounded setup보다
    약했습니다. 실행된 relaxation test에서 이미 FiLM 자유도가 커질수록
    regression이 늘어나는 패턴이 확인되어 B/C는 deferred로 닫습니다.
- [x] 4-N13-B. Optional sentiment/event feature extension
  - 결정: Stage 4에서 더 진행하지 않고 Stage 5로 이관했습니다.
  - 이유: Stage 4는 numeric context, TF-IDF/SVD news, FSI/RORO,
    derivatives/leverage, image-spec complement check까지 non-LLM context
    cycle을 닫았습니다. FinBERT/GPT/Claude sentiment/event feature는
    `stage5_llm_news_embedding`에서 관리합니다.
  - Stage 4 상태: 최종 Stage 4 보고의 blocker가 아닙니다.
- [x] 4-N15. I60/R20 image-spec context-complement ablation
  - 목적: chart image에 빠진 정보가 있을 때 context-FiLM이 그 정보를 보완할 수
    있는지 확인합니다.
  - 고정 조건: `I60/R20`, image spec `ohlc`, `ohlc_ma`, `ohlc_vb`,
    `ohlc_ma_vb`, 각 image spec에 대응하는 seed-matched Stage 2 checkpoint,
    frozen visual CNN/classifier, bounded final-block FiLM, scale `0.02`, seeds
    `42,43,44,45,46`.
  - Stage 2 기준선: `ohlc_ma_vb` accuracy `0.579320`, `ohlc_vb`
    `0.567384`, `ohlc` `0.558085`, `ohlc_ma` `0.557529`.
  - N15-A. 네 가지 image spec의 same-image Stage 2 reload.
    - context 추가 전 각 image spec의 frozen baseline을 정확히 고정합니다.
    - 중요: 각 image spec은 자기 Stage 2 checkpoint를 써야 합니다.
      `ohlc_ma_vb` checkpoint를 다른 image spec에 재사용하지 않습니다.
    - Runner:
      [kaggle_stage4_n15a_rebuild_i60_r20_four_image_stage2_checkpoints_one_cell.md](notebooks/kaggle_stage4_n15a_rebuild_i60_r20_four_image_stage2_checkpoints_one_cell.md).
    - 준비 메모:
      [4-N15-A I60/R20 four-image Stage 2 checkpoint bundle](checklist_results/4-N15-A_i60_r20_stage2_four_image_checkpoint_bundle.md).
  - N15-B. Image-missing-feature complement FiLM.
    - F&G-across-images보다 먼저 실행합니다.
    - `ohlc + technical_all`: `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`,
      `rv_60`.
    - `ohlc_ma + volume_volatility`: `mfi_60`, `rv_60`.
    - `ohlc_vb + bb_trend`: `bb_percent_b_60`, `bb_bandwidth_60`.
    - `ohlc_ma_vb + technical_all_control`: 같은 technical vector를 붙입니다.
      full image에서는 technical context가 중복이면 neutral할 것으로 예상합니다.
    - 질문: context-FiLM이 이미지에 직접 그리지 않은 chart-derived information을
      일부 대체할 수 있는가?
  - N15-C. F&G-only across all image specs.
    - `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`.
    - 질문: image 정보량과 무관하게 외부 market-regime 정보가 도움이 되는가?
    - 완료: same-image Stage 2 대비 `ohlc` -0.0004, `ohlc_ma` -0.0008,
      `ohlc_vb` +0.0006, `ohlc_ma_vb` +0.0010. Volume-aware image spec에서만
      작은 positive delta가 나왔지만 효과 크기는 약함.
    - 결과:
      [4-N15-C F&G-only across image specs](checklist_results/4-N15-C_fg_only_across_image_specs.md).
  - N15-D. B/C에서 signal이 있을 때만 selected hybrid 실행.
    - 후보: `ohlc + technical_all + F&G`,
      `ohlc_ma + volume_volatility + F&G`, `ohlc_vb + bb_trend + F&G`.
    - 결정: 일단 skip합니다. N15-B technical context는 same-image baseline
      보존 수준이었고, N15-C F&G-only는 volume-aware image spec에서만 작은
      positive delta를 보였습니다. 더 genuinely external한 derivatives/leverage
      context를 먼저 테스트하는 편이 더 방어 가능합니다.
  - 필수 metric: accuracy, ROC-AUC, Brier score, predicted-positive rate,
    same-image Stage 2 delta, `ohlc_ma_vb` gap delta, correction/regression,
    net correction.
  - 통합 결과 요약:
    [4-N15 integrated result summary](checklist_results/4-N15_integrated_result_summary.md).
  - 계획:
    [4-N15 I60/R20 image-spec context-complement plan](checklist_results/4-N15_i60_r20_image_spec_context_complement_plan.md).
- [x] 4-N16. Derivatives/leverage-regime context
  - 목적: OHLC/MA/VB chart image나 technical indicator와 덜 중복되는
    leverage/positioning context를 테스트합니다.
  - 고정 조건: `I60/R20`, Stage 2 checkpoint reload, visual CNN/classifier
    freeze, bounded final-block FiLM first, scale `0.02`, seeds
    `42,43,44,45,46`.
  - Primary image spec: `ohlc_ma_vb`. Signal이 보이면 N15-C에서 positive
    delta가 있었던 volume-aware weaker image인 `ohlc_vb`에 best candidate를
    반복합니다.
  - 사용 가능한 local source:
    - BitMEX XBTUSD funding rate: `2018-01-01`부터 `2024-12-31`, Stage 4
      missing rate `0%`.
    - BitMEX XBTUSD derivatives activity/futures volume: `2018-01-01`부터
      `2024-12-31`, Stage 4 missing rate `0%`.
    - CFTC/CME Bitcoin futures COT open interest/positioning: 공식 CFTC 주간
      source. Release-lagged daily forward-fill 후 Stage 4 missing rate `0%`.
  - CFTC leakage rule:
    - COT report를 화요일 report date에 바로 붙이지 않습니다.
    - `available_date = report_date + 3 calendar days`로 둔 뒤 다음 available
      report 전까지 forward-fill합니다.
    - feature table에 `cot_source_report_date`, `cot_age_days`를 남겨 audit
      가능하게 합니다.
  - 4-N16-0. Source inventory and coverage lock.
    - 결과: local data는 `data_inventory/crypto_derivatives/`에 저장하고
      문서화했습니다.
    - Source note:
      [4-N16 derivatives/leverage context plan](checklist_results/4-N16_derivatives_leverage_context_plan.md).
  - [x] 4-N16-1. Derivatives context feature builder.
    - BitMEX 후보: `funding_rate_mean/sum/abs_mean/min/max`, derivatives
      `trades`, `volume`, `turnover`, `homeNotional`, `foreignNotional`.
    - CFTC/CME 후보: `Open_Interest_All`, `Lev_Money_Net_Ratio_All`,
      `Asset_Mgr_Net_Ratio_All`, `Dealer_Net_Ratio_All`, OI change/z-score.
    - Window: BitMEX daily feature는 `7/20/60`; release-lagged weekly CFTC
      context는 current, `20/60` change/level 중심.
    - train-only imputation/clipping/z-score로 normalize합니다.
    - 결과: local에서 완료했습니다. `I60/R20/ohlc_ma_vb`, seeds `42-46` 기준
      `derivatives_all`, `funding_only`, `funding_plus_cftc_oi`,
      `funding_plus_activity`, `funding_plus_activity_plus_cftc_oi` prebuilt
      context artifact를 만들었습니다. Context dimension은 각각 `46`, `15`,
      `33`, `28`, `46`입니다. Funding/activity는 raw missing `0%`; CFTC
      rolling/change feature는 train/validation 초반 raw missing이 `5%`보다
      낮고 train median으로 impute됩니다.
    - 결과:
      [4-N16-1 derivatives context feature builder](checklist_results/4-N16-1_derivatives_context_feature_builder.md).
    - Runner:
      [kaggle_stage4_n16_1_derivatives_context_features_one_cell.md](notebooks/kaggle_stage4_n16_1_derivatives_context_features_one_cell.md).
  - [x] 4-N16-2. Train-only derivatives feature audit.
    - missing rate, feature-label/future-return correlation, F&G/technical
      context와의 redundancy, Stage 2 error correlation을 확인합니다.
    - 모든 derivatives column을 한 번에 넣지 않고 audit로 compact set을
      고릅니다.
    - 결과: 로컬 완료. BitMEX funding이 가장 강한 train-only derivatives
      signal입니다(`funding_rate_max_7` score `0.4414`,
      `funding_rate_max_20` score `0.4041`). BitMEX activity는 2순위이고
      CFTC/CME positioning은 더 약하며 time-trend risk가 더 큽니다.
      F&G/technical prior context와의 median max abs correlation은 `0.7268`라서
      N16-3에서는 full vector만 넣지 말고 lean feature set을 비교합니다.
    - 결과 노트:
      [4-N16-2 derivatives feature audit](checklist_results/4-N16-2_derivatives_feature_audit.md).
    - Report:
      [stage4_n16_2_derivatives_feature_audit_report.md](reports/tables/stage4_n16_2_derivatives_feature_audit_report.md).
  - [x] 4-N16-3. `ohlc_ma_vb` frozen bounded FiLM feature-set grid.
    - N16-2 우선순위: `funding_only`, `funding_plus_activity`,
      `funding_plus_cftc_oi`, `funding_plus_activity_plus_cftc_oi`.
    - 필수 metric: accuracy, ROC-AUC, Brier, F1, predicted-Up rate,
      correction/regression/net correction, changed-decision rate, collapse
      warning.
    - Runner 준비:
      [kaggle_stage4_n16_3_derivatives_feature_set_grid_one_cell.md](notebooks/kaggle_stage4_n16_3_derivatives_feature_set_grid_one_cell.md).
    - 결과: 완료. 최고 row인 `funding_plus_cftc_oi`는 같은 image Stage 2
      baseline accuracy (`0.579320`)와 동률이었지만 net correction은 개선되지
      않았습니다. `funding_only`도 안정적이지만 baseline보다 아주 낮았습니다.
      seed collapse는 없었습니다.
    - 결과 노트:
      [4-N16-3 derivatives feature-set grid](checklist_results/4-N16-3_derivatives_feature_set_grid_prepared.md).
    - Report:
      [stage4_n16_3_derivatives_feature_set_grid_mean_std_results.csv](reports/tables/stage4_n16_3_derivatives_feature_set_grid_mean_std_results.csv).
  - [x] 4-N16-4. Selected volume-aware repeat.
    - N16-3의 selected candidate를 `ohlc_vb`에 반복합니다.
    - 목적: derivatives context가 가장 강한 `ohlc_ma_vb`보다, volume-aware지만
      시각적으로 약한 `ohlc_vb`에 더 잘 보완되는지 확인합니다.
    - 선택 feature set: `funding_plus_cftc_oi`, `funding_only`.
    - Runner 준비:
      [kaggle_stage4_n16_4_ohlc_vb_derivatives_repeat_one_cell.md](notebooks/kaggle_stage4_n16_4_ohlc_vb_derivatives_repeat_one_cell.md).
    - 결과: 완료. `ohlc_vb + funding_plus_cftc_oi`는 같은 image Stage 2
      `ohlc_vb` baseline을 `0.567384`에서 `0.569466`으로 개선했습니다
      (`+0.002082`, net correction `+15`). `funding_only`는 거의 동률입니다
      (`+0.000278`, net correction `+2`). 이는 small same-image positive
      result이지, `ohlc_ma_vb` 전체 최고 baseline을 넘은 것은 아닙니다.
    - 결과 노트:
      [4-N16-4 OHLC-VB derivatives repeat](checklist_results/4-N16-4_ohlc_vb_derivatives_repeat.md).
    - Report:
      [stage4_n16_4_ohlc_vb_derivatives_repeat_mean_std_results.csv](reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_mean_std_results.csv).
  - [x] 4-N16-5. Interpretability export.
    - target panel: extreme funding, high/low CFTC OI, leveraged-money long/short
      imbalance, Stage 2 wrong -> FiLM correct, Stage 2 correct -> FiLM wrong.
    - Grad-CAM, context value, gamma/beta summary, `prob_up` change를 export합니다.
    - 결과: tabular interpretation 완료. 선택된
      `ohlc_vb + funding_plus_cftc_oi` FiLM model은 같은 image Stage 2
      `ohlc_vb` baseline 대비 accuracy `+0.002082`, net correction `+15`를
      만들었습니다. 주된 패턴은 higher derivatives/leverage regime에서 약한
      bullish prediction을 낮추는 방향입니다. five-seed targeted
      Stage2/Stage4 Grad-CAM과 Stage4 gamma/beta export도 로컬에서
      완료했고, Kaggle 재현 cell도 준비했습니다.
    - 결과 노트:
      [4-N16-5 derivatives interpretability export](checklist_results/4-N16-5_derivatives_interpretability_export.md).
    - Report:
      [stage4_n16_5_ohlc_vb_derivatives_interpretability_report.md](reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_report.md).
- [x] 4-N14. Final Stage 4 interpretability report
  - 목적: 선택된 Stage 4 model을 단순 metric table이 아니라 논문에 넣을 수 있는
    해석 evidence로 정리합니다.
  - 필수 내용: Stage 2 baseline vs selected context-FiLM metric,
    correction/regression table, predicted-Up distribution, targeted Grad-CAM,
    gamma/beta/modulation-gate summary, 대표적인 `Stage2 wrong -> Stage4 correct`
    및 `Stage2 correct -> Stage4 wrong` sample.
  - output: GitHub와 교수님 보고용 compact report. 큰 bundle/checkpoint는
    local 또는 Kaggle dataset에만 보관합니다.
  - 결과:
    [4-N14 final Stage 4 interpretability report](checklist_results/4-N14_final_stage4_interpretability_report.md).
- [x] 4-N14-B. Conditional regime analysis
  - 목적: 전체 test 평균 accuracy가 크게 오르지 않았더라도, 특정 market regime에서
    context-FiLM이 Stage2 visual-only prediction을 보정했는지 분석합니다.
  - 이 단계는 post-training analysis입니다. 새 모델을 학습하지 않습니다.
  - 첫 번째 분석 대상은 N16 same-image 비교입니다:
    `stage2_i60_ohlc_vb_r20` vs
    `stage4_film_full_bounded_last_block_i60_ohlc_vb_r20_c60_n16d_funding_plus_cftc_oi_pretrained_frozen_s0p02`.
  - 후보 regime: F&G extreme fear/greed, high volatility,
    high derivatives/leverage, high news/macro intensity, Stage2 uncertainty.
  - 좋아 보이는 bucket을 사후에 고르지 않고, 미리 정한 bucket rule만 사용합니다.
  - 보고 기준: bucket당 `100`개 이상 seed-decision, `30`개 이상 unique sample.
    더 작은 bucket은 diagnostic only로 표시합니다.
  - 계획:
    [4-N14-B conditional regime analysis plan](checklist_results/4-N14-B_conditional_regime_analysis_plan.md).
  - [x] 4-N14-B1. Prediction/context merge table.
    - Stage2 prediction, Stage4 prediction, context feature,
      `prob_up_delta`, `true_prob_delta`, transition type을 포함한 long
      decision-level table을 만듭니다.
    - script:
      [build_stage4_n14b_conditional_merge_table.py](scripts/build_stage4_n14b_conditional_merge_table.py).
    - Kaggle runner:
      [kaggle_stage4_n14b1_conditional_merge_table_one_cell.md](notebooks/kaggle_stage4_n14b1_conditional_merge_table_one_cell.md).
    - 결과 summary:
      [stage4_n14b1_n16_derivatives_conditional_merge_report.md](reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_report.md).
    - 참고: full merged-decision CSV는 커서 local/Kaggle-only로 두고, compact
      summary만 GitHub에 올립니다.
  - [x] 4-N14-B2. Predefined regime bucket builder.
    - N14-B1 merged table에서 F&G, volatility, derivatives/leverage,
      news/macro, Stage2 uncertainty bucket을 생성합니다.
  - [x] 4-N14-B3. Bucket-level metrics.
    - bucket별 Stage2 accuracy, Stage4 accuracy, delta,
      correction/regression/net correction, changed-decision rate,
      probability delta를 계산합니다.
  - [x] 4-N14-B4. Representative correction/regression samples.
    - 유용한 bucket마다 Stage2 wrong -> Stage4 correct, Stage2 correct ->
      Stage4 wrong, high-confidence wrong, 큰 `prob_up_delta` sample을 고릅니다.
  - [x] 4-N14-B5. Targeted Grad-CAM/gamma-beta linkage.
    - 선택 sample을 Stage2/Stage4 Grad-CAM, context value, gamma/beta summary,
      probability shift와 연결합니다.
  - [x] 4-N14-B6. Conditional-regime report.
    - context-FiLM correction이 특정 regime에서 해석 가능하게 작동했는지,
      논문에서 conditional improvement로 주장할 수 있는지 정리합니다.
    - 결과:
      [stage4_n14b2_b6_n16_derivatives_conditional_buckets_report.md](reports/tables/stage4_n14b2_b6_n16_derivatives_conditional_buckets_report.md).
    - 논문 핵심 row: uncertain chart + high funding bucket에서 accuracy delta
      `+0.039604`, corrections `24`, regressions `12`.

중요:
- Main Stage 4 실험에서 context 값을 chart image 위에 추가로 그리지 않습니다.
- context는 별도 vector로 들어갑니다.
- 모든 context feature는 image end date `t` 또는 그 이전에 알 수 있어야 합니다.
- context normalization은 train split 통계로만 fit합니다.
- N7 이후 Stage 4의 핵심 risk는 context 품질만이 아닙니다. 이전 Stage 4 run은
  Stage 2 architecture는 재사용했지만 Stage 2 learned weight는 재사용하지
  않았습니다. N8은 더 많은 context source를 추가하기 전에 이 문제를 먼저
  분리합니다.
