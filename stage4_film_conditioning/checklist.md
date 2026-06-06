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
- Completed current track: `4-N8-A/B`, Stage 2 pretrained
  baseline-preserving FiLM. Stage 4 now reloads the selected Stage 2
  checkpoint, reproduces the baseline, freezes the visual CNN/classifier, and
  trains only the F&G context encoder plus bounded final-block FiLM heads.
- Current next track: move from crypto/news context to official macro
  risk-regime context. Use OFR FSI as an official financial-stress/risk-off
  proxy and a KC Fed-inspired public-data RORO proxy as separate N13 sources.
- First news version: headline-only, non-LLM, train-only TF-IDF/SVD over
  7/20/60-day trailing news windows.
- Main order now: finish the frozen Stage 2 context-source comparison -> run
  N13 FSI/RORO source audit -> test FSI-only -> test RORO-proxy-only -> compare
  FSI/RORO/F&G/news under the same frozen protocol -> prepare final
  Grad-CAM/gamma-beta interpretation.

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
- [ ] 4-N9. News-only and News + F&G pretrained/frozen ablation
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
- [ ] 4-N11. LLM summary/embedding decision
  - Deferred until headline-only no-leakage track is stable.
  - If used, record model name, prompt, version/date, cache hash, and runtime.
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
- [ ] 4-N13. Macro/RORO context extension
  - Purpose: test image-external macro risk-regime context after F&G/news and
    technical context produced only tiny gains.
  - Thesis question: can official financial stress or risk-on/risk-off regime
    condition the frozen Stage 2 BTC chart features more meaningfully than
    context derived from BTC OHLCV?
  - Shared protocol: `I60/R20/ohlc_ma_vb`, five seeds `42-46`, Stage 2
    checkpoints loaded/frozen, classifier frozen, bounded last-block FiLM first
    with conservative scale `0.02`.
- [ ] 4-N13-0. Macro/RORO source audit and terminology lock
  - Distinguish `OFR FSI` from `RORO`: OFR FSI is not a direct RORO index; it is
    an official financial-stress/risk-off proxy.
  - Record source links, date coverage, CSV load path, missing-date policy, and
    whether each feature is available at or before image end date `t`.
  - Candidate source 1: OFR Financial Stress Index CSV, covering 2000-present.
  - Candidate source 2: public-data RORO proxy inspired by KC Fed methodology,
    built from FRED/Cboe indicators rather than proprietary/refinitiv series.
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
- [ ] 4-N13-5A. Cross-context feature audit
  - Merge already-built context features on the same sample/date index:
    F&G, news SVD/count, technical BB/MFI/RV, OFR FSI, public RORO, label,
    future return, Stage 2 `prob_up`, and Stage 2 `correct`.
  - Use train split only for feature selection diagnostics. Validation/test are
    for confirmation only.
  - Audit: missing rate, feature-label correlation, feature-future-return
    correlation, feature-Stage2-error correlation, feature-feature correlation,
    and redundancy clusters.
  - Output: a small interpretable selected feature list, not a large
    all-context vector.
- [ ] 4-N13-5B. Selected-combo context FiLM
  - Run one controlled selected-combo context experiment only if 4-N13-5A finds
    a non-redundant feature set.
  - Candidate size: roughly 3-6 features.
  - Example candidates: `fg_value`, `fg_delta_60`, one news SVD/count feature,
    `roro_proxy_value`, `roro_proxy_delta_20`, `fsi_delta_60`.
  - Stage 2 frozen protocol, bounded last-block FiLM, five seeds, scale `0.02`
    and optionally `0.05`.
  - Decision rule: keep only if it improves or clarifies the source comparison
    without reducing interpretability.
- [ ] 4-N13-6. Macro interpretability export
  - Target samples: Stage 2 wrong -> N13 correct, Stage 2 correct -> N13 wrong,
    and high-stress / low-stress regimes.
  - Export targeted Grad-CAM, FSI/RORO values, gamma/beta summaries,
    modulation gate if used, and `prob_up` changes.
- [ ] 4-N13-7. Final FiLM constraint and scale ablation on the selected context
  source
  - Run only after 4-N13-5/6 select the strongest stable context source
    (`F&G`, `news`, `FSI`, or `RORO`).
  - Purpose: test whether the current bounded FiLM was too conservative under
    the Stage 2 frozen protocol.
  - Baseline-preserving fixed setup: Stage 2 I60/R20/ohlc_ma_vb visual CNN and
    classifier frozen, same split, same seeds, same selected context features.
  - A. Same bounded equation, larger scale, no new model code required:
    bounded last-block FiLM scale `0.02`, `0.05`, `0.10`, `0.20`.
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
- [ ] 4-N13-B. Optional sentiment/event feature extension
  - Run only if headline TF-IDF/SVD is too weak or hard to interpret after N13
    macro/RORO planning.
  - Candidate features: FinBERT-style sentiment score, positive/negative/neutral
    counts, crypto-regulation/exchange/ETF/macro event tags, or cached
    headline-level sentiment/event labels.
  - Leakage rule: sentiment/event extraction must use only headlines available
    by strict `t-1`; encoder/model/version/date/cache hash must be recorded.
  - Purpose: test whether explicit news polarity/event type is more useful than
    unsupervised TF-IDF/SVD vectors for context-FiLM correction.
- [ ] 4-N14. Final Stage 4 interpretability report
  - Purpose: turn the selected Stage 4 model into thesis-ready evidence, not
    just another metric table.
  - Required content: Stage 2 baseline vs selected context-FiLM metrics,
    correction/regression table, predicted-Up distribution, targeted Grad-CAM,
    gamma/beta/modulation-gate summaries, and representative
    `Stage2 wrong -> Stage4 correct` plus `Stage2 correct -> Stage4 wrong`
    samples.
  - Output: compact report for GitHub and professor update; large bundles stay
    local/Kaggle dataset only.

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
- 완료된 현재 track: `4-N8-A/B`, Stage 2 pretrained baseline-preserving
  FiLM입니다. Stage 4 code path에서 선택된 Stage 2 checkpoint를 불러와 baseline을
  재현했고, visual CNN/classifier를 freeze한 뒤 F&G context encoder와 bounded
  final-block FiLM head만 학습했습니다.
- 현재 다음 track: 같은 pretrained/frozen FiLM path로 news-only를 먼저 테스트하고,
  이후 필요하면 F&G + news combined context와 Stage 2 vs N8-B 해석 비교로
  넘어갑니다.
- 첫 news version: headline-only, non-LLM, train-only TF-IDF/SVD를 7/20/60-day
  trailing news window에 적용합니다.
- 현재 순서: N8-B 결과 반영 -> news-only pretrained/frozen FiLM 실행 가능 여부 결정
  -> news-only가 유망하면 F&G + news combined context -> Grad-CAM/gamma-beta 해석
  -> Stage 4 최종 보고와 교수님 보고 정리.

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
- [ ] 4-N9. News-only와 News + F&G pretrained/frozen ablation
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
- [ ] 4-N11. LLM summary/embedding decision
  - Headline-only no-leakage track이 안정화된 뒤로 미룹니다.
  - 사용한다면 model name, prompt, version/date, cache hash, runtime을
    기록해야 합니다.
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
- [ ] 4-N13. Macro/RORO context extension
  - 목적: F&G/news/technical context가 작은 개선만 보였기 때문에, 이미지 밖
    macro risk-regime context를 frozen Stage 2 FiLM 구조에 넣어봅니다.
  - thesis question: 공식 financial stress 또는 risk-on/risk-off regime이 BTC
    OHLCV에서 파생한 context보다 Stage 2 visual feature를 더 의미 있게
    condition할 수 있는지 확인합니다.
  - shared protocol: `I60/R20/ohlc_ma_vb`, seeds `42-46`, Stage 2 checkpoint
    loaded/frozen, classifier frozen, bounded last-block FiLM, 우선 conservative
    scale `0.02`.
- [ ] 4-N13-0. Macro/RORO source audit and terminology lock
  - `OFR FSI`와 `RORO`를 명확히 구분합니다. OFR FSI는 직접 RORO가 아니라
    공식 financial-stress/risk-off proxy입니다.
  - source link, date coverage, CSV load path, missing-date policy, image end
    date `t` 기준 사용 가능성을 기록합니다.
  - source 1: OFR Financial Stress Index CSV, 2000-present coverage.
  - source 2: KC Fed methodology를 참고한 public-data RORO proxy. KC Fed의
    proprietary/full input을 복제한다고 쓰지 않습니다.
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
- [ ] 4-N13-5A. Cross-context feature audit
  - 이미 만든 context feature들을 같은 sample/date index 기준으로 merge합니다:
    F&G, news SVD/count, technical BB/MFI/RV, OFR FSI, public RORO, label,
    future return, Stage 2 `prob_up`, Stage 2 `correct`.
  - feature selection diagnostic은 train split만 사용합니다. validation/test는
    확인용으로만 둡니다.
  - audit 항목: missing rate, feature-label correlation, feature-future-return
    correlation, feature-Stage2-error correlation, feature-feature correlation,
    redundancy cluster.
  - output: 큰 all-context vector가 아니라 작고 해석 가능한 selected feature list.
- [ ] 4-N13-5B. Selected-combo context FiLM
  - 4-N13-5A에서 중복이 적은 feature set이 보일 때만 selected-combo context
    실험을 한 번 실행합니다.
  - 후보 크기: 대략 3-6개 feature.
  - 예시 후보: `fg_value`, `fg_delta_60`, news SVD/count 1개,
    `roro_proxy_value`, `roro_proxy_delta_20`, `fsi_delta_60`.
  - Stage 2 frozen protocol, bounded last-block FiLM, five seeds, scale `0.02`
    그리고 필요하면 `0.05`.
  - 결정 기준: 성능을 높이거나 source comparison 해석을 명확하게 만들 때만
    유지합니다.
- [ ] 4-N13-6. Macro interpretability export
  - target sample: Stage 2 wrong -> N13 correct, Stage 2 correct -> N13 wrong,
    high-stress / low-stress regime.
  - targeted Grad-CAM, FSI/RORO value, gamma/beta summary, modulation gate,
    `prob_up` change를 export합니다.
- [ ] 4-N13-7. 선택된 context source에 대한 최종 FiLM constraint/scale ablation
  - 4-N13-5/6에서 가장 안정적인 context source(`F&G`, `news`, `FSI`, `RORO`)를
    고른 뒤에만 실행합니다.
  - 목적: Stage 2 frozen protocol에서는 현재 bounded FiLM이 너무 보수적이었는지
    확인합니다.
  - 고정 조건: Stage 2 I60/R20/ohlc_ma_vb visual CNN과 classifier freeze, 같은
    split, 같은 seeds, 같은 selected context feature.
  - A. 같은 bounded equation에서 scale만 키우는 grid, 새 모델 코드 불필요:
    bounded last-block FiLM scale `0.02`, `0.05`, `0.10`, `0.20`.
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
- [ ] 4-N13-B. Optional sentiment/event feature extension
  - N13 macro/RORO planning 이후에도 headline TF-IDF/SVD가 너무 약하거나
    해석하기 어려울 때만 실행합니다.
  - 후보 feature: FinBERT-style sentiment score, positive/negative/neutral count,
    crypto regulation/exchange/ETF/macro event tag, 또는 cached headline-level
    sentiment/event label.
  - Leakage rule: sentiment/event extraction은 strict `t-1`까지 사용 가능한
    headline만 써야 하며 encoder/model/version/date/cache hash를 기록해야 합니다.
  - 목적: 명시적 news polarity/event type이 unsupervised TF-IDF/SVD vector보다
    context-FiLM correction에 더 유용한지 확인합니다.
- [ ] 4-N14. Final Stage 4 interpretability report
  - 목적: 선택된 Stage 4 model을 단순 metric table이 아니라 논문에 넣을 수 있는
    해석 evidence로 정리합니다.
  - 필수 내용: Stage 2 baseline vs selected context-FiLM metric,
    correction/regression table, predicted-Up distribution, targeted Grad-CAM,
    gamma/beta/modulation-gate summary, 대표적인 `Stage2 wrong -> Stage4 correct`
    및 `Stage2 correct -> Stage4 wrong` sample.
  - output: GitHub와 교수님 보고용 compact report. 큰 bundle/checkpoint는
    local 또는 Kaggle dataset에만 보관합니다.

중요:
- Main Stage 4 실험에서 context 값을 chart image 위에 추가로 그리지 않습니다.
- context는 별도 vector로 들어갑니다.
- 모든 context feature는 image end date `t` 또는 그 이전에 알 수 있어야 합니다.
- context normalization은 train split 통계로만 fit합니다.
- N7 이후 Stage 4의 핵심 risk는 context 품질만이 아닙니다. 이전 Stage 4 run은
  Stage 2 architecture는 재사용했지만 Stage 2 learned weight는 재사용하지
  않았습니다. N8은 더 많은 context source를 추가하기 전에 이 문제를 먼저
  분리합니다.
