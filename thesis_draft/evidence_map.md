# Thesis Evidence Map

This file maps thesis claims to local result artifacts. It is a working index for writing, not the final results section.

## Baseline Evidence

| Claim | Evidence | Artifact |
|---|---|---|
| The strongest BTC visual baseline is Stage2 `I60/R20/ohlc_ma_vb`. | Five-seed accuracy mean `0.579320`, ROC-AUC mean `0.584862`. | `stage4_film_conditioning/checklist_results/4-N14_final_stage4_interpretability_report.md`; `5_9e_results/reports/tables/stage4_n14_final_stage4_interpretability_report.md` |
| The four `I60/R20` image variants form a clear visual-strength ordering. | Stage2 five-seed accuracy means: `ohlc` `0.558085`, `ohlc_ma` `0.557529`, `ohlc_vb` `0.567384`, `ohlc_ma_vb` `0.579320`. | `stage4_film_conditioning/checklist_results/4-N15-A_i60_r20_stage2_four_image_checkpoint_bundle.md`; `stage4_film_conditioning/checklist_results/4-N15_integrated_result_summary.md` |
| Volume-aware chart images are stronger than OHLC-only or MA-only images. | `ohlc_vb` beats `ohlc` by `+0.009299` accuracy and `ohlc_ma_vb` beats `ohlc_ma` by `+0.021791` accuracy. | `stage4_film_conditioning/reports/tables/stage2_n15a_i60_r20_four_image_specs_five_seed_seed_results.csv`; `stage4_film_conditioning/checklist_results/4-N15_integrated_result_summary.md` |
| Stage1 reproduces the original Re-image style pipeline enough to serve as a pipeline reference. | Stage1 I20/R5 and I20/R20 seed-42 results exist; Stage1 pipeline and Grad-CAM outputs were generated. | `stage1_reimage_reproduction/checklist_results/`; local Stage1 output summaries |
| Stage3 linear adapter is a negative/simple-parameter ablation. | Accuracy dropped relative to Stage2 best single-seed result in the initial comparison. | `stage4_film_conditioning/docs/professor_meeting_stage4_direction_brief.md` |

## Stage4 Evidence

| Claim | Evidence | Artifact |
|---|---|---|
| Scratch-trained context concat/gating/FiLM did not robustly beat the strong Stage2 baseline. | Stage4 v1 `film_full` five-seed accuracy mean `0.5510`, ROC-AUC mean `0.5677`, below Stage2 baseline. | `stage4_film_conditioning/checklist_results/4-I13_kaggle_five_seed_runner.md`; `5_9e_results/reports/tables/stage4_n14_final_stage4_interpretability_report.md` |
| Preserving the Stage2 visual model is the most defensible Stage4 protocol. | Frozen Stage2 CNN/classifier + bounded last-block FiLM became the later protocol. | `stage4_film_conditioning/checklist_results/4-N14_final_stage4_interpretability_report.md` |
| F&G is a compact market-regime signal but the gain is small. | N8B F&G-only frozen bounded FiLM: accuracy mean `0.580291`, ROC-AUC mean `0.584930`. | `5_9e_results/reports/tables/stage4_n8b_fg_only_pretrained_frozen_bounded_film_mean_std_results.csv` |
| Chart-derived technical context is mostly redundant with the image. | N15-B same-image accuracy deltas are tiny: `ohlc` `+0.000000`, `ohlc_ma` `+0.000139`, `ohlc_vb` `+0.000139`, `ohlc_ma_vb` `+0.000416`; changed-decision rates stay below `0.5%`. | `stage4_film_conditioning/checklist_results/4-N15-B_image_missing_feature_complement_film.md`; `stage4_film_conditioning/reports/tables/stage4_n15b_image_missing_feature_complement_seed_results.csv` |
| F&G helps volume-aware image specs more than OHLC-only or MA-only specs. | N15-C same-image accuracy deltas: `ohlc` `-0.000416`, `ohlc_ma` `-0.000833`, `ohlc_vb` `+0.000555`, `ohlc_ma_vb` `+0.000972`. | `stage4_film_conditioning/checklist_results/4-N15-C_fg_only_across_image_specs.md`; `stage4_film_conditioning/reports/tables/stage4_n15c_fg_only_across_image_specs_seed_results.csv` |
| Derivatives/leverage context gives a small same-image improvement for `ohlc_vb`. | Stage2 `ohlc_vb` accuracy `0.567384`; N16 `ohlc_vb + funding+CFTC` accuracy `0.569466`; delta `+0.002082`; net corrections `+15`. | `5_9e_results/reports/tables/stage4_n14_final_stage4_interpretability_report.md`; `N14_b1_results/reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_report.md` |
| Derivatives/leverage FiLM mainly suppresses weak bullish predictions. | Mean `prob_up` decreases in several seeds; final report interprets FiLM as small bearish correction in hotter leverage/funding conditions. | `5_9e_results/reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_report.md`; `N14_b1_results/reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_seed_summary.csv` |
| Derivatives/leverage context is most useful when the visual prediction is uncertain and funding is high. | N14-B2..B6 predefined bucket analysis: `uncertain_and_high_funding` bucket has delta accuracy `+0.039604`, corrections `24`, regressions `12`, net `+12`; `funding_mean_20_high20` bucket has delta accuracy `+0.008304`, net `+12`. | `N14_b1_results/reports/tables/stage4_n14b2_b6_n16_derivatives_conditional_buckets_report.md`; `N14_b1_results/reports/tables/stage4_n14b2_b6_n16_derivatives_conditional_buckets_bucket_summary.csv` |

## Stage5 Evidence

| Claim | Evidence | Artifact |
|---|---|---|
| Generic OpenAI embedding context did not clearly beat the Stage2 visual baseline. | 5-8A embedding-only bounded FiLM: accuracy `0.5782`, ROC-AUC `0.5844`, essentially tied but slightly below Stage2. | `stage5_llm_news_embedding/checklist_results/5-8A_embedding_only_bounded_film_results.md` |
| Increasing embedding dimension/scale did not solve the problem. | 5-8C dim32 scale 0.05: accuracy `0.5768`, ROC-AUC `0.5847`; not a better main path. | `stage5_llm_news_embedding/checklist_results/5-8C_dim32_s0p05_embedding_film_results.md` |
| FinBERT-only news sentiment improves ranking metrics more than hard accuracy. | 5-9D: accuracy `0.578487`, ROC-AUC `0.586072`, AP `0.611943`; Up-bias observed. | `stage5_llm_news_embedding/checklist_results/5-9D_finbert_only_bounded_film_results.md` |
| FinBERT+F&G is the best current Stage5 candidate by accuracy, but the gain is small. | 5-9E five-seed mean: accuracy `0.580569`, ROC-AUC `0.585843`, AP `0.611899`, Brier `0.272701`; Stage2 baseline accuracy `0.579320`. | `5_9e_results/reports/tables/stage5_9e_finbert_fg_sentiment_pretrained_frozen_bounded_film_s0p02_seed_results.csv` |
| Stage5 5-9E changes few decisions, so conditional analysis is required. | Across 7205 decisions: correction `95`, regression `86`, net `+9`. | `5_9e_results/reports/tables/stage5_9e_stage2_vs_finbert_fg_correction_transition_summary.csv` |
| FinBERT+F&G helps most when Stage2 is uncertain or the market/news regime is active. | Stage2 uncertain 45-55 bucket: delta accuracy `+0.012484`; F&G greed bucket: delta accuracy `+0.010849`; high 7-day news-count bucket: delta accuracy `+0.007509`. | `stage5_llm_news_embedding/reports/tables/stage5_5_11_finbert_fg_condition_analysis_report.md`; `stage5_llm_news_embedding/checklist_results/5-11_finbert_fg_condition_analysis.md` |
| FinBERT+F&G is weaker in neutral or low-news regimes. | F&G neutral bucket: delta accuracy `-0.006751`; low 60-day news-count bucket: delta accuracy `-0.008276`; extreme-fear bucket: delta accuracy `-0.004138`. | `stage5_llm_news_embedding/reports/tables/stage5_5_11_finbert_fg_condition_analysis_bucket_summary.csv` |
| Stage5 FinBERT+F&G FiLM acts as a conservative calibration layer. | In 5-12 targeted correction/regression export, correction and regression panels both have gamma mean about `1.0003` and beta mean about `0.00008`; selected samples show mainly downward `prob_up` calibration. | `stage5_llm_news_embedding/checklist_results/5-12_gradcam_modulation_export.md`; `stage5_llm_news_embedding/reports/tables/stage5_5_12_finbert_fg_targeted_gradcam_modulation_by_panel.csv`; `stage5_llm_news_embedding/reports/figures/gradcam/*5_12_finbert_fg_targeted_label*` |
| Stage5 is closed for the first thesis draft with a cautious title and claim. | 5-13 recommends `Context-Conditioned FiLM for Bitcoin Direction Prediction from Price Charts`; allowed Stage5 claim is small positive / conditional calibration, not large LLM/news outperformance. | `stage5_llm_news_embedding/checklist_results/5-13_final_stage5_report_title_decision.md`; `stage5_llm_news_embedding/reports/tables/stage5_5_13_final_stage5_report.md` |

## Open Evidence Gaps

These are not finished yet and should be completed before the first full thesis draft.

| Gap | Needed output |
|---|---|
| Citation verification | formal citation table for Re-image, FiLM, Grad-CAM, LLM/embedding/news papers |
| License verification | data/model/API source and redistribution constraints |
