# 4-N13-5B Selected-Combo Context FiLM

## Goal

Run one controlled combined-context experiment after the N13-5A cross-context
audit. This is **not** an all-context stacking run. The purpose is to test
whether a small, non-redundant context vector can improve or clarify the frozen
Stage 2 FiLM setting without adding the noise seen in larger context vectors.

## Selected Context Vector

The primary N13-5B vector has six features:

| Source | Source feature | N13-5B feature |
| --- | --- | --- |
| News TF-IDF/SVD32 | `news_svd_60d_09` | `combo_news_svd_60d_09` |
| News TF-IDF/SVD32 | `news_svd_20d_18` | `combo_news_svd_20d_18` |
| Fear & Greed | `fg_mean_60` | `combo_fg_mean_60` |
| Fear & Greed | `fg_delta_60` | `combo_fg_delta_60` |
| OFR FSI | `ofr_fsi_std_60` | `combo_ofr_fsi_std_60` |
| Public RORO proxy | `riskoff_dollar_return_20` | `combo_riskoff_dollar_return_20` |

These features are selected from the train-only N13-5A audit, with family caps
to avoid simply stacking every available context feature.

## Protocol

- Visual model: Stage 2 `I60/R20/ohlc_ma_vb`.
- Stage 2 CNN and classifier are frozen.
- Context path: selected six-feature vector -> context MLP -> bounded
  last-block FiLM.
- FiLM equation: `gamma = 1 + scale * tanh(raw_gamma)`,
  `beta = scale * tanh(raw_beta)`.
- First scale: `0.02`.
- Seeds: `42, 43, 44, 45, 46`.
- Grad-CAM is deferred to N13-6 if N13-5B gives a useful comparison.

## Local Builder Check

The selected-combo builder was validated locally on seed `42` using already
built source artifacts.

| Check | Value |
| --- | --- |
| Context name | `stage4_selected_combo_context_i60_ohlc_ma_vb_r20_n13_5a_combo6` |
| Rows | `2,399` |
| Split counts | train `671`, validation `287`, test `1,441` |
| Context dimension | `6` |
| Missing normalized values | `0.0` for all six selected features |

## Artifacts

- [selected-combo builder](../scripts/build_stage4_selected_combo_context_features.py)
- [Kaggle one-cell runner](../notebooks/kaggle_stage4_n13_5b_selected_combo_pretrained_frozen_bounded_film_one_cell.md)
- [local builder audit](../reports/tables/stage4_selected_combo_context_i60_ohlc_ma_vb_r20_n13_5a_combo6_seed42_selected_combo_context_feature_audit.json)
- [local builder summary](../reports/tables/stage4_selected_combo_context_i60_ohlc_ma_vb_r20_n13_5a_combo6_seed42_selected_combo_context_feature_summary.csv)
- [local builder manifest](../reports/tables/stage4_selected_combo_context_i60_ohlc_ma_vb_r20_n13_5a_combo6_seed42_selected_combo_context_manifest.json)
- [five-seed seed results](../reports/tables/stage4_n13_5b_selected_combo_pretrained_frozen_bounded_film_seed_results.csv)
- [five-seed mean/std results](../reports/tables/stage4_n13_5b_selected_combo_pretrained_frozen_bounded_film_mean_std_results.csv)
- [five-seed run summary](../reports/tables/stage4_n13_5b_selected_combo_pretrained_frozen_bounded_film_run_summary.json)

## Five-Seed Result

N13-5B completed as a metric-only run. The raw seed table shows `status=failed`
because Grad-CAM was intentionally disabled while the old output checker still
required Grad-CAM artifacts. Classification, trading, checkpoint, prediction,
and context artifacts are available for all five seeds.

| Metric | N13-5B mean | Stage2 frozen baseline | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | `0.579320` | `0.579320` | `+0.000000` |
| ROC-AUC | `0.584866` | `0.584862` | `+0.000004` |
| F1 | `0.650202` | `0.651071` | `-0.000869` |
| Brier score | `0.274135` | `0.274337` | `-0.000202` |
| Predicted Up rate | `0.661346` | `0.664400` | `-0.003054` |
| Net correction | `0.0` | n/a | n/a |

Seed-level correction counts are very small. The selected combo corrected and
regressed the same average number of Stage 2 decisions (`2.6` each), so the net
hard-decision gain is zero.

## Interpretation

The selected six-feature vector is stable and does not collapse, but it mostly
preserves the frozen Stage 2 model rather than adding a stronger decision signal.
The train-only N13-5A feature audit correctly identified individually related
signals, especially news SVD and F&G features, but combining them did not improve
five-seed performance.

This result argues against expanding into a larger all-context vector. For the
final Stage 4 narrative, N13-5B should be treated as a negative/neutral ablation:
small selected context is safer than naive stacking, but the best final
candidates remain compact single-source context rows such as F&G-only `s0.02`
and news SVD32 `s0.02`.
