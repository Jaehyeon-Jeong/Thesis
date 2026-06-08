# 4-N16-5 Derivatives/Leverage Interpretability Export

## Status

Completed for tabular interpretation and local five-seed targeted Grad-CAM/gamma-beta export. A Kaggle one-cell is also prepared so the export can be reproduced after Kaggle resets.

## Selected Candidate

```text
Stage2 baseline: stage2_i60_ohlc_vb_r20
Stage4 candidate: stage4_film_full_bounded_last_block_i60_ohlc_vb_r20_c60_n16d_funding_plus_cftc_oi_pretrained_frozen_s0p02
Context: BitMEX funding + release-lagged CFTC/CME Bitcoin futures OI/positioning
FiLM: bounded final-block gamma/beta, scale 0.02
Frozen policy: Stage2 visual CNN frozen, Stage2 classifier frozen
```

## Metric Result

| model | accuracy mean | ROC-AUC mean | Brier mean | predicted-Up mean |
| --- | ---: | ---: | ---: | ---: |
| Stage2 `ohlc_vb` | 0.567384 | 0.561247 | 0.258306 | 0.741013 |
| N16-4 `ohlc_vb + funding+CFTC` | 0.569466 | 0.561820 | 0.257451 | 0.728938 |
| delta | 0.002082 | 0.000573 | -0.000855 | -0.012075 |

The candidate improves the same-image `ohlc_vb` Stage 2 baseline by `+0.002082` accuracy and lowers Brier score by `-0.000855`. It does not exceed the stronger `ohlc_ma_vb` visual baseline, so this is a same-image context-complement result.

## Correction Pattern

Across five seeds:

```text
Stage2 wrong -> FiLM correct: 66
Stage2 correct -> FiLM wrong: 51
Net corrections: 15
Changed decisions: 117 / 7205 (1.6239%)
```

The clearest pattern is bearish correction. In `Stage2 wrong -> FiLM correct` samples, the average future return is `-0.0610`, while in `Stage2 correct -> FiLM wrong` samples it is `0.1270`. The FiLM model tends to reduce `prob_up`:

```text
Correction samples mean prob_up delta: -0.0128
Regression samples mean prob_up delta: -0.0138
Correction samples mean true-prob delta: 0.0148
Regression samples mean true-prob delta: -0.0151
```

This supports the interpretation that derivatives/CFTC context acted as a risk/leverage regime signal that sometimes suppresses weak Stage2 Up calls into Down calls.

## Context Contrast

Correction samples have higher funding context than stable both-correct/both-wrong groups:

```text
funding_rate_mean_20_normalized:
  correction: 0.3022
  regression: 0.2575

funding_rate_max_7_normalized:
  correction: 0.0296
  regression: -0.1366

cot_open_interest_change_20_normalized:
  correction: 0.4066
  regression: 0.2823
```

This does not prove causality, but it gives a defensible interpretation: when funding/leverage context indicates a hotter derivatives regime, the FiLM head slightly down-modulates the visual model's bullish probability.

## Targeted Grad-CAM And Modulation Export

Five-seed targeted exports were generated for the same selected sample table:

```text
Stage2 Grad-CAM files: 5 figures + 5 sample tables
Stage4 Grad-CAM files: 5 figures + 5 sample tables
Stage4 modulation files: 5 summary CSVs + 5 raw JSON files
Selected Stage4 targeted rows: 70
```

The modulation values confirm that the bounded FiLM layer is conservative:

```text
block4_gamma_mean:       1.000082
block4_delta_gamma_mean: 0.000082
block4_beta_mean:        0.000018
block4_delta_gamma_l2:   0.027434
block4_beta_l2:          0.028753
```

Panel-level `prob_up` shifts:

```text
correction:                      -0.029176
regression:                      -0.028333
extreme_high_funding_20d:        -0.019017
extreme_low_funding_20d:         -0.003220
leveraged_money_short_pressure:  -0.007708
leveraged_money_long_pressure:   -0.008994
```

So the correction mechanism is not a large visual re-attention effect. It is a
small final-block probability suppression effect. It helps when the suppressed
Up call was false, and hurts when the suppressed Up call was true.

## Deliverables

```text
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_selected_for_gradcam_augmented.csv
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_selected_sample_view.csv
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_transition_context_summary.csv
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_report.md
notebooks/kaggle_stage4_n16_5_derivatives_interpretability_export_one_cell.md
reports/figures/gradcam/*n16_5_derivatives_targeted_label*
../stage2_btc_extension/reports/figures/gradcam/*n16_5_derivatives_targeted_label*
stage4_n16_5_derivatives_interpretability_local_bundle.zip
```

## Thesis Use

This is the strongest Stage 4 interpretability result so far:

1. It is not an overall best model.
2. It is a controlled same-image improvement on `ohlc_vb`.
3. The improvement is small but directionally consistent across seeds.
4. The correction behavior is interpretable: the context-FiLM module mostly reduces weak bullish calls during higher derivatives/leverage conditions.
