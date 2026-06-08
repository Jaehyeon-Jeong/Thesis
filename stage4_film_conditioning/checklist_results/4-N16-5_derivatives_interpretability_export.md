# 4-N16-5 Derivatives/Leverage Interpretability Export

## Goal

Use the best N16-4 same-image positive result to check whether the FiLM
correction has an interpretable pattern.

Selected candidate:

```text
image: I60/R20/ohlc_vb
context: BitMEX funding + release-lagged CFTC/CME Bitcoin futures OI/positioning
model: Stage2 CNN/classifier frozen + bounded last-block FiLM
scale: 0.02
experiment: stage4_film_full_bounded_last_block_i60_ohlc_vb_r20_c60_n16d_funding_plus_cftc_oi_pretrained_frozen_s0p02
```

## Result

N16-5 is completed for tabular interpretation and local five-seed targeted
Stage2/Stage4 Grad-CAM plus Stage4 gamma/beta modulation export. A targeted
Kaggle export runner is also prepared for reproducibility after Kaggle resets.

Same-image metric comparison:

| model | accuracy mean | ROC-AUC mean | Brier mean | predicted-Up mean |
| --- | ---: | ---: | ---: | ---: |
| Stage2 `ohlc_vb` | 0.567384 | 0.561247 | 0.258306 | 0.741013 |
| N16-4 `ohlc_vb + funding+CFTC` | 0.569466 | 0.561820 | 0.257451 | 0.728938 |
| delta | +0.002082 | +0.000573 | -0.000855 | -0.012075 |

The model does not beat the strongest visual baseline (`ohlc_ma_vb`), but it
does improve the same-image `ohlc_vb` baseline. This is the clearest Stage 4
context-complement result so far.

## Correction Pattern

Across five seeds:

```text
Stage2 wrong -> FiLM correct: 66
Stage2 correct -> FiLM wrong: 51
Net corrections: +15
Changed decisions: 117 / 7205 = 1.6239%
```

The main effect is not broad re-ranking. It is a small bearish correction:

```text
Correction samples mean future_return: -0.0610
Regression samples mean future_return:  0.1270

Correction samples mean prob_up delta: -0.0128
Regression samples mean prob_up delta: -0.0138
```

So the derivatives/CFTC FiLM head usually pushes `prob_up` slightly downward.
That helps when the Stage2 model was weakly bullish before a negative 20-day
return, but it hurts when the future return is strongly positive.

## Context Interpretation

Correction samples have higher leverage/funding context than regression samples:

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

This is not causal proof, but it is a defensible mechanism: derivatives context
acts as a leverage/risk regime suppressor for weak bullish visual predictions.

## Targeted Grad-CAM And Modulation Export

Local five-seed targeted exports were generated:

```text
Stage2 Grad-CAM: 5 figures + 5 sample tables
Stage4 Grad-CAM: 5 figures + 5 sample tables
Stage4 modulation: 5 summary CSVs + 5 raw JSON files
Targeted Stage4 rows: 70
```

The bounded FiLM modulation is intentionally small:

```text
block4_gamma_mean:       1.000082
block4_delta_gamma_mean: 0.000082
block4_beta_mean:        0.000018
block4_delta_gamma_l2:   0.027434
block4_beta_l2:          0.028753
```

Panel-level `prob_up` shifts show the same bearish-suppression pattern:

```text
correction:                     -0.029176
regression:                     -0.028333
extreme_high_funding_20d:       -0.019017
extreme_low_funding_20d:        -0.003220
leveraged_money_short_pressure: -0.007708
leveraged_money_long_pressure:  -0.008994
```

This is why the result should be described as a small, interpretable
probability correction, not as a broad visual representation improvement.

## Deliverables

```text
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_report.md
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_selected_for_gradcam_augmented.csv
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_selected_sample_view.csv
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_transition_context_summary.csv
notebooks/kaggle_stage4_n16_5_derivatives_interpretability_export_one_cell.md
reports/figures/gradcam/*n16_5_derivatives_targeted_label*
../stage2_btc_extension/reports/figures/gradcam/*n16_5_derivatives_targeted_label*
stage4_n16_5_derivatives_interpretability_local_bundle.zip
```

## Thesis Note

Phrase this carefully:

> Derivatives/leverage context does not produce a new overall best model, but
> under a controlled same-image comparison it gives a small, stable correction
> to the `ohlc_vb` visual model. The correction is interpretable because the
> FiLM module mostly suppresses weak bullish predictions in higher
> funding/open-interest regimes.
