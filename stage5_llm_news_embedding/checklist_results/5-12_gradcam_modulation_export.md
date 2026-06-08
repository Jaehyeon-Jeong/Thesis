# 5-12 Grad-CAM and FiLM Modulation Export

## Status

Completed.

## Goal

Export targeted interpretability artifacts for the Stage5 FinBERT+F&G model:

- Stage2 baseline Grad-CAM on selected samples.
- Stage5 FinBERT+F&G Grad-CAM on the same samples.
- FiLM gamma/beta modulation summaries.
- FinBERT/F&G context values and probability changes.

The selected samples come from 5-11 correction/regression analysis and focus on:

- Stage2 wrong -> Stage5 correct.
- Stage2 correct -> Stage5 wrong.

## Compared Models

- Stage2 baseline: `stage2_i60_ohlc_ma_vb_r20`.
- Stage5 candidate:
  `stage4_film_full_bounded_last_block_i60_ohlc_ma_vb_r20_c60_stage5_finbert_fg_sentiment_v1_pretrained_frozen_s0p02`.
- Context:
  `stage5_finbert_fg_context_i60_ohlc_ma_vb_r20_stage5_finbert_fg_sentiment_v1`.
- Seeds: `42,43,44,45,46`.
- Grad-CAM target: true label.

## Outputs

- Report:
  `reports/tables/stage5_5_12_finbert_fg_targeted_gradcam_report.md`
- Modulation summary:
  `reports/tables/stage5_5_12_finbert_fg_targeted_gradcam_modulation_summary.csv`
- Panel-level modulation:
  `reports/tables/stage5_5_12_finbert_fg_targeted_gradcam_modulation_by_panel.csv`
- Stage5 sample table:
  `reports/tables/stage5_5_12_finbert_fg_targeted_gradcam_stage5_samples.csv`
- Stage2 sample table:
  `reports/tables/stage5_5_12_finbert_fg_targeted_gradcam_stage2_samples.csv`
- Grad-CAM artifacts:
  `reports/figures/gradcam/*5_12_finbert_fg_targeted_label*`

Exported Grad-CAM/report artifacts: `30` files.
Selected modulation rows: `40` rows, split into `20` correction and `20`
regression samples.

## Key Quantitative Read

The exported FiLM modulation is very conservative.

| Panel | block4 gamma mean | block4 delta-gamma mean | block4 beta mean |
|---|---:|---:|---:|
| Correction | `1.000334` | `0.000334` | `0.000079` |
| Regression | `1.000349` | `0.000349` | `0.000082` |

The correction and regression panels have almost identical gamma/beta scale.
This means the model is not using FinBERT+F&G to create a large channel-level
rewrite of the chart representation. It is acting as a small bounded
calibration layer.

## Probability-Level Pattern

For the selected samples, Stage5 mainly pushes `prob_up` downward.

| Panel | Mean Stage2 prob_up | Mean Stage5 prob_up | Mean prob_up delta |
|---|---:|---:|---:|
| Correction | `0.523529` | `0.486073` | `-0.037456` |
| Regression | `0.517489` | `0.478454` | `-0.039035` |

This explains both sides:

- Corrections happen when the Stage2 model was weakly bullish but the true
  class was Down.
- Regressions happen when the same bearish correction pushes true-Up samples
  below the decision threshold.

## Thesis Interpretation

5-12 supports a cautious interpretability claim:

> FinBERT+F&G context does not strongly rewrite the visual representation.
> Instead, bounded last-block FiLM applies a small calibration to the final
> visual features. This calibration is useful in some uncertain or active
> sentiment/regime cases, but it can also create regressions when the same
> downward probability shift is applied to true-Up samples.

This is consistent with 5-11: the overall gain is small, so the thesis should
frame Stage5 as conditional context calibration, not as a large performance
breakthrough.

## Caveats

- This is post-hoc analysis only; no model was trained or tuned in 5-12.
- Grad-CAM figures should be used as qualitative visual evidence, not as a
  standalone quantitative proof.
- Raw news text is not redistributed; only numeric context summaries are
  exported.
