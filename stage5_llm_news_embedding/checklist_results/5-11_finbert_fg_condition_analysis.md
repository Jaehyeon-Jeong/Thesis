# 5-11 FinBERT + F&G Conditional Correction Analysis

## Purpose

This step analyzes where the Stage5 FinBERT+F&G bounded FiLM model improves
or harms the frozen Stage2 `I60/R20/ohlc_ma_vb` visual baseline. It is not a
new training run. It is the condition-level evidence for the thesis
interpretability section.

## Inputs

- Stage2 baseline predictions: `I60/R20/ohlc_ma_vb`
- Stage5 candidate: FinBERT headline sentiment + F&G, Stage2 frozen,
  bounded last-block FiLM, scale `0.02`
- Result root used locally: `../5_9e_results`

## Overall Result

| Metric | Value |
|---|---:|
| Matched decisions | 7,205 |
| Stage2 accuracy | 0.579320 |
| Stage5 accuracy | 0.580569 |
| Delta accuracy | +0.001249 |
| Corrections | 95 |
| Regressions | 86 |
| Net corrections | +9 |
| Changed prediction rate | 2.5121% |
| Mean probability-up delta | -0.016158 |

Interpretation: the model is conservative. It changes only a small subset of
Stage2 decisions, and the net correction is positive but modest. This should be
claimed as conditional correction evidence, not as a large performance gain.

## Stronger Positive Conditions

- Stage2 uncertainty `0.45 <= prob_up <= 0.55`: delta accuracy `+0.012484`,
  corrections `94`, regressions `84`, net `+10`.
- F&G `greed` regime: delta accuracy `+0.010849`, corrections `46`,
  regressions `23`, net `+23`.
- Low 20-day FinBERT negative ratio: delta accuracy `+0.008997`, net `+13`.
- High 60-day F&G mean: delta accuracy `+0.008247`, net `+12`.
- High 7-day news count: delta accuracy `+0.007509`, net `+11`.

## Weaker / Negative Conditions

- Low 60-day news count: delta accuracy `-0.008276`, net `-12`.
- F&G `neutral` regime: delta accuracy `-0.006751`, net `-8`.
- Low 20-day news count: delta accuracy `-0.006164`, net `-9`.
- F&G `extreme_fear`: delta accuracy `-0.004138`, net `-6`.

## Mechanism-Level Reading

The average probability-up shift is negative. Corrections are concentrated in
cases where Stage2 was uncertain or where F&G/news context was active enough to
justify a small correction. This suggests that bounded FiLM is acting more like
a selective calibration layer than a new dominant predictor.

## Outputs

- Full report:
  `reports/tables/stage5_5_11_finbert_fg_condition_analysis_report.md`
- Merged decision table:
  `reports/tables/stage5_5_11_finbert_fg_condition_analysis_merged_decisions.csv`
- Bucket summary:
  `reports/tables/stage5_5_11_finbert_fg_condition_analysis_bucket_summary.csv`
- Transition context summary:
  `reports/tables/stage5_5_11_finbert_fg_condition_analysis_transition_context_summary.csv`
- Selected samples for 5-12:
  `reports/tables/stage5_5_11_finbert_fg_condition_analysis_selected_samples_for_5_12.csv`

## Next Step

Use the selected sample CSV in `5-12` to export targeted Grad-CAM,
gamma/beta summaries, and news/F&G context for:

- Stage2 wrong -> Stage5 correct
- Stage2 correct -> Stage5 wrong
- high-context and weak-context buckets
