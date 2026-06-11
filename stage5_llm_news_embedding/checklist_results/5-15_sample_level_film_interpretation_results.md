# 5-15. Sample-level FiLM Interpretation Results

## Status

Completed the post-hoc sample-level analysis for the final FinBERT+F&G FiLM
candidate. No new training was performed.

## Inputs

- `reports/tables/stage5_5_11_finbert_fg_condition_analysis_merged_decisions.csv`
- `reports/tables/stage5_5_11_finbert_fg_condition_analysis_selected_samples_for_5_12.csv`
- `reports/tables/stage5_5_12_finbert_fg_targeted_gradcam_modulation_summary.csv`
- Existing clean Grad-CAM candidate:
  `Thesis/thesis_draft/figures/gradcam_finbert_fg_corrections_clean.png`

## Outputs

- Selected case table:
  `reports/tables/stage5_5_15_selected_film_interpretation_cases.csv`
- Compact selected case table:
  `reports/tables/stage5_5_15_selected_film_interpretation_cases_compact.csv`
- Gamma/beta summary:
  `reports/tables/stage5_5_15_gamma_beta_correction_vs_regression_summary.csv`
- Compact gamma/beta summary:
  `reports/tables/stage5_5_15_gamma_beta_correction_vs_regression_summary_compact.csv`
- Gamma/beta plot:
  `reports/figures/stage5_5_15_gamma_beta_correction_vs_regression_plot.png`
- Grad-CAM correction case figure:
  `reports/figures/gradcam/stage5_5_15_gradcam_correction_cases.png`
- Full analysis report:
  `reports/tables/stage5_5_15_sample_level_film_interpretation_report.md`
- Thesis figure copies:
  `Thesis/thesis_draft/figures/gamma_beta_correction_vs_regression_plot.png`
  and `Thesis/thesis_draft/figures/gradcam_correction_cases.png`.

## Selected Cases

The final transition counts remain:

- Corrections: `95`
- Regressions: `86`
- Net corrections: `+9`

The selected analysis cases contain:

- `4` correction cases, all `Stage2 false-Up -> FiLM correct Down`
- `4` regression cases, all `Stage2 true-Up -> FiLM wrong Down`

The first three correction examples are aligned with the clean thesis Grad-CAM
figure. The fourth correction example adds a cleaner fear/negative-news
context case:

- `2021-04-20`, seed `43`, sample `1087`
- `2021-05-19`, seed `43`, sample `1116`
- `2021-07-02`, seed `43`, sample `1160`
- `2021-06-26`, seed `42`, sample `1154`

In all four correction examples, FiLM reduces `P(up)` enough to move the
decision from Up to Down. The three Grad-CAM-aligned examples have shifts of
about `0.047` to `0.051`; C4 is a smaller but cleaner borderline correction
under low F&G and negative-leaning FinBERT context.

## Gamma/Beta Summary

The bounded modulation is intentionally small:

| Group | n | mean delta gamma | mean beta | mean delta-gamma L2 | mean beta L2 |
|---|---:|---:|---:|---:|---:|
| correction | 20 | 0.000334 | 0.000079 | 0.052070 | 0.055608 |
| regression | 20 | 0.000349 | 0.000082 | 0.055658 | 0.058997 |

This confirms the same interpretation as 5-12: FinBERT+F&G FiLM is not
rewriting the visual model. It applies a small, bounded calibration to frozen
visual features. The same downward calibration can fix false-Up cases and can
also create regressions when the original Up prediction was correct.

## Thesis Claim

Use this analysis to support a diagnostic interpretability claim:

- The model can be inspected through context values, probability transition,
  bounded gamma/beta modulation, and Grad-CAM.
- The selected correction examples show conservative downward calibration; C2,
  C3, and C4 are the cleanest bearish or risk-off-like contexts, while C1 is a
  more mixed sentiment-deterioration case.
- The selected regression examples show the limitation: the same downward
  calibration can over-correct true-Up samples.
- The gamma/beta channels are anonymous CNN channels, so they should not be
  translated directly into economic concepts.
- This is not causal proof that a specific headline or F&G value caused the
  return. It is post-hoc diagnostic evidence about how the context branch
  modulated the chart-only model.
