# Stage 4 v1 Interpretation Report

## Setup

- Model family: `I60/R20/ohlc_ma_vb`
- Context window: `60`
- Main interpretation target: `film_full`
- Context features: F&G value, F&G 60-day summaries, BB60 %B/bandwidth, MFI60, RV60
- This report does not retrain models. It reads saved five-seed output artifacts.

## Stage 4 v1 Result Summary

| context_method | seed_count | accuracy_mean | accuracy_std | roc_auc_mean | roc_auc_std | f1_mean | f1_std | brier_score_mean | long_flat_sharpe_net_mean | long_short_sharpe_net_mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| film_full | 5 | 0.5510 | 0.0520 | 0.5677 | 0.0388 | 0.5393 | 0.3016 | 0.2688 | 2.7866 | 1.6953 |
| film_gamma | 5 | 0.5442 | 0.0500 | 0.5559 | 0.0337 | 0.5212 | 0.2907 | 0.2719 | 2.8095 | 1.4262 |
| concat | 5 | 0.5414 | 0.0421 | 0.5496 | 0.0548 | 0.6042 | 0.0833 | 0.2635 | 2.8911 | 1.3807 |
| gating | 5 | 0.5366 | 0.0448 | 0.5530 | 0.0237 | 0.4988 | 0.2805 | 0.2696 | 2.3810 | 0.6143 |

## Stage 2 Baseline Comparison

| model | accuracy_mean | accuracy_std | roc_auc_mean | roc_auc_std | f1_mean | f1_std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Stage 2 selected baseline | 0.5793 | 0.0182 | 0.5849 | 0.0233 | 0.6511 | 0.0070 |
| Stage 4 v1 film_full | 0.5510 | 0.0520 | 0.5677 | 0.0388 | 0.5393 | 0.3016 |

Delta for `film_full` versus Stage 2 selected baseline:

- Accuracy delta: `-0.0283`
- ROC-AUC delta: `-0.0172`
- F1 delta: `-0.1118`

Interpretation:

- Stage 4 v1 does not robustly beat the Stage 2 selected baseline.
- `film_full` is still the most relevant Stage 4 method because it is the best
  FiLM variant and showed promising individual seeds.
- The main issue is instability, not only low average performance.

## Good Seed vs Bad Seed

| role | seed | accuracy | roc_auc | f1 |
| :--- | :--- | :--- | :--- | :--- |
| good_seed | 42 | 0.5843 | 0.5968 | 0.6805 |
| bad_or_collapse_seed | 45 | 0.4587 | 0.5018 | 0.0000 |

Prediction diagnostics for all `film_full` seeds:

| context_method | run_seed | predictions_available | num_predictions | predicted_positive_rate | true_positive_rate | correct_rate | prob_up_mean | prob_up_std | prob_up_q05 | prob_up_q50 | prob_up_q95 | correct_up_count | incorrect_up_count | correct_down_count | incorrect_down_count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| film_full | 42 | True | 1441 | 0.7599 | 0.5413 | 0.5843 | 0.6700 | 0.2313 | 0.2303 | 0.7170 | 0.9571 | 638 | 457 | 204 | 142 |
| film_full | 43 | True | 1441 | 0.7002 | 0.5413 | 0.5760 | 0.6146 | 0.2264 | 0.1900 | 0.6454 | 0.9192 | 589 | 420 | 241 | 191 |
| film_full | 44 | True | 1441 | 0.7988 | 0.5413 | 0.5690 | 0.6632 | 0.1944 | 0.2793 | 0.6979 | 0.9169 | 655 | 496 | 165 | 125 |
| film_full | 45 | True | 1441 | 0.0000 | 0.5413 | 0.4587 | 0.4769 | 0.0065 | 0.4658 | 0.4769 | 0.4873 | 0 | 0 | 661 | 780 |
| film_full | 46 | True | 1441 | 0.8078 | 0.5413 | 0.5670 | 0.6655 | 0.1807 | 0.3258 | 0.6946 | 0.9090 | 660 | 504 | 157 | 120 |

Interpretation:

- A seed with very low F1 or extreme predicted-positive rate indicates class
  collapse.
- If the bad seed shows a much more one-sided prediction distribution than the
  good seed, the v1 FiLM path is likely overpowering or destabilizing image
  evidence.
- This should be treated as diagnostic evidence, not as a market interpretation.

## Correct/Incorrect Case Table

The table below selects high-confidence correct/incorrect Up/Down cases from
the best and worst `film_full` seeds. Use these sample IDs when inspecting
Grad-CAM images and modulation exports.

| sample_index | Date | label_end_date | future_return | label | pred_class | correct | prob_down | prob_up | case | context_method | run_seed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 996 | 2021-01-19T00:00:00 | 2021-02-08T00:00:00 | 0.2921 | 1 | 1 | 1 | 0.0031 | 0.9969 | correct_up | film_full | 42 |
| 995 | 2021-01-18T00:00:00 | 2021-02-07T00:00:00 | 0.0591 | 1 | 1 | 1 | 0.0050 | 0.9950 | correct_up | film_full | 42 |
| 2199 | 2024-05-06T00:00:00 | 2024-05-26T00:00:00 | 0.0846 | 1 | 1 | 1 | 0.0086 | 0.9914 | correct_up | film_full | 42 |
| 991 | 2021-01-14T00:00:00 | 2021-02-03T00:00:00 | -0.0390 | 0 | 1 | 0 | 0.0100 | 0.9900 | incorrect_up | film_full | 42 |
| 984 | 2021-01-07T00:00:00 | 2021-01-27T00:00:00 | -0.2299 | 0 | 1 | 0 | 0.0110 | 0.9890 | incorrect_up | film_full | 42 |
| 983 | 2021-01-06T00:00:00 | 2021-01-26T00:00:00 | -0.1170 | 0 | 1 | 0 | 0.0115 | 0.9885 | incorrect_up | film_full | 42 |
| 2246 | 2024-06-22T00:00:00 | 2024-07-12T00:00:00 | -0.0992 | 0 | 0 | 1 | 0.9638 | 0.0362 | correct_down | film_full | 42 |
| 2233 | 2024-06-09T00:00:00 | 2024-06-29T00:00:00 | -0.1244 | 0 | 0 | 1 | 0.9594 | 0.0406 | correct_down | film_full | 42 |
| 1456 | 2022-04-24T00:00:00 | 2022-05-14T00:00:00 | -0.2373 | 0 | 0 | 1 | 0.9542 | 0.0458 | correct_down | film_full | 42 |
| 1231 | 2021-09-11T00:00:00 | 2021-10-01T00:00:00 | 0.0657 | 1 | 0 | 0 | 0.9625 | 0.0375 | incorrect_down | film_full | 42 |
| 2256 | 2024-07-02T00:00:00 | 2024-07-22T00:00:00 | 0.0869 | 1 | 0 | 0 | 0.9163 | 0.0837 | incorrect_down | film_full | 42 |
| 1235 | 2021-09-15T00:00:00 | 2021-10-05T00:00:00 | 0.0696 | 1 | 0 | 0 | 0.9113 | 0.0887 | incorrect_down | film_full | 42 |
| 1314 | 2021-12-03T00:00:00 | 2021-12-23T00:00:00 | -0.0515 | 0 | 0 | 1 | 0.5430 | 0.4570 | correct_down | film_full | 45 |
| 1825 | 2023-04-28T00:00:00 | 2023-05-18T00:00:00 | -0.0850 | 0 | 0 | 1 | 0.5412 | 0.4588 | correct_down | film_full | 45 |
| 1115 | 2021-05-18T00:00:00 | 2021-06-07T00:00:00 | -0.2170 | 0 | 0 | 1 | 0.5410 | 0.4590 | correct_down | film_full | 45 |
| 1782 | 2023-03-16T00:00:00 | 2023-04-05T00:00:00 | 0.1269 | 1 | 0 | 0 | 0.5451 | 0.4549 | incorrect_down | film_full | 45 |
| 2370 | 2024-10-24T00:00:00 | 2024-11-13T00:00:00 | 0.3252 | 1 | 0 | 0 | 0.5419 | 0.4581 | incorrect_down | film_full | 45 |
| 2369 | 2024-10-23T00:00:00 | 2024-11-12T00:00:00 | 0.3192 | 1 | 0 | 0 | 0.5405 | 0.4595 | incorrect_down | film_full | 45 |

## Modulation Summary

The table written to `stage4_v1_interpretation_film_full_good_bad_modulation.csv` compares saved gamma/beta summaries
from the existing Grad-CAM samples for the good and bad seeds.

Key interpretation rule:

- Grad-CAM explains where the chart image affected the predicted class.
- FiLM gamma/beta summaries explain how market context modulated CNN feature
  channels.
- Because Stage 4 v1 underperformed, gamma/beta should be interpreted as a
  failure diagnostic first, not as validated alpha signal.

## v2 Design Rationale

Stage 4 v2 should be justified from this v1 diagnosis:

1. If the bad seed collapses toward one class, reduce the modulation strength.
2. If gamma/beta summaries are more volatile in the bad seed, use bounded FiLM:
   `gamma = 1 + s_gamma * tanh(raw_gamma)`, `beta = s_beta * tanh(raw_beta)`.
3. If early-layer modulation appears unstable, test last-block-only FiLM.
4. Keep the same data and context features first, so v2 is a structural
   stabilization experiment rather than a different dataset experiment.

Recommended v2 candidates:

- `film_full_bounded_all_blocks`
- `film_full_bounded_last_block`

## Artifacts

- Method summary: `/kaggle/working/stage4_film_conditioning/reports/stage4_v1_interpretation/tables/stage4_v1_interpretation_method_summary.csv`
- Film seed diagnostics: `/kaggle/working/stage4_film_conditioning/reports/stage4_v1_interpretation/tables/stage4_v1_interpretation_film_full_seed_diagnostics.csv`
- Good/bad sample cases: `/kaggle/working/stage4_film_conditioning/reports/stage4_v1_interpretation/tables/stage4_v1_interpretation_film_full_good_bad_cases.csv`
- Good/bad modulation summary: `/kaggle/working/stage4_film_conditioning/reports/stage4_v1_interpretation/tables/stage4_v1_interpretation_film_full_good_bad_modulation.csv`
- Good seed Grad-CAM copy: `/kaggle/working/stage4_film_conditioning/reports/stage4_v1_interpretation/figures/good_seed_film_full_seed42_gradcam.png`
- Bad seed Grad-CAM copy: `/kaggle/working/stage4_film_conditioning/reports/stage4_v1_interpretation/figures/bad_seed_film_full_seed45_gradcam.png`
