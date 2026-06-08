# 4-N16-2 Derivatives Feature Audit

## Status

Completed locally.

Feature-ranking signals below are train-only. Stage 2 error correlations are test diagnostics only.

## Stage 2 Diagnostic

|   num_test_rows |   stage2_coverage |   stage2_correct_rate_mean |   stage2_error_rate_mean |   stage2_prob_up_mean |   stage2_pred_up_rate_mean |   stage2_seed_count_min |   stage2_seed_count_max |
|----------------:|------------------:|---------------------------:|-------------------------:|----------------------:|---------------------------:|------------------------:|------------------------:|
|            1441 |                 1 |                    0.57932 |                  0.42068 |              0.597935 |                     0.6644 |                       5 |                       5 |

## Group Summary

| group                   |   num_features | best_feature                   |   max_train_only_candidate_score |   mean_train_only_candidate_score |   max_train_abs_auc_lift |   max_abs_train_spearman_future_return |   max_abs_train_spearman_end_index |   max_test_abs_spearman_stage2_error_rate |   mean_missing_rate_train |
|:------------------------|---------------:|:-------------------------------|---------------------------------:|----------------------------------:|-------------------------:|---------------------------------------:|-----------------------------------:|------------------------------------------:|--------------------------:|
| bitmex_funding          |             15 | funding_rate_max_7             |                         0.44145  |                          0.262768 |                 0.19649  |                               0.44145  |                           0.633919 |                                 0.047794  |                         0 |
| bitmex_activity         |             13 | bitmex_foreignnotional_mean_60 |                         0.299252 |                          0.2002   |                 0.149626 |                               0.286188 |                           0.621386 |                                 0.0511459 |                         0 |
| cftc_cme_oi_positioning |             18 | cot_open_interest              |                         0.229254 |                          0.109438 |                 0.112771 |                               0.229254 |                           0.875175 |                                 0.108596  |                         0 |

## Top Train-Only Features

| feature                        | group           |   train_only_candidate_score |   train_univariate_auc |   train_spearman_future_return |   train_spearman_end_index |   test_spearman_stage2_error_rate |
|:-------------------------------|:----------------|-----------------------------:|-----------------------:|-------------------------------:|---------------------------:|----------------------------------:|
| funding_rate_max_7             | bitmex_funding  |                     0.44145  |               0.30351  |                      -0.44145  |                 -0.0795212 |                       -0.0199078  |
| funding_rate_max_20            | bitmex_funding  |                     0.404114 |               0.306704 |                      -0.404114 |                 -0.1245    |                       -0.0352113  |
| funding_rate_sum_20            | bitmex_funding  |                     0.347915 |               0.326042 |                      -0.332716 |                  0.209308  |                       -0.0477825  |
| funding_rate_mean_20           | bitmex_funding  |                     0.347897 |               0.326051 |                      -0.332676 |                  0.209331  |                       -0.047794   |
| funding_rate_mean_7            | bitmex_funding  |                     0.308719 |               0.345641 |                      -0.304435 |                  0.237322  |                       -0.0285151  |
| funding_rate_sum_7             | bitmex_funding  |                     0.308701 |               0.34565  |                      -0.304428 |                  0.237359  |                       -0.0285509  |
| bitmex_foreignnotional_mean_60 | bitmex_activity |                     0.299252 |               0.350374 |                      -0.286188 |                 -0.297131  |                        0.0124645  |
| bitmex_volume_mean_60          | bitmex_activity |                     0.299252 |               0.350374 |                      -0.286188 |                 -0.297131  |                        0.0124645  |
| funding_rate_max_60            | bitmex_funding  |                     0.274852 |               0.362574 |                      -0.241318 |                 -0.251931  |                        0.0105997  |
| funding_rate_mean_60           | bitmex_funding  |                     0.266975 |               0.375608 |                      -0.266975 |                  0.278844  |                       -0.00150938 |

## Selected Candidate Features

|   selected_rank | feature                           | group                   |   train_only_candidate_score |   train_univariate_auc |   train_spearman_future_return |   train_spearman_end_index |
|----------------:|:----------------------------------|:------------------------|-----------------------------:|-----------------------:|-------------------------------:|---------------------------:|
|               1 | funding_rate_max_7                | bitmex_funding          |                     0.44145  |               0.30351  |                      -0.44145  |                 -0.0795212 |
|               2 | funding_rate_max_20               | bitmex_funding          |                     0.404114 |               0.306704 |                      -0.404114 |                 -0.1245    |
|               3 | funding_rate_mean_7               | bitmex_funding          |                     0.308719 |               0.345641 |                      -0.304435 |                  0.237322  |
|               4 | bitmex_foreignnotional_mean_20    | bitmex_activity         |                     0.244417 |               0.396875 |                      -0.244417 |                 -0.210095  |
|               5 | funding_rate_min_7                | bitmex_funding          |                     0.216479 |               0.39176  |                      -0.207626 |                  0.375225  |
|               6 | cot_lev_money_net_ratio           | cftc_cme_oi_positioning |                     0.193763 |               0.406876 |                      -0.193763 |                 -0.60915   |
|               7 | cot_dealer_net_ratio              | cftc_cme_oi_positioning |                     0.178833 |               0.441223 |                      -0.131779 |                 -0.303346  |
|               8 | bitmex_turnover_mean_7            | bitmex_activity         |                     0.163868 |               0.441456 |                      -0.163868 |                 -0.57676   |
|               9 | bitmex_trades_mean_7              | bitmex_activity         |                     0.161015 |               0.443368 |                      -0.161015 |                 -0.136744  |
|              10 | cot_open_interest_pct_change_60   | cftc_cme_oi_positioning |                     0.14897  |               0.43675  |                      -0.14897  |                  0.0939028 |
|              11 | cot_lev_money_net_ratio_change_20 | cftc_cme_oi_positioning |                     0.122508 |               0.561254 |                       0.101005 |                 -0.0321976 |

## Redundancy Notes

Internal high-correlation pairs: `14`.
Prior-context max absolute correlation median: `0.7268`.

## Decision

Use this audit to prioritize N16-3 feature-set grids. A strong candidate should have low missing rate, non-trivial train-only label/future-return signal, and not be purely redundant with previous F&G/technical context.
