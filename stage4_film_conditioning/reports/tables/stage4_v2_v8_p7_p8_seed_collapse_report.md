# 4-V8 P7/P8 Seed-Collapse Diagnostic

## Purpose

- Compare P7 `film_full` and P8 `film_full_bounded_last_block` without retraining.
- Check whether seed collapse is caused by threshold/class calibration rather than loss of ranking signal.
- Calibrate a threshold on validation and apply it to test.

## Settings

- Image: `I60/R20/ohlc_ma_vb`
- Context window: `60`
- Methods: `film_full, film_full_bounded_last_block`
- Seeds: `42, 43, 44, 45, 46`
- Calibration metric: `balanced_accuracy`

## Default Threshold Summary

| context_method               |   run_seed | split      | collapse_flag   | collapse_direction   |   accuracy |   balanced_accuracy |   roc_auc |       f1 |   predicted_positive_rate |   tp |   tn |   fp |   fn |
|:-----------------------------|-----------:|:-----------|:----------------|:---------------------|-----------:|--------------------:|----------:|---------:|--------------------------:|-----:|-----:|-----:|-----:|
| film_full                    |         42 | test       | False           | none                 |   0.580153 |            0.574673 |  0.594333 | 0.623053 |                  0.572519 |  500 |  336 |  325 |  280 |
| film_full_bounded_last_block |         42 | test       | False           | none                 |   0.587092 |            0.570697 |  0.593914 | 0.668524 |                  0.704372 |  600 |  246 |  415 |  180 |
| film_full                    |         43 | test       | False           | none                 |   0.505205 |            0.476245 |  0.477955 | 0.644034 |                  0.848716 |  645 |   83 |  578 |  135 |
| film_full_bounded_last_block |         43 | test       | True            | mostly_down          |   0.501735 |            0.533512 |  0.583165 | 0.244211 |                  0.117974 |  116 |  607 |   54 |  664 |
| film_full                    |         44 | test       | True            | mostly_up            |   0.512838 |            0.482142 |  0.464745 | 0.654867 |                  0.870229 |  666 |   73 |  588 |  114 |
| film_full_bounded_last_block |         44 | test       | True            | mostly_down          |   0.458709 |            0.5      |  0.537781 | 0        |                  0        |    0 |  661 |    0 |  780 |
| film_full                    |         45 | test       | False           | none                 |   0.580153 |            0.575019 |  0.603057 | 0.621639 |                  0.568355 |  497 |  339 |  322 |  283 |
| film_full_bounded_last_block |         45 | test       | False           | none                 |   0.586398 |            0.573287 |  0.587782 | 0.657077 |                  0.664816 |  571 |  274 |  387 |  209 |
| film_full                    |         46 | test       | False           | none                 |   0.583622 |            0.568069 |  0.592544 | 0.662921 |                  0.693963 |  590 |  251 |  410 |  190 |
| film_full_bounded_last_block |         46 | test       | False           | none                 |   0.578765 |            0.555734 |  0.57876  | 0.682032 |                  0.783484 |  651 |  183 |  478 |  129 |
| film_full                    |         42 | validation | False           | none                 |   0.728223 |            0.709117 |  0.796251 | 0.782123 |                  0.630662 |  140 |   69 |   41 |   37 |
| film_full_bounded_last_block |         42 | validation | False           | none                 |   0.714286 |            0.678891 |  0.783513 | 0.781915 |                  0.69338  |  147 |   58 |   52 |   30 |
| film_full                    |         43 | validation | True            | mostly_up            |   0.609756 |            0.527042 |  0.478069 | 0.735849 |                  0.860627 |  156 |   19 |   91 |   21 |
| film_full_bounded_last_block |         43 | validation | True            | mostly_down          |   0.442509 |            0.525655 |  0.49887  | 0.272727 |                  0.149826 |   30 |   97 |   13 |  147 |
| film_full                    |         44 | validation | False           | none                 |   0.627178 |            0.551489 |  0.657268 | 0.743405 |                  0.836237 |  155 |   25 |   85 |   22 |
| film_full_bounded_last_block |         44 | validation | True            | mostly_down          |   0.383275 |            0.5      |  0.561222 | 0        |                  0        |    0 |  110 |    0 |  177 |
| film_full                    |         45 | validation | False           | none                 |   0.724739 |            0.709733 |  0.780534 | 0.776204 |                  0.61324  |  137 |   71 |   39 |   40 |
| film_full_bounded_last_block |         45 | validation | False           | none                 |   0.728223 |            0.693631 |  0.768259 | 0.792553 |                  0.69338  |  149 |   60 |   50 |   28 |
| film_full                    |         46 | validation | False           | none                 |   0.700348 |            0.667591 |  0.758552 | 0.768817 |                  0.679443 |  143 |   58 |   52 |   34 |
| film_full_bounded_last_block |         46 | validation | False           | none                 |   0.686411 |            0.637365 |  0.69584  | 0.769231 |                  0.74216  |  150 |   47 |   63 |   27 |

## Validation Threshold Calibration

| context_method               |   run_seed |   threshold |   validation_threshold_score |   test_default_accuracy |   test_calibrated_accuracy |   test_accuracy_delta |   test_default_balanced_accuracy |   test_calibrated_balanced_accuracy |   test_balanced_accuracy_delta |   test_default_predicted_positive_rate |   test_calibrated_predicted_positive_rate | test_calibrated_collapse_flag   | test_calibrated_collapse_direction   |
|:-----------------------------|-----------:|------------:|-----------------------------:|------------------------:|---------------------------:|----------------------:|---------------------------------:|------------------------------------:|-------------------------------:|---------------------------------------:|------------------------------------------:|:--------------------------------|:-------------------------------------|
| film_full                    |         42 |    0.542672 |                     0.737006 |                0.580153 |                   0.564885 |           -0.0152672  |                         0.574673 |                            0.563225 |                   -0.0114483   |                               0.572519 |                                  0.52533  | False                           | none                                 |
| film_full_bounded_last_block |         42 |    0.604709 |                     0.721777 |                0.587092 |                   0.576683 |           -0.0104094  |                         0.570697 |                            0.570083 |                   -0.000613872 |                               0.704372 |                                  0.585704 | False                           | none                                 |
| film_full                    |         43 |    0.496533 |                     0.543246 |                0.505205 |                   0.528105 |            0.0229008  |                         0.476245 |                            0.492206 |                    0.0159607   |                               0.848716 |                                  0.934074 | True                            | mostly_up                            |
| film_full_bounded_last_block |         43 |    0.500531 |                     0.529712 |                0.501735 |                   0.494795 |           -0.00693963 |                         0.533512 |                            0.527679 |                   -0.00583324  |                               0.117974 |                                  0.104094 | True                            | mostly_down                          |
| film_full                    |         44 |    0.50629  |                     0.651027 |                0.512838 |                   0.476752 |           -0.0360861  |                         0.482142 |                            0.472813 |                   -0.0093293   |                               0.870229 |                                  0.545455 | False                           | none                                 |
| film_full_bounded_last_block |         44 |    0.450338 |                     0.558166 |                0.458709 |                   0.538515 |            0.0798057  |                         0.5      |                            0.529287 |                    0.0292874   |                               0        |                                  0.614157 | False                           | none                                 |
| film_full                    |         45 |    0.572362 |                     0.728043 |                0.580153 |                   0.560028 |           -0.0201249  |                         0.575019 |                            0.561161 |                   -0.0138582   |                               0.568355 |                                  0.491325 | False                           | none                                 |
| film_full_bounded_last_block |         45 |    0.611818 |                     0.712198 |                0.586398 |                   0.569049 |           -0.0173491  |                         0.573287 |                            0.568456 |                   -0.00483145  |                               0.664816 |                                  0.512838 | False                           | none                                 |
| film_full                    |         46 |    0.58042  |                     0.714407 |                0.583622 |                   0.58848  |            0.00485774 |                         0.568069 |                            0.581442 |                    0.0133733   |                               0.693963 |                                  0.59195  | False                           | none                                 |
| film_full_bounded_last_block |         46 |    0.559342 |                     0.668207 |                0.578765 |                   0.573907 |           -0.00485774 |                         0.555734 |                            0.557133 |                    0.00139843  |                               0.783484 |                                  0.707842 | False                           | none                                 |

## Pairwise P7/P8 Comparison

|   run_seed | split      |   prediction_agreement_rate |   prob_up_correlation |   right_minus_left_prob_up_mean |   left_predicted_positive_rate |   right_predicted_positive_rate |   p7_up_p8_down_count |   p7_down_p8_up_count |   p7_up_p8_down_label_positive_rate |   p7_down_p8_up_label_positive_rate |
|-----------:|:-----------|----------------------------:|----------------------:|--------------------------------:|-------------------------------:|--------------------------------:|----------------------:|----------------------:|------------------------------------:|------------------------------------:|
|         42 | test       |                    0.696044 |            0.532483   |                       0.0860886 |                       0.572519 |                        0.704372 |                   124 |                   314 |                            0.524194 |                            0.525478 |
|         43 | test       |                    0.231783 |            0.00736001 |                      -0.0159273 |                       0.848716 |                        0.117974 |                  1080 |                    27 |                            0.509259 |                            0.777778 |
|         44 | test       |                    0.129771 |            0.0930507  |                      -0.0541933 |                       0.870229 |                        0        |                  1254 |                     0 |                            0.5311   |                          nan        |
|         45 | test       |                    0.698126 |            0.555911   |                       0.060523  |                       0.568355 |                        0.664816 |                   148 |                   287 |                            0.533784 |                            0.533101 |
|         46 | test       |                    0.74948  |            0.502701   |                       0.0569579 |                       0.693963 |                        0.783484 |                   116 |                   245 |                            0.543103 |                            0.506122 |
|         42 | validation |                    0.783972 |            0.69123    |                       0.0266619 |                       0.630662 |                        0.69338  |                    22 |                    40 |                            0.636364 |                            0.525    |
|         43 | validation |                    0.268293 |            0.0488601  |                      -0.0147346 |                       0.860627 |                        0.149826 |                   207 |                     3 |                            0.618357 |                            0.666667 |
|         44 | validation |                    0.163763 |            0.154069   |                      -0.0572449 |                       0.836237 |                        0        |                   240 |                     0 |                            0.645833 |                          nan        |
|         45 | validation |                    0.759582 |            0.60878    |                       0.0345758 |                       0.61324  |                        0.69338  |                    23 |                    46 |                            0.565217 |                            0.543478 |
|         46 | validation |                    0.721254 |            0.526663   |                       0.0558431 |                       0.679443 |                        0.74216  |                    31 |                    49 |                            0.548387 |                            0.489796 |

## Next Decision Rule

- If validation threshold calibration fixes seed 43/44 on test, prioritize calibrated evaluation/checkpoint criteria.
- If ROC-AUC is good but class distribution collapses, avoid blind gamma/beta scale search.
- If both validation and test collapse in the same direction, inspect train history and gamma/beta modulation exports for those seeds.
