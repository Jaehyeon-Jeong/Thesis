# 4-N16-3 Derivatives Feature-Set Grid

## Status

Completed. Five-seed Kaggle run finished and report tables were archived.

Runner:

```text
notebooks/kaggle_stage4_n16_3_derivatives_feature_set_grid_one_cell.md
```

## Fixed Setup

```text
image_window: 60
return_horizon: 20
image_spec: ohlc_ma_vb
context_method: film_full_bounded_last_block
modulation_scale: 0.02
run_seeds: 42, 43, 44, 45, 46
Stage 2: pretrained I60/R20/ohlc_ma_vb checkpoint loaded
freeze_visual_backbone: true
freeze_classifier: true
trainable modules: context encoder + bounded final-block FiLM heads
```

## Feature-Set Grid

Priority follows the N16-2 train-only audit:

| Order | Feature set | Purpose |
| ---: | --- | --- |
| 1 | `funding_only` | Test the cleanest derivatives/leverage signal. |
| 2 | `funding_plus_activity` | Add BitMEX futures activity/volume to funding. |
| 3 | `funding_plus_cftc_oi` | Add official CFTC/CME OI/positioning to funding. |
| 4 | `funding_plus_activity_plus_cftc_oi` | Full derivatives context comparison. |

The runner rebuilds N16 prebuilt context artifacts if they are missing, so it is
safe after a Kaggle working-directory reset as long as the Stage 4 code snapshot
contains `data_inventory/crypto_derivatives/`.

## Expected Outputs

```text
reports/tables/stage4_n16_3_derivatives_feature_set_grid_seed_results.csv
reports/tables/stage4_n16_3_derivatives_feature_set_grid_mean_std_results.csv
reports/tables/stage4_n16_3_derivatives_feature_set_grid_correction_seed_summary.csv
reports/tables/stage4_n16_3_derivatives_feature_set_grid_correction_transition_summary.csv
reports/tables/stage4_n16_3_derivatives_feature_set_grid_run_summary.json
/kaggle/working/stage4_n16_3_derivatives_feature_set_grid_result_bundle.zip
```

## Result Summary

Same-image Stage 2 baseline:

```text
stage2_i60_ohlc_ma_vb_r20
accuracy_mean: 0.579320
roc_auc_mean: 0.584862
f1_mean: 0.651071
brier_score_mean: 0.274337
predicted_positive_rate_mean: 0.664400
```

N16-3 results:

| feature set | accuracy mean | delta vs Stage 2 | ROC-AUC mean | Brier mean | net corrections |
| --- | ---: | ---: | ---: | ---: | ---: |
| `funding_plus_cftc_oi` | 0.579320 | +0.000000 | 0.584983 | 0.273684 | 0 |
| `funding_only` | 0.579181 | -0.000139 | 0.585001 | 0.273910 | -1 |
| `funding_plus_activity_plus_cftc_oi` | 0.578765 | -0.000555 | 0.585054 | 0.273724 | -4 |
| `funding_plus_activity` | 0.578765 | -0.000555 | 0.584957 | 0.273855 | -4 |

No feature set produced a meaningful accuracy or correction improvement. ROC-AUC
and Brier moved only at the fourth decimal place, while F1 decreased relative to
the visual baseline. Predicted-Up rates stayed in the same high range as the
Stage 2 baseline, so this was not a seed-collapse failure.

## Decision

Continue to N16-4 only if at least one N16-3 row shows a meaningful signal:

```text
accuracy_mean > same-image Stage 2 baseline
or ROC-AUC improves without collapse
or net correction is positive and changed-decision rate is non-trivial
```

If no feature set improves, derivatives/leverage context should be reported as
a tested but non-improving market-context family.

N16-3 did not satisfy the continuation rule. N16-4 is therefore low priority and
should be skipped unless a narrow exploratory `ohlc_vb + funding_only` repeat is
needed for completeness.

## Linked Tables

```text
reports/tables/stage4_n16_3_derivatives_feature_set_grid_seed_results.csv
reports/tables/stage4_n16_3_derivatives_feature_set_grid_mean_std_results.csv
reports/tables/stage4_n16_3_derivatives_feature_set_grid_correction_seed_summary.csv
reports/tables/stage4_n16_3_derivatives_feature_set_grid_correction_transition_summary.csv
```
