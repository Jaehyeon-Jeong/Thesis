# 4-N14-B1 Conditional Merge Table

Status: prepared by joining Stage 2 predictions, Stage 4 predictions, and context features.

## Inputs

- Analysis name: `n16_ohlc_vb_funding_plus_cftc_oi`
- Stage 2 experiment: `stage2_i60_ohlc_vb_r20`
- Stage 4 experiment: `stage4_film_full_bounded_last_block_i60_ohlc_vb_r20_c60_n16d_funding_plus_cftc_oi_pretrained_frozen_s0p02`
- Context artifact: `stage4_derivatives_context_i60_ohlc_vb_r20_n16d_funding_plus_cftc_oi`
- Split: `test`
- Seeds: `42, 43, 44, 45, 46`

## Output

- Merged decisions: `/kaggle/working/stage4_film_conditioning/reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_merged_decisions.csv`
- Seed summary: `/kaggle/working/stage4_film_conditioning/reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_seed_summary.csv`
- Transition summary: `/kaggle/working/stage4_film_conditioning/reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_transition_summary.csv`
- Context feature inventory: `/kaggle/working/stage4_film_conditioning/reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_context_feature_inventory.csv`

## Seed Summary

| analysis_name                    |   run_seed |   num_decisions |   stage2_accuracy |   stage4_accuracy |   delta_accuracy |   stage2_predicted_up_rate |   stage4_predicted_up_rate |   mean_prob_up_delta |   mean_true_prob_delta |   correction_count |   regression_count |   net_correction |   changed_decision_rate |
|:---------------------------------|-----------:|----------------:|------------------:|------------------:|-----------------:|---------------------------:|---------------------------:|---------------------:|-----------------------:|-------------------:|-------------------:|-----------------:|------------------------:|
| n16_ohlc_vb_funding_plus_cftc_oi |         42 |            1441 |          0.594726 |          0.599584 |         0.004858 |                   0.810548 |                   0.780708 |            -0.011633 |              -0.000835 |                 25 |                 18 |                7 |                0.029840 |
| n16_ohlc_vb_funding_plus_cftc_oi |         43 |            1441 |          0.564192 |          0.564885 |         0.000694 |                   0.753643 |                   0.725191 |            -0.014906 |              -0.000885 |                 21 |                 20 |                1 |                0.028452 |
| n16_ohlc_vb_funding_plus_cftc_oi |         44 |            1441 |          0.546842 |          0.548924 |         0.002082 |                   0.621096 |                   0.624566 |             0.001469 |               0.000236 |                  5 |                  2 |                3 |                0.004858 |
| n16_ohlc_vb_funding_plus_cftc_oi |         45 |            1441 |          0.562804 |          0.564192 |         0.001388 |                   0.945177 |                   0.934074 |            -0.006979 |              -0.000417 |                  9 |                  7 |                2 |                0.011103 |
| n16_ohlc_vb_funding_plus_cftc_oi |         46 |            1441 |          0.568355 |          0.569743 |         0.001388 |                   0.574601 |                   0.580153 |             0.002458 |               0.000512 |                  6 |                  4 |                2 |                0.006940 |

## Transition Counts

| analysis_name                    |   run_seed |   both_correct |   both_wrong |   correction |   regression |
|:---------------------------------|-----------:|---------------:|-------------:|-------------:|-------------:|
| n16_ohlc_vb_funding_plus_cftc_oi |         42 |            839 |          559 |           25 |           18 |
| n16_ohlc_vb_funding_plus_cftc_oi |         43 |            793 |          607 |           21 |           20 |
| n16_ohlc_vb_funding_plus_cftc_oi |         44 |            786 |          648 |            5 |            2 |
| n16_ohlc_vb_funding_plus_cftc_oi |         45 |            804 |          621 |            9 |            7 |
| n16_ohlc_vb_funding_plus_cftc_oi |         46 |            815 |          616 |            6 |            4 |

## Context Feature Inventory

| feature_type      |   num_columns |
|:------------------|--------------:|
| imputed_clipped   |           165 |
| missing_indicator |           245 |
| normalized        |           165 |
| raw_or_metadata   |           335 |

## Next Step

Use the merged decision table for N14-B2 bucket construction. The first
recommended buckets are high funding, high CFTC OI change, Stage2 uncertainty,
and transition groups (`correction`, `regression`, `both_correct`, `both_wrong`).

## Required Columns For N14-B2

- `stage2_correct`, `stage4_correct`, `transition_type`
- `prob_up_stage2`, `prob_up_stage4`, `prob_up_delta`
- `stage2_uncertain_45_55`, `stage2_uncertain_40_60`
- `context__*_normalized` and raw context columns
