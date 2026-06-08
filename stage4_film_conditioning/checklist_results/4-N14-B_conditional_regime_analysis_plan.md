# 4-N14-B Conditional Regime Analysis Plan

## Status

Prepared. This is a post-training analysis track. It does not train a new model.

## Purpose

Stage 4 average accuracy does not substantially improve over the strongest
Stage 2 visual baseline. N14-B asks a narrower thesis question:

```text
In which predefined market regimes does context-FiLM correct Stage 2
visual-only mistakes, and is the correction direction interpretable?
```

## Analysis Scope

Primary first case:

```text
Stage2 baseline: stage2_i60_ohlc_vb_r20
Stage4 candidate: stage4_film_full_bounded_last_block_i60_ohlc_vb_r20_c60_n16d_funding_plus_cftc_oi_pretrained_frozen_s0p02
Context: BitMEX funding + release-lagged CFTC/CME Bitcoin futures OI/positioning
Reason: strongest current same-image positive case, acc +0.002082 and net +15 corrections.
```

Additional candidates can reuse the same process:

```text
F&G: Stage2 ohlc_ma_vb vs ohlc_ma_vb + F&G FiLM
News: Stage2 ohlc_ma_vb vs ohlc_ma_vb + news SVD FiLM
FSI/RORO: Stage2 ohlc_ma_vb vs ohlc_ma_vb + FSI/RORO FiLM
```

Do not compare N16 `ohlc_vb` directly against the overall-best
`ohlc_ma_vb` baseline as a performance claim. N16 is a same-image
context-complement result.

## Fixed Workflow

### N14-B1. Prediction/context merge table

Build one long decision-level table:

```text
run_seed
sample_index
Date
label
future_return
stage2_prob_up
stage2_pred
stage2_correct
stage4_prob_up
stage4_pred
stage4_correct
prob_up_delta
true_prob_delta
transition_type
context raw/normalized columns
```

Transition labels:

```text
both_correct
both_wrong
correction = Stage2 wrong -> Stage4 correct
regression = Stage2 correct -> Stage4 wrong
```

Prepared script:

```text
scripts/build_stage4_n14b_conditional_merge_table.py
```

Prepared Kaggle runner:

```text
notebooks/kaggle_stage4_n14b1_conditional_merge_table_one_cell.md
```

Kaggle reset dependency:

```text
If /kaggle/working was reset, attach the Stage 2 N15-A output bundle:
stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15

N14-B1 needs Stage 2 test_predictions.csv files, not only Stage 2 code.
The prepared runner auto-detects the folder or zip form of this bundle.
```

Expected outputs:

```text
reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_merged_decisions.csv
reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_seed_summary.csv
reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_transition_summary.csv
reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_context_feature_inventory.csv
reports/tables/stage4_n14b1_n16_derivatives_conditional_merge_report.md
```

### N14-B2. Predefined regime buckets

Do not choose buckets after seeing results. Use predefined rules.

F&G:

```text
extreme_fear: fg_value <= 25
extreme_greed: fg_value >= 75
or quantile fallback: low 20%, middle 60%, high 20%
```

Volatility:

```text
high_rv_60: rv_60 top 20%
high_bb_bandwidth_60: bb_bandwidth_60 top 20%
low_volatility: rv_60 bottom 20%
```

Derivatives/leverage:

```text
high_funding: funding_rate_mean_20 top 20%
extreme_funding: funding_rate_max_7 top 20%
high_cftc_oi_change: cot_open_interest_change_20 top 20%
leveraged_short_pressure: relevant CFTC short-pressure feature top 20%
leveraged_long_pressure: relevant CFTC long-pressure feature top 20%
```

News/macro:

```text
high_news_count: news_count_7d or 20d top 20%
high_news_vector_norm: news SVD vector norm top 20%
high_fsi: OFR FSI top 20%
high_roro_riskoff: RORO proxy top 20%
```

Stage2 uncertainty:

```text
uncertain_45_55: 0.45 <= stage2_prob_up <= 0.55
uncertain_40_60: 0.40 <= stage2_prob_up <= 0.60
confident_up: stage2_prob_up >= 0.70
confident_down: stage2_prob_up <= 0.30
```

### N14-B3. Bucket metric table

For each bucket:

```text
bucket_name
num_decisions
num_unique_samples
stage2_accuracy
stage4_accuracy
delta_accuracy
correction_count
regression_count
net_correction
changed_decision_rate
mean_prob_up_delta
mean_true_prob_delta
mean_future_return_correction
mean_future_return_regression
```

Minimum reporting threshold:

```text
num_decisions >= 100
num_unique_samples >= 30
```

Smaller buckets may be kept as diagnostic only.

### N14-B4. Representative sample extraction

Select samples in this order:

```text
1. Stage2 wrong -> Stage4 correct
2. Stage2 correct -> Stage4 wrong
3. Stage2 high-confidence wrong
4. largest negative prob_up_delta
5. largest positive prob_up_delta
```

### N14-B5. Targeted Grad-CAM/gamma-beta linkage

For representative samples, link or export:

```text
Stage2 Grad-CAM
Stage4 Grad-CAM
label / future_return
stage2_prob_up
stage4_prob_up
prob_up_delta
context raw values
gamma/beta summary
bucket name
```

### N14-B6. Conditional regime report

Final markdown report sections:

```text
1. Objective
2. Compared models
3. Predefined regime buckets
4. Main conditional result table
5. Best positive regimes
6. Failure/regression regimes
7. Representative correction/regression samples
8. Thesis wording
```

## Thesis Wording Target

Use this only if the bucket results support it:

```text
Although context-FiLM did not substantially improve average accuracy over the
strongest visual baseline, conditional regime analysis shows that
derivatives/leverage context produced interpretable corrections in high-leverage
regimes by suppressing weak bullish visual predictions.
```

If the bucket result is weak, use the conservative version:

```text
Conditional analysis suggests that the context-FiLM module behaves as a
conservative probability correction mechanism, but the effect is small and not
strong enough to claim broad regime-specific superiority.
```
