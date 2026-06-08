# 4-N16-2 Derivatives Feature Audit

## Status

Completed locally.

Script:

```text
scripts/build_stage4_n16_2_derivatives_feature_audit.py
```

Main report:

```text
reports/tables/stage4_n16_2_derivatives_feature_audit_report.md
```

## Inputs

Derivatives context:

```text
outputs/stage4/context/stage4_derivatives_context_i60_ohlc_ma_vb_r20_n16_derivatives_all/seed_42/context_features.csv
```

Stage 2 baseline predictions:

```text
stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15/outputs/stage2/predictions/stage2_i60_ohlc_ma_vb_r20/seed_42..46/test_predictions.csv
```

Prior-context redundancy references:

```text
outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60_fg_only/seed_42/context_features.csv
outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_features.csv
```

## Method

Feature ranking uses train-only signals:

```text
max(
  abs(correlation with train label),
  abs(correlation with train future_return),
  2 * abs(univariate AUC - 0.5)
)
```

The following are diagnostics only and are not used as selection evidence:

```text
test split Stage 2 error correlation
train end_index/time-trend correlation
```

## Group Result

| Group | Features | Best feature | Best train-only score | Note |
| --- | ---: | --- | ---: | --- |
| BitMEX funding | 15 | `funding_rate_max_7` | `0.4414` | strongest train-only signal |
| BitMEX activity | 13 | `bitmex_foreignnotional_mean_60` | `0.2993` | useful but partly redundant with volume/volatility |
| CFTC/CME OI/positioning | 18 | `cot_open_interest` | `0.2293` | weaker train-only signal; some time-trend risk |

Stage 2 diagnostic baseline:

```text
test rows: 1441
Stage 2 five-seed correct-rate mean: 0.57932
Stage 2 five-seed error-rate mean:   0.42068
```

## Top Train-Only Features

| Rank | Feature | Group | Train-only score | Train future-return Spearman | Time-trend Spearman |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `funding_rate_max_7` | BitMEX funding | `0.4414` | `-0.4414` | `-0.0795` |
| 2 | `funding_rate_max_20` | BitMEX funding | `0.4041` | `-0.4041` | `-0.1245` |
| 3 | `funding_rate_sum_20` | BitMEX funding | `0.3479` | `-0.3327` | `0.2093` |
| 4 | `funding_rate_mean_20` | BitMEX funding | `0.3479` | `-0.3327` | `0.2093` |
| 5 | `funding_rate_mean_7` | BitMEX funding | `0.3087` | `-0.3044` | `0.2373` |
| 6 | `bitmex_foreignnotional_mean_60` | BitMEX activity | `0.2993` | `-0.2862` | `-0.2971` |

Interpretation:

```text
Higher funding / higher derivatives activity in the train period is associated
with lower future R20 returns. This is economically plausible as a crowded-long
or leverage-overheated regime signal, but it must be validated by N16-3 because
the train period is small and time-regime effects can be strong.
```

## Redundancy

Internal redundancy:

```text
14 pairs above abs(train correlation) >= 0.85
```

Examples:

```text
funding_rate_mean_20 ~= funding_rate_sum_20
bitmex_volume_mean_20 ~= bitmex_foreignnotional_mean_20
bitmex_turnover_mean_20 ~= bitmex_homenotional_mean_20
```

Prior-context redundancy:

```text
median max abs correlation with F&G/technical context: 0.7268
```

Important caveat:

```text
Several 60-day funding/activity features are highly correlated with `rv_60`.
So N16-3 should not only test `derivatives_all`; it must compare leaner
feature sets such as `funding_only`, `funding_plus_activity`, and
`funding_plus_cftc_oi`.
```

## Decision

N16-2 supports running N16-3.

Priority order for N16-3:

```text
1. funding_only
2. funding_plus_activity
3. funding_plus_cftc_oi
4. funding_plus_activity_plus_cftc_oi
```

The audit suggests funding is the cleanest derivatives/leverage regime signal.
Activity and CFTC should be tested as additions, not assumed to improve the
model.

## Output Tables

```text
reports/tables/stage4_n16_2_derivatives_feature_audit.csv
reports/tables/stage4_n16_2_derivatives_group_summary.csv
reports/tables/stage4_n16_2_derivatives_internal_redundancy_pairs.csv
reports/tables/stage4_n16_2_derivatives_prior_context_redundancy.csv
reports/tables/stage4_n16_2_derivatives_selected_feature_candidates.csv
reports/tables/stage4_n16_2_derivatives_stage2_error_diagnostics.csv
reports/tables/stage4_n16_2_derivatives_manifest.json
```
