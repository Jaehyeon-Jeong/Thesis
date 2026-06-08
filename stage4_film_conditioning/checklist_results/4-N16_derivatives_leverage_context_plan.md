# 4-N16 Derivatives/Leverage Context Plan

## Decision

Add one final Stage 4 context-source track before the final N14 report:

```text
Stage 2 frozen chart CNN
+ BitMEX funding/activity
+ CFTC/CME open interest and positioning
+ bounded last-block FiLM
```

This track is more defensible than another technical-indicator variant because
it uses external derivatives/leverage information that is not directly drawn in
the OHLC/MA/VB chart image.

## Why This Track

Completed Stage 4 results show:

- chart-derived technical context mostly duplicates information already encoded
  in the image;
- F&G is external and gave the best compact improvement, but only at a very
  small scale;
- news, FSI, and RORO preserve or slightly recalibrate the Stage 2 baseline but
  do not materially improve hard decisions;
- N15-C showed F&G helped only on volume-aware image specs, suggesting that
  market-regime context may interact with volume/liquidity information.

Derivatives/leverage context is the next most logical source family because it
captures positioning and leverage pressure rather than chart shape alone.

## Local Data Inventory

### BitMEX XBTUSD Funding

- File:
  `data_inventory/crypto_derivatives/bitmex_xbtusd/processed/bitmex_xbtusd_funding_daily_2018_2024.csv`
- Coverage: `2018-01-01` to `2024-12-31`
- Stage 4 missing rate: `0%`
- Meaning: perpetual swap funding pressure; long/short crowding proxy.

Candidate features:

- `funding_rate_mean_7/20/60`
- `funding_rate_sum_7/20/60`
- `funding_rate_abs_mean_7/20/60`
- `funding_rate_min_20/60`
- `funding_rate_max_20/60`

### BitMEX XBTUSD Derivatives Activity

- File:
  `data_inventory/crypto_derivatives/bitmex_xbtusd/processed/bitmex_xbtusd_derivatives_activity_daily_2018_2024.csv`
- Coverage: `2018-01-01` to `2024-12-31`
- Stage 4 missing rate: `0%`
- Meaning: futures-market participation/volume, not spot chart volume.

Candidate features:

- `bitmex_trades_mean_7/20/60`
- `bitmex_volume_mean_7/20/60`
- `bitmex_turnover_mean_7/20/60`
- `bitmex_home_notional_mean_20/60`
- `bitmex_foreign_notional_mean_20/60`

### CFTC/CME Bitcoin Futures COT Open Interest

- Main CME daily file:
  `data_inventory/crypto_derivatives/cftc_cme_bitcoin_cot/processed/cftc_cme_bitcoin_main_cot_daily_release_lag3_ffill_2018_2024.csv`
- Main + micro aggregate file:
  `data_inventory/crypto_derivatives/cftc_cme_bitcoin_cot/processed/cftc_cme_bitcoin_main_plus_micro_cot_daily_release_lag3_ffill_2018_2024.csv`
- Source frequency: weekly CFTC COT report.
- Stage 4 missing rate after release-lagged forward-fill: `0%`.
- Stage 4 `cot_age_days`: minimum `3`, maximum `10`.

Leakage rule:

```text
available_date = report_date + 3 calendar days
daily value = last available report forward-filled
```

The feature table must keep:

- `cot_source_report_date`
- `cot_age_days`

Candidate features:

- `Open_Interest_All`
- `oi_change_20/60`
- `oi_zscore_60`
- `Lev_Money_Net_Ratio_All`
- `Asset_Mgr_Net_Ratio_All`
- `Dealer_Net_Ratio_All`

## Experimental Order

### 4-N16-1 Feature Builder

Build sample-aligned context features for `I60/R20` with strict availability at
or before the image end date `t`.

Use train-only preprocessing:

- impute on train median;
- clip with train 1/99 percentiles;
- z-score with train mean/std.

### 4-N16-2 Train-Only Audit

Before training FiLM, run a compact feature audit:

- missing rate;
- label/future-return correlation;
- correlation with Stage 2 prediction error;
- redundancy with F&G, technical context, FSI/RORO, and news features;
- feature-feature correlation.

Purpose: avoid a large noisy derivatives vector.

### 4-N16-3 Primary Frozen-FiLM Grid

Run on `I60/R20/ohlc_ma_vb` first:

```text
funding_only
funding_plus_cftc_oi
funding_plus_activity
funding_plus_activity_plus_cftc_oi
```

Fixed protocol:

- seed-matched Stage 2 checkpoint reload;
- frozen visual CNN;
- frozen classifier;
- train only context encoder + bounded final-block FiLM;
- scale `0.02`;
- seeds `42,43,44,45,46`.

Metrics:

- accuracy;
- ROC-AUC;
- Brier;
- F1;
- predicted-Up rate;
- changed-decision rate;
- correction/regression/net correction versus Stage 2;
- collapse warnings.

### 4-N16-4 Volume-Aware Repeat

Only if N16-3 shows signal, repeat the best feature set on `ohlc_vb`.

Reason: N15-C showed F&G produced small positive deltas only on volume-aware
images. A derivatives/leverage signal may interact with volume-aware image
features more naturally than with OHLC-only images.

### 4-N16-5 Interpretability Export

Export targeted cases:

- extreme funding;
- high/low CFTC OI;
- leveraged-money long/short imbalance;
- Stage 2 wrong -> FiLM correct;
- Stage 2 correct -> FiLM wrong.

Required outputs:

- Stage 2 vs Stage 4 Grad-CAM;
- context values;
- gamma/beta summaries;
- `prob_up` changes;
- correction/regression table.

## Reporting Position

If N16 improves:

> Derivatives/leverage context is a more suitable market-regime conditioner than
> chart-derived technical context because it captures external positioning and
> leverage pressure.

If N16 does not improve:

> The frozen Stage 2 chart CNN already captures most useful I60/R20 signal, and
> external context mainly acts as a small calibration/interpretable regime probe
> rather than a large accuracy booster.
