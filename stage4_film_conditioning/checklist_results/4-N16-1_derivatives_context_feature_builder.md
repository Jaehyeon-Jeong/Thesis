# 4-N16-1 Derivatives Context Feature Builder

## Status

Completed locally.

This step adds a prebuilt context builder for crypto derivatives/leverage
features:

```text
scripts/build_stage4_derivatives_context_features.py
```

The output format matches the existing Stage 4 prebuilt context contract:

```text
context_features.csv
context_features.parquet
context_scaler.json
context_feature_audit.json
context_feature_summary.csv
derivatives_context_manifest.json
```

## Fixed Alignment

Target experiment:

```text
image_window: 60
image_spec: ohlc_ma_vb
return_horizon: 20
run_seed: 42, 43, 44, 45, 46 output paths available
```

Sample counts:

```text
train: 671
validation: 287
test: 1441
```

Source timing:

```text
BitMEX daily funding/activity:
  use source date <= image end date t - 1 day

CFTC/CME COT:
  use release-lagged daily forward-filled file
  no additional lag because source was already shifted by +3 calendar days
```

CFTC audit:

```text
cot_age_days max: 10
cot_age_days mean: about 6.00
```

## Source Files

BitMEX funding:

```text
data_inventory/crypto_derivatives/bitmex_xbtusd/processed/bitmex_xbtusd_funding_daily_2018_2024.csv
```

BitMEX derivatives activity:

```text
data_inventory/crypto_derivatives/bitmex_xbtusd/processed/bitmex_xbtusd_derivatives_activity_daily_2018_2024.csv
```

CFTC/CME Bitcoin futures COT:

```text
data_inventory/crypto_derivatives/cftc_cme_bitcoin_cot/processed/cftc_cme_bitcoin_main_plus_micro_cot_daily_release_lag3_ffill_2018_2024.csv
```

## Generated Feature Sets

| Feature set | Context name | Dim | Groups |
| --- | --- | ---: | --- |
| `funding_only` | `stage4_derivatives_context_i60_ohlc_ma_vb_r20_n16_funding_only` | `15` | BitMEX funding only |
| `funding_plus_cftc_oi` | `stage4_derivatives_context_i60_ohlc_ma_vb_r20_n16_funding_plus_cftc_oi` | `33` | funding + CFTC/CME OI/positioning |
| `funding_plus_activity` | `stage4_derivatives_context_i60_ohlc_ma_vb_r20_n16_funding_plus_activity` | `28` | funding + BitMEX activity |
| `funding_plus_activity_plus_cftc_oi` | `stage4_derivatives_context_i60_ohlc_ma_vb_r20_n16_funding_plus_activity_plus_cftc_oi` | `46` | funding + activity + CFTC/CME |
| `derivatives_all` | `stage4_derivatives_context_i60_ohlc_ma_vb_r20_n16_derivatives_all` | `46` | audit alias for full set |

Every feature set has `context_features.csv`, `context_scaler.json`, and audit
files under:

```text
outputs/stage4/context/<context_name>/seed_42
outputs/stage4/context/<context_name>/seed_43
outputs/stage4/context/<context_name>/seed_44
outputs/stage4/context/<context_name>/seed_45
outputs/stage4/context/<context_name>/seed_46
```

This matters because the Stage 4 prebuilt loader resolves context artifacts by
the current training `run_seed`.

The `46`-dimensional full set consists of:

```text
BitMEX funding: 15 features
BitMEX activity: 13 features
CFTC/CME OI/positioning: 18 features
```

## Missing / Normalization Check

Funding-only and funding-plus-activity artifacts have raw missing rate `0%`
for all splits.

CFTC-containing artifacts have small raw missing rates in early train/validation
rows because some `20/60` CFTC rolling/change features are not available at the
very beginning of 2018:

```text
train max raw missing rate:      about 4.62%
validation max raw missing rate: about 4.53%
test max raw missing rate:       0%
```

This is below the `5%` warning threshold. The model-ready normalized matrix is
finite because preprocessing uses:

```text
train median imputation
train 1/99 percentile clipping
train z-score normalization
```

## Report Tables

Report copies were written under:

```text
reports/tables/stage4_derivatives_context_i60_ohlc_ma_vb_r20_n16_*_seed42_derivatives_context_feature_audit.json
reports/tables/stage4_derivatives_context_i60_ohlc_ma_vb_r20_n16_*_seed42_derivatives_context_feature_summary.csv
reports/tables/stage4_derivatives_context_i60_ohlc_ma_vb_r20_n16_*_seed42_derivatives_context_manifest.json
```

Report copies are written for seed `42`; the model-ready context artifacts are
available for seeds `42-46`.

## Decision

N16-1 is complete.

Next step:

```text
4-N16-2 Train-only derivatives feature audit
```

The audit should check which derivatives features relate to labels, future
returns, Stage 2 errors, and whether they are redundant with previously tested
F&G/technical/news/FSI/RORO context.
