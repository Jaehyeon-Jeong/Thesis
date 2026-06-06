# 4-N13-1 OFR FSI Feature Builder

Status: completed.

Latest Kaggle upload snapshot prepared locally:

```text
/Users/jaehyeonjeong/Desktop/논문/stage4_film_conditioning_n13_1_latest.zip
```

This zip excludes old `outputs/` and result bundles, but keeps the Stage 4
code, notebooks, reports, `FG_data`, and `news_data` needed for continued
Kaggle runs.

## Purpose

N13 moves from crypto/news/technical context to official macro risk-regime
context. The first source is the OFR Financial Stress Index.

OFR FSI is not a direct RORO index. It is used as an official
financial-stress / risk-off proxy:

```text
higher OFR FSI = higher financial stress = risk-off context
```

The BTC direction is not hard-coded. The Stage 4 context-FiLM model must learn
whether high financial stress should suppress or strengthen BTC chart signals.

## Implementation

Added:

```text
scripts/build_stage4_fsi_context_features.py
notebooks/kaggle_stage4_n13_1_ofr_fsi_context_features_one_cell.md
```

The builder:

```text
loads official OFR FSI CSV
aligns FSI to BTC sample Date with as-of policy
uses default lag: t - 1 day
builds trailing 20/60-day FSI features
fits imputation/clipping/z-score on train split only
writes a prebuilt context artifact for the existing Stage 4 runner
```

Default raw features:

```text
ofr_fsi_value
ofr_fsi_mean_20
ofr_fsi_mean_60
ofr_fsi_delta_20
ofr_fsi_delta_60
ofr_fsi_std_60
```

Optional category features:

```text
ofr_credit
ofr_equity_valuation
ofr_funding
ofr_safe_assets
ofr_volatility
```

## Output Artifact

Default context name:

```text
stage4_fsi_context_i60_ohlc_ma_vb_r20_ofr_fsi_lag1_w20_60
```

Expected files:

```text
outputs/stage4/context/<context_name>/seed_42/context_features.csv
outputs/stage4/context/<context_name>/seed_42/context_scaler.json
outputs/stage4/context/<context_name>/seed_42/context_feature_audit.json
outputs/stage4/context/<context_name>/seed_42/context_feature_summary.csv
outputs/stage4/context/<context_name>/seed_42/fsi_context_manifest.json
```

Kaggle artifact was generated and archived locally at:

```text
/Users/jaehyeonjeong/Desktop/논문/N13_1_result
```

Report copies:

```text
reports/tables/stage4_fsi_context_i60_ohlc_ma_vb_r20_ofr_fsi_lag1_w20_60_seed42_fsi_context_feature_audit.json
reports/tables/stage4_fsi_context_i60_ohlc_ma_vb_r20_ofr_fsi_lag1_w20_60_seed42_fsi_context_feature_summary.csv
reports/tables/stage4_fsi_context_i60_ohlc_ma_vb_r20_ofr_fsi_lag1_w20_60_seed42_fsi_context_manifest.json
```

## Verification

Static compile passed:

```text
python3 -m py_compile scripts/build_stage4_fsi_context_features.py
```

Official OFR FSI CSV load was verified locally:

```text
date coverage: 2000-01-03 to 2026-06-03
rows: 6685
columns: Date, OFR FSI, Credit, Equity valuation, Safe assets, Funding,
         Volatility, United States, Other advanced economies, Emerging markets
```

Kaggle artifact build passed:

```text
status: ok
context_dim: 6
split counts: train 671, validation 287, test 1441
BTC sample date range: 2018-04-29 to 2024-12-11
OFR FSI date range: 2000-01-03 to 2026-06-03
as-of policy: latest OFR FSI source date <= BTC image end date - 1 day
mean OFR source age: 1.46 days
max OFR source age: 4.0 days
normalized features finite: true
```

The only warnings are early-window missing rates for 60-day trailing features
in train/validation:

```text
ofr_fsi_mean_60
ofr_fsi_delta_60
ofr_fsi_std_60
```

This is expected because the first Stage 4 BTC samples do not yet have enough
prior FSI observations to compute full 60-day trailing windows. The builder
uses train-only median imputation, train quantile clipping, and train z-score
normalization. The test split has zero raw missing rate for all six FSI
features.

## Next Step

Proceed to `4-N13-2`: FSI-only frozen bounded FiLM five-seed run.
