# 4-N13-3 Public RORO Proxy Builder

## Goal

Build a macro risk-regime context source that is closer to the professor's
market-regime motivation than chart-derived technical indicators.

## Source Check

KC Fed publishes an official Risk-On Risk-Off index:

https://www.kansascityfed.org/data-and-trends/risk-on-risk-off-index/

The page states that the index uses daily data from U.S. and euro-area asset
markets and aggregates credit risk, equity volatility, funding conditions,
currencies, and gold through the first principal component of daily changes.

Local source audit:

```text
KC Fed daily file:  2023-06-06 to 2026-06-03, 777 observations
KC Fed weekly file: 2023-06-07 to 2026-06-03, 156 observations
```

This is official and useful for terminology, but it does not cover Stage 4
train/validation dates. Therefore it is not used directly for the first N13-4
training run.

## Implemented Builder

Added:

```text
scripts/build_stage4_roro_context_features.py
notebooks/kaggle_stage4_n13_3_public_roro_context_features_one_cell.md
data_inventory/roro_public/README.md
data_inventory/roro_public/kc_fed_official/roro_daily.csv
data_inventory/roro_public/kc_fed_official/roro_weekly.csv
data_inventory/roro_public/kc_fed_official/RORO_Index_README.pdf
data_inventory/roro_public/raw/VIXCLS.csv
data_inventory/roro_public/raw/SP500.csv
data_inventory/roro_public/raw/DGS10.csv
data_inventory/roro_public/raw/BAMLH0A0HYM2.csv
```

The experimental builder creates a KC Fed-inspired public-data RORO proxy from
longer-history public macro series when local cached files are available:

```text
VIXCLS from Cboe VIX history, converted to FRED-like CSV
SP500 from FRED CSV
DGS10 from U.S. Treasury daily yield XML, converted to FRED-like CSV
BAMLH0A0HYM2 from FRED CSV, cached but excluded from PCA because train coverage is missing
```

Construction rule:

```text
risk-off aligned public components
-> train-only median/clip/z-score
-> train-only PCA/SVD first component
-> sign fixed so larger proxy means stronger risk-off pressure
-> trailing 20/60 proxy features
```

No BTC labels are used in the proxy construction.

## Current Execution Result

Local full feature generation passed:

```text
context_name = stage4_roro_context_i60_ohlc_ma_vb_r20_public_roro_pca_lag1_w20_60
context_dim = 9
split counts = train 671 / validation 287 / test 1441
missing warnings = none
PCA explained variance ratio = 0.720108
```

PCA components used:

```text
riskoff_vix_change_20                 weight 0.617254
riskoff_neg_sp500_return_20           weight 0.639203
riskoff_neg_10y_yield_change_20       weight 0.458713
```

All context features have `0.0` raw missing rate in train, validation, and test
after source-level daily forward-fill and strict `t-1` alignment.

Output reports:

```text
reports/tables/stage4_roro_context_i60_ohlc_ma_vb_r20_public_roro_pca_lag1_w20_60_seed42_roro_context_feature_audit.json
reports/tables/stage4_roro_context_i60_ohlc_ma_vb_r20_public_roro_pca_lag1_w20_60_seed42_roro_context_feature_summary.csv
reports/tables/stage4_roro_context_i60_ohlc_ma_vb_r20_public_roro_pca_lag1_w20_60_seed42_roro_context_manifest.json
```

## Next Step

Proceed to `4-N13-4` with the generated N13-3 context artifact:

```text
stage4_roro_context_i60_ohlc_ma_vb_r20_public_roro_pca_lag1_w20_60
```

N13-4 should use the same frozen Stage 2 bounded-FiLM protocol as N13-2, first
testing proxy-only features before raw component features.
