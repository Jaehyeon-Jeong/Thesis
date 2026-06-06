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
```

The experimental builder creates a KC Fed-inspired public-data RORO proxy from
longer-history public macro series when the FRED files are available:

```text
VIXCLS
BAMLH0A0HYM2
SP500
NASDAQCOM
DTWEXBGS
DGS10
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

## Current Execution Note

Local syntax validation passed:

```text
python3 -m py_compile scripts/build_stage4_roro_context_features.py
```

Local full feature generation was blocked by temporary FRED `504 Gateway
Time-out` responses for several series. The Kaggle one-cell is ready to retry
with internet enabled or to run from cached `data_inventory/roro_public/raw`
CSV files when those files are attached in the Stage 4 snapshot.

## Next Step

Proceed to `4-N13-4` only after the N13-3 context artifact is successfully
created:

```text
stage4_roro_context_i60_ohlc_ma_vb_r20_public_roro_pca_lag1_w20_60
```

N13-4 should use the same frozen Stage 2 bounded-FiLM protocol as N13-2, first
testing proxy-only features before raw component features.
