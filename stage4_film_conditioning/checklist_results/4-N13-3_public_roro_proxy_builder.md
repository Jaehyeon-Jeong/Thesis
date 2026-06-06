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

## Formula Mapping

KC Fed releases category-level daily scores rather than all raw proprietary
inputs. Conceptually, its RORO index can be written as:

```text
KC_RORO_t = PC1(
    credit risk / spreads,
    equity market and volatility,
    funding liquidity,
    currencies and gold
)
```

The official cached file exposes these already-processed categories:

```text
z_spreads
z_equities
z_liquidity
z_goldcurrency
z_roro
```

Because `z_roro` starts in June 2023 in the available KC Fed file, Stage 4 uses
a public-data proxy that preserves the same idea: align public market
components so larger values mean risk-off pressure, then fit the first PCA/SVD
component on the train period only.

Current implemented proxy:

```text
x1_t = VIX_t - VIX_{t-20}
x2_t = -log(SP500_t / SP500_{t-20})
x3_t = -(DGS10_t - DGS10_{t-20})

Our_RORO_proxy_t = PC1_train_only(
    z_train(x1_t),
    z_train(x2_t),
    z_train(x3_t)
)
```

The fitted first-component weights in the current local artifact are:

```text
Our_RORO_proxy_t
= 0.617254 * z_train(VIX_t - VIX_{t-20})
+ 0.639203 * z_train(-log(SP500_t / SP500_{t-20}))
+ 0.458713 * z_train(-(DGS10_t - DGS10_{t-20}))
```

Higher `Our_RORO_proxy_t` means stronger risk-off pressure. PCA weights,
normalization, clipping, and missing-value imputation are fitted only on the
train split, then reused unchanged for validation/test.

Candidate extended proxy if clean long-history DXY and high-yield index price
CSV files are added:

```text
x4_t = -log(HY_Index_t / HY_Index_{t-20})
x5_t = log(DXY_t / DXY_{t-20})

Our_RORO_proxy_extended_t = PC1_train_only(
    z_train(x1_t),
    z_train(x2_t),
    z_train(x3_t),
    z_train(x4_t),
    z_train(x5_t)
)
```

Important naming rule: the Investing.com ICE BofA U.S. High Yield historical
page is a high-yield bond index price series, not HY OAS. If used, it must be
described as a `HY bond index price based credit-risk proxy`, where negative
high-yield index return means stronger credit-risk pressure.

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
