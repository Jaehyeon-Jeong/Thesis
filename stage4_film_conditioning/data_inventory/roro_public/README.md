# Public RORO Proxy Source Cache

These files are cached for Stage 4 N13-3 so Kaggle runs do not depend on
live public endpoints. The training context is a KC Fed-inspired public-data
RORO proxy, not a full replication of the Kansas City Fed RORO index.

## Official KC Fed RORO Reference

The official KC Fed daily/weekly files are cached for terminology and source
audit only:

- `kc_fed_official/roro_daily.csv`
- `kc_fed_official/roro_weekly.csv`
- `kc_fed_official/RORO_Index_README.pdf`

The cached KC Fed files start in June 2023, so they do not cover the Stage 4
train/validation period. They are therefore not used directly in N13-4.

## Local Public Proxy Inputs

| Local file | Series | Cached coverage | Use in current PCA proxy | Interpretation |
| --- | --- | --- | --- | --- |
| `raw/VIXCLS.csv` | VIX close, converted from Cboe VIX history | 1990-01-02 to 2026-06-05 | yes | VIX up over 20 observations is risk-off pressure. |
| `raw/SP500.csv` | S&P 500 via FRED CSV | 2016-06-06 to 2026-06-04 | yes | Negative S&P 500 return is risk-off pressure. |
| `raw/DGS10.csv` | 10-year Treasury yield, parsed from U.S. Treasury daily yield XML | 2016-01-04 to 2026-06-05 | yes | Falling 10-year yield is treated as risk-off pressure with caution. |
| `raw/BAMLH0A0HYM2.csv` | High-yield OAS via FRED CSV | 2023-06-06 to 2026-06-04 | no, no train-period coverage | High-yield spread widening is risk-off pressure. |

`NASDAQCOM` and `DTWEXBGS` are optional inputs in the builder, but their direct
CSV downloads were unavailable during the local cache build. The script skips
missing optional sources and fits PCA only on components with train-period
coverage.

Current N13-3 artifact:

```text
stage4_roro_context_i60_ohlc_ma_vb_r20_public_roro_pca_lag1_w20_60
context_dim = 9
PCA components = VIX change 20, negative S&P 500 return 20, negative 10Y yield change 20
PCA explained variance ratio = 0.720108
missing warnings = none
```

Current implemented formula:

```text
Our_RORO_proxy_t
= PC1_train_only(
    z_train(VIX_t - VIX_{t-20}),
    z_train(-log(SP500_t / SP500_{t-20})),
    z_train(-(DGS10_t - DGS10_{t-20}))
)
```

Candidate extension if clean long-history manual CSV files are added:

```text
credit risk proxy: -log(HY_Index_t / HY_Index_{t-20})
currency proxy:     log(DXY_t / DXY_{t-20})
```

The high-yield index price proxy must not be described as HY OAS. It is a
high-yield bond index price based credit-risk proxy.
