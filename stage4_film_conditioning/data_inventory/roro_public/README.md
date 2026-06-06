# Public RORO / Macro Source Inventory

Stage 4 N13 uses macro risk-regime context to test whether image-external
market stress information can improve the frozen Stage 2 chart baseline.

## KC Fed Official RORO

The Kansas City Fed publishes an official Risk-On Risk-Off index. The cached
files are:

- `kc_fed_official/roro_daily.csv`
- `kc_fed_official/roro_weekly.csv`

Source page:
https://www.kansascityfed.org/data-and-trends/risk-on-risk-off-index/

The KC Fed page states that the RORO index uses daily asset-market data from
the United States and euro area, aggregates credit risk, equity volatility,
funding conditions, currencies, and gold, and uses the first principal
component of daily changes.

Important Stage 4 limitation:
the downloaded KC Fed daily and weekly files currently start in June 2023.
That does not cover the Stage 4 train/validation period. Therefore they are
kept as source evidence and audit material, but not used directly as the first
N13-4 training context.

## Public Proxy Inputs

The experimental N13-3 builder uses longer-history public macro series from
FRED when available:

- `VIXCLS`: VIX; higher 20-observation change is risk-off.
- `BAMLH0A0HYM2`: high-yield OAS; higher 20-observation change is risk-off.
- `SP500`: S&P 500; lower 20-observation return is risk-off.
- `NASDAQCOM`: NASDAQ Composite; lower 20-observation return is risk-off.
- `DTWEXBGS`: broad U.S. dollar index; higher 20-observation return is treated
  as risk-off pressure.
- `DGS10`: 10-year Treasury yield; falling 20-observation change is treated as
  risk-off with caution.

The proxy is fitted without BTC labels:

```text
risk-off aligned public components
-> train-only median/clip/z-score
-> train-only PCA/SVD first component
-> sign fixed so larger value means stronger risk-off pressure
```

This is a KC Fed-inspired public-data RORO proxy, not a full replication of the
KC Fed index.
