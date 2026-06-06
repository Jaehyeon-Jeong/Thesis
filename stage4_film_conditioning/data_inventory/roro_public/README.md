# Public RORO Proxy Source Cache

These files are cached for Stage 4 N13-3 so Kaggle runs do not depend on
live public endpoints. The training context is a KC Fed-inspired public-data
RORO proxy, not a full replication of the Kansas City Fed RORO index.

The official KC Fed daily/weekly files are kept in `kc_fed_official/` for
terminology and source audit. They currently start in June 2023, so the
training proxy uses longer-history public source caches from `raw/`.

| Local file | Series | Cached | Source URL | Interpretation |
| --- | --- | --- | --- | --- |
| `raw/VIXCLS.csv` | `VIXCLS` | yes | https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv | VIX up is risk-off pressure. |
| `raw/SP500.csv` | `SP500` | yes | https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500 | Negative S&P 500 return is risk-off pressure. |
| `raw/NASDAQCOM.csv` | `NASDAQCOM` | optional/missing | https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQCOM | Negative NASDAQ return is risk-off pressure. |
| `raw/DTWEXBGS.csv` | `DTWEXBGS` | optional/missing | https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS | Dollar strength is treated as risk-off pressure. |
| `raw/DGS10.csv` | `DGS10` | yes | https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve | Falling 10-year yield is treated as risk-off pressure with caution. |
| `raw/BAMLH0A0HYM2.csv` | `BAMLH0A0HYM2` | yes | https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2 | High-yield spread widening is risk-off pressure. |
| `raw/DXY_yahoo_DX-Y.NYB.csv` | `DXY` | yes | https://finance.yahoo.com/quote/DX-Y.NYB/history/ | Dollar strength is treated as risk-off pressure. |
