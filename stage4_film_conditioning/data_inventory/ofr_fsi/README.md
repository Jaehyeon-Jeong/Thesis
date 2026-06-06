# OFR Financial Stress Index Input

Official source:

```text
https://www.financialresearch.gov/financial-stress-index/data/fsi.csv
```

This CSV is used in Stage 4 N13 as an official financial-stress / risk-off
proxy, not as a direct RORO index. Higher `OFR FSI` means stronger financial
stress. The BTC direction is not hard-coded; the context-FiLM model learns the
relationship from the training split.

Local snapshot:

```text
fsi.csv
```

Snapshot verification when added:

```text
rows: 6685
date range: 2000-01-03 to 2026-06-03
columns: Date, OFR FSI, Credit, Equity valuation, Safe assets, Funding,
         Volatility, United States, Other advanced economies, Emerging markets
```
