# Crypto On-Chain Context Data

Purpose: candidate external context for Stage 4 FiLM experiments. This data is not directly encoded in OHLC/VB chart images, so it can test whether BTC network activity helps the frozen Stage 2 visual baseline.

## Downloaded local files

### Coin Metrics BTC community asset metrics

- Source: Coin Metrics Community API `/v4/timeseries/asset-metrics`
- Raw file: `coinmetrics_btc_community/raw/coinmetrics_btc_community_asset_metrics_2018_2024.json`
- Daily file: `coinmetrics_btc_community/processed/coinmetrics_btc_community_onchain_daily_2018_2024.csv`
- Coverage downloaded: 2018-01-01 to 2024-12-31
- Downloaded metrics:
  - `AdrActCnt`: active address count
  - `TxCnt`: transaction count
  - `HashRate`: network hash rate

Candidate features:
- trailing mean/std/delta over 7, 20, and 60 days
- optional log transform before train-only normalization

Interpretation:
- Active addresses and transaction count approximate network usage/activity.
- Hash rate approximates miner/security regime and may react differently from price-only chart features.

Limitations:
- Some richer community metrics, such as adjusted transfer value USD and fee total USD, were not available through the unauthenticated request used here.
- These are network-level metrics, not direct sentiment or leverage data, so the expected effect may be slower and more regime-like than funding or F&G.
