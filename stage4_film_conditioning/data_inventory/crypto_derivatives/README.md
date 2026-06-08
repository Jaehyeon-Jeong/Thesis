# Crypto Derivatives Context Data

Purpose: candidate external context for Stage 4 FiLM experiments after F&G showed a small positive effect mainly on volume-aware images.

## Why derivatives data

The current visual baseline already contains OHLC, moving-average, and volume-bar information. A useful Stage 4 context should therefore add information that is not directly visible in the chart image. BTC perpetual futures data is a good next candidate because it describes leverage positioning and market pressure rather than only spot OHLCV.

## Downloaded local files

### CFTC CME Bitcoin futures open interest and trader positioning

- Source: CFTC Financial Futures Commitments of Traders historical files
- Raw file: `cftc_cme_bitcoin_cot/raw/cftc_financial_futures_bitcoin_rows_2018_2024.csv`
- CME main weekly file: `cftc_cme_bitcoin_cot/processed/cftc_cme_bitcoin_main_cot_weekly_2018_2024.csv`
- CME main daily release-lagged file: `cftc_cme_bitcoin_cot/processed/cftc_cme_bitcoin_main_cot_daily_release_lag3_ffill_2018_2024.csv`
- CME main + micro aggregate daily release-lagged file: `cftc_cme_bitcoin_cot/processed/cftc_cme_bitcoin_main_plus_micro_cot_daily_release_lag3_ffill_2018_2024.csv`
- Coverage downloaded: 2018-01-02 to 2024-12-31 across Bitcoin futures reports
- Stage 4 date coverage for CME main: no missing dates for 2018-04-29 to 2024-12-11 after release-lagged forward-fill.

Candidate features:
- `Open_Interest_All` level/change/z-score over 7, 20, and 60 days
- `Lev_Money_Net_Ratio_All`, `Asset_Mgr_Net_Ratio_All`, `Dealer_Net_Ratio_All`
- open-interest change combined with BitMEX funding, for example `funding_mean_20 * oi_change_20`

Interpretation:
- CFTC/CME open interest is regulated futures positioning, not offshore perpetual OI.
- It gives reliable institutional derivatives context from an official U.S. source.
- Weekly COT values use `available_date = report_date + 3 calendar days`, are
  forward-filled to daily rows, and include `cot_source_report_date`,
  `cot_available_date`, `cot_age_days`, and `cot_release_lag_days` for audit.

### BitMEX XBTUSD funding rate and derivatives activity

- Source: BitMEX REST API `GET /api/v1/funding` and `GET /api/v1/trade/bucketed`
- Raw funding file: `bitmex_xbtusd/raw/bitmex_xbtusd_funding_8h_2018_2024.csv`
- Daily funding file: `bitmex_xbtusd/processed/bitmex_xbtusd_funding_daily_2018_2024.csv`
- Raw trade bucket file: `bitmex_xbtusd/raw/bitmex_xbtusd_trade_bucketed_1d_2018_2024.csv`
- Daily derivatives activity file: `bitmex_xbtusd/processed/bitmex_xbtusd_derivatives_activity_daily_2018_2024.csv`
- Coverage downloaded: 2018-01-01 to 2024-12-31
- Stage 4 date coverage: no missing dates for 2018-04-29 to 2024-12-11.

Candidate features:
- `funding_rate_mean`, `funding_rate_sum`, `funding_rate_abs_mean`, `funding_rate_min`, `funding_rate_max` over 7, 20, and 60 days
- derivatives activity: `trades`, `volume`, `turnover`, `homeNotional`, `foreignNotional` over 7, 20, and 60 days

Interpretation:
- BitMEX XBTUSD is a long-running BTC perpetual swap market, so funding rate is a clean leverage/crowding context available from 2018.
- Positive funding means longs pay shorts; negative funding means shorts pay longs.
- Extreme absolute funding can indicate leverage imbalance or crowded positioning.
- BitMEX derivatives volume/activity is not the same as spot OHLCV volume and can act as a market-participation context.

### Binance BTCUSDT funding rate

- Source: Binance USD-M Futures REST `GET /fapi/v1/fundingRate`
- Raw file: `binance_btcusdt_funding_rate/raw/binance_btcusdt_funding_rate_2019_2024.csv`
- Daily file: `binance_btcusdt_funding_rate/processed/binance_btcusdt_funding_daily_2019_2024.csv`
- Coverage downloaded: 2019-10-01 to 2024-12-31
- Candidate features: trailing mean/sum/std/absolute mean over 7, 20, and 60 days.

Interpretation:
- Positive/high funding means long perpetual positions are paying shorts, often a crowded-long or bullish-leverage regime.
- Negative funding means shorts are paying longs, often a bearish/crowded-short regime.
- Extreme funding may be more useful than the raw level because it can indicate leverage imbalance.

### Binance BTCUSDT premium index

- Source: Binance USD-M Futures REST `GET /fapi/v1/premiumIndexKlines`
- Raw file: `binance_btcusdt_premium_index/raw/binance_btcusdt_premium_index_1d_2019_2024.csv`
- Daily file: `binance_btcusdt_premium_index/processed/binance_btcusdt_premium_index_daily_2019_2024.csv`
- Coverage downloaded: 2019-12-24 to 2024-12-31
- Candidate features: trailing close/mean/std/range/absolute close over 7, 20, and 60 days.

Interpretation:
- Premium index describes the futures premium/discount relative to the index price.
- It is related to basis pressure and can complement chart volume/price patterns.

## Limitations

- These Binance futures series do not cover the earliest Stage 4 training rows from 2018. If used, the experiment must either audit/impute early missing context or restrict training rows to dates with derivatives coverage.
- Binance REST open-interest statistics and long/short ratio endpoints are not suitable for our full 2018-2024 experiment through REST because the official API only exposes recent history for those endpoints.
- Open interest can still be considered later if a reliable full historical source is added as a local dataset.
