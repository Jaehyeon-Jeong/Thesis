# 2-2 BTC OHLCV Data Audit

## English

Status: complete.

Primary source:
- Kaggle dataset: `https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024`
- Local audited folder: `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV`
- Audited file: `btc_1d_data_2018_to_2025.csv`

Local files found:
- `btc_15m_data_2018_to_2025.csv`
- `btc_1d_data_2018_to_2025.csv`
- `btc_1h_data_2018_to_2025.csv`
- `btc_4h_data_2018_to_2025.csv`

Audit script:
- `stage2_btc_extension/scripts/audit_btc_ohlcv.py`

Audit outputs:
- `stage2_btc_extension/reports/data_audit/btc_ohlcv_audit.json`
- `stage2_btc_extension/reports/data_audit/btc_ohlcv_audit.md`
- `stage2_btc_extension/reports/data_audit/btc_ohlcv_head.csv`

Data quality result:
- Raw shape: `2997 x 12`
- Date range: `2018-01-01` to `2026-03-16`
- Frequency: daily, median delta `1 days`
- Daily delta share: `1.0`
- Missing calendar days: `0`
- Duplicate dates: `0`
- Missing OHLCV values: `0`
- Invalid OHLCV rows: `0`
- Zero-volume rows: `0`

Canonical Stage 2 mapping:
- `Date` <- `Open time`
- `Open` <- `Open`
- `High` <- `High`
- `Low` <- `Low`
- `Close` <- `Close`
- `Volume` <- `Volume`

Window/horizon availability:

| Image | Horizon | Samples | Positive rate | First image end date | Last image end date |
| --- | --- | ---: | ---: | --- | --- |
| I5 | R5 | 2988 | 0.5268 | 2018-01-05 | 2026-03-11 |
| I5 | R20 | 2973 | 0.5298 | 2018-01-05 | 2026-02-24 |
| I5 | R60 | 2933 | 0.5261 | 2018-01-05 | 2026-01-15 |
| I20 | R5 | 2973 | 0.5291 | 2018-01-20 | 2026-03-11 |
| I20 | R20 | 2958 | 0.5325 | 2018-01-20 | 2026-02-24 |
| I20 | R60 | 2918 | 0.5288 | 2018-01-20 | 2026-01-15 |
| I60 | R5 | 2933 | 0.5288 | 2018-03-01 | 2026-03-11 |
| I60 | R20 | 2918 | 0.5343 | 2018-03-01 | 2026-02-24 |
| I60 | R60 | 2878 | 0.5361 | 2018-03-01 | 2026-01-15 |

Decision:
- Use `btc_1d_data_2018_to_2025.csv` for Stage 2 baseline.
- No daily resampling is needed.
- Use `Volume` for the Re-image-style volume bars.
- Use `Open time` as the canonical daily `Date`.
- BTC `I5`, `I20`, and `I60` are all feasible.
- Stage 2 must implement three window-specific CNN variants:
  `StockCNNI5`, `StockCNNI20`, and `StockCNNI60`.

Next checklist:
- 2-3 BTC image-generation detail plan.

## 한국어

상태: 완료.

Primary source:
- Kaggle dataset: `https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024`
- local audited folder: `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV`
- audited file: `btc_1d_data_2018_to_2025.csv`

로컬에서 확인한 파일:
- `btc_15m_data_2018_to_2025.csv`
- `btc_1d_data_2018_to_2025.csv`
- `btc_1h_data_2018_to_2025.csv`
- `btc_4h_data_2018_to_2025.csv`

Audit script:
- `stage2_btc_extension/scripts/audit_btc_ohlcv.py`

Audit outputs:
- `stage2_btc_extension/reports/data_audit/btc_ohlcv_audit.json`
- `stage2_btc_extension/reports/data_audit/btc_ohlcv_audit.md`
- `stage2_btc_extension/reports/data_audit/btc_ohlcv_head.csv`

데이터 품질 결과:
- raw shape: `2997 x 12`
- date range: `2018-01-01`부터 `2026-03-16`까지
- frequency: daily, median delta `1 days`
- daily delta share: `1.0`
- missing calendar days: `0`
- duplicate dates: `0`
- missing OHLCV values: `0`
- invalid OHLCV rows: `0`
- zero-volume rows: `0`

Stage 2 canonical mapping:
- `Date` <- `Open time`
- `Open` <- `Open`
- `High` <- `High`
- `Low` <- `Low`
- `Close` <- `Close`
- `Volume` <- `Volume`

Window/horizon별 사용 가능 sample:

| Image | Horizon | Samples | Positive rate | 첫 image 종료일 | 마지막 image 종료일 |
| --- | --- | ---: | ---: | --- | --- |
| I5 | R5 | 2988 | 0.5268 | 2018-01-05 | 2026-03-11 |
| I5 | R20 | 2973 | 0.5298 | 2018-01-05 | 2026-02-24 |
| I5 | R60 | 2933 | 0.5261 | 2018-01-05 | 2026-01-15 |
| I20 | R5 | 2973 | 0.5291 | 2018-01-20 | 2026-03-11 |
| I20 | R20 | 2958 | 0.5325 | 2018-01-20 | 2026-02-24 |
| I20 | R60 | 2918 | 0.5288 | 2018-01-20 | 2026-01-15 |
| I60 | R5 | 2933 | 0.5288 | 2018-03-01 | 2026-03-11 |
| I60 | R20 | 2918 | 0.5343 | 2018-03-01 | 2026-02-24 |
| I60 | R60 | 2878 | 0.5361 | 2018-03-01 | 2026-01-15 |

결정:
- Stage 2 baseline에는 `btc_1d_data_2018_to_2025.csv`를 사용합니다.
- daily resampling은 필요 없습니다.
- Re-image-style volume bar에는 `Volume` column을 사용합니다.
- canonical daily `Date`는 `Open time`을 사용합니다.
- BTC `I5`, `I20`, `I60` 모두 가능합니다.
- Stage 2는 세 window-specific CNN variant를 구현해야 합니다:
  `StockCNNI5`, `StockCNNI20`, `StockCNNI60`.

다음 체크리스트:
- 2-3 BTC image-generation 세부계획.
