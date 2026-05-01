# 2-I2 BTC OHLCV Loader

## English

Status: complete

Implemented:
- `src/stage2_btc/data/ohlcv.py`
- `scripts/check_stage2_data_loading.py`

Verified result:
- Source file: `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV/btc_1d_data_2018_to_2025.csv`
- Rows loaded: `2997`
- Date range: `2018-01-01` to `2026-03-16`
- Canonical columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `source_row`

## 한국어

상태: 완료

구현:
- `src/stage2_btc/data/ohlcv.py`
- `scripts/check_stage2_data_loading.py`

확인 결과:
- Source file: `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV/btc_1d_data_2018_to_2025.csv`
- 로드 row 수: `2997`
- 날짜 범위: `2018-01-01`부터 `2026-03-16`
- 표준 column: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `source_row`
