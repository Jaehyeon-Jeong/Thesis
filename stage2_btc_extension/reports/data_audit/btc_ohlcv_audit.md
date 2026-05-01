# BTC OHLCV Data Audit

## English

- Status: `ok`
- Source file: `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV/btc_1d_data_2018_to_2025.csv`
- Raw shape: `[2997, 12]`
- Date range: `2018-01-01T00:00:00` to `2026-03-16T00:00:00`
- Clean rows: `2997`
- Duplicate dates: `0`
- Invalid OHLCV rows: `0`

Canonical column map:

- `Date` <- `Open time`
- `Open` <- `Open`
- `High` <- `High`
- `Low` <- `Low`
- `Close` <- `Close`
- `Volume` <- `Volume`

Frequency:

- Median delta: `1 days`
- Mostly daily: `True`
- Daily delta share: `1.0`

Window/horizon availability:

| Image | Horizon | Samples | Positive rate | First end date | Last end date |
| --- | --- | ---: | ---: | --- | --- |
| I5 | R5 | 2988 | 0.5268 | 2018-01-05T00:00:00 | 2026-03-11T00:00:00 |
| I5 | R20 | 2973 | 0.5298 | 2018-01-05T00:00:00 | 2026-02-24T00:00:00 |
| I5 | R60 | 2933 | 0.5261 | 2018-01-05T00:00:00 | 2026-01-15T00:00:00 |
| I20 | R5 | 2973 | 0.5291 | 2018-01-20T00:00:00 | 2026-03-11T00:00:00 |
| I20 | R20 | 2958 | 0.5325 | 2018-01-20T00:00:00 | 2026-02-24T00:00:00 |
| I20 | R60 | 2918 | 0.5288 | 2018-01-20T00:00:00 | 2026-01-15T00:00:00 |
| I60 | R5 | 2933 | 0.5288 | 2018-03-01T00:00:00 | 2026-03-11T00:00:00 |
| I60 | R20 | 2918 | 0.5343 | 2018-03-01T00:00:00 | 2026-02-24T00:00:00 |
| I60 | R60 | 2878 | 0.5361 | 2018-03-01T00:00:00 | 2026-01-15T00:00:00 |

## 한국어

- 상태: `ok`
- 원본 파일: `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV/btc_1d_data_2018_to_2025.csv`
- raw shape: `[2997, 12]`
- 날짜 범위: `2018-01-01T00:00:00`부터 `2026-03-16T00:00:00`까지
- cleaning 후 row 수: `2997`
- duplicate date 수: `0`
- 비정상 OHLCV row 수: `0`

canonical column mapping:

- `Date` <- `Open time`
- `Open` <- `Open`
- `High` <- `High`
- `Low` <- `Low`
- `Close` <- `Close`
- `Volume` <- `Volume`

frequency 확인:

- median delta: `1 days`
- 대부분 daily인지: `True`
- daily delta 비율: `1.0`

window/horizon별 사용 가능 sample:

| Image | Horizon | Samples | Positive rate | 첫 image 종료일 | 마지막 image 종료일 |
| --- | --- | ---: | ---: | --- | --- |
| I5 | R5 | 2988 | 0.5268 | 2018-01-05T00:00:00 | 2026-03-11T00:00:00 |
| I5 | R20 | 2973 | 0.5298 | 2018-01-05T00:00:00 | 2026-02-24T00:00:00 |
| I5 | R60 | 2933 | 0.5261 | 2018-01-05T00:00:00 | 2026-01-15T00:00:00 |
| I20 | R5 | 2973 | 0.5291 | 2018-01-20T00:00:00 | 2026-03-11T00:00:00 |
| I20 | R20 | 2958 | 0.5325 | 2018-01-20T00:00:00 | 2026-02-24T00:00:00 |
| I20 | R60 | 2918 | 0.5288 | 2018-01-20T00:00:00 | 2026-01-15T00:00:00 |
| I60 | R5 | 2933 | 0.5288 | 2018-03-01T00:00:00 | 2026-03-11T00:00:00 |
| I60 | R20 | 2918 | 0.5343 | 2018-03-01T00:00:00 | 2026-02-24T00:00:00 |
| I60 | R60 | 2878 | 0.5361 | 2018-03-01T00:00:00 | 2026-01-15T00:00:00 |
