# 0-2 Label File Check

## English

Result:
- Confirmed label columns include `Date`, `StockID`, `MarketCap`, `Ret_5d`,
  `Ret_20d`, `Ret_60d`, `Ret_month`, and `EWMA_vol`.
- Interpreted `Ret_5d`, `Ret_20d`, and `Ret_60d` as individual-stock future
  holding-period returns.

Outputs:
- [monthly20_labels.csv](../data_inventory/monthly20_labels.csv)
- [monthly20_labels_by_year.csv](../data_inventory/monthly20_labels_by_year.csv)
- [monthly20_data_check.md](../docs/monthly20_data_check.md)

## 한국어

결과:
- label column에 `Date`, `StockID`, `MarketCap`, `Ret_5d`, `Ret_20d`,
  `Ret_60d`, `Ret_month`, `EWMA_vol`이 있음을 확인했습니다.
- `Ret_5d`, `Ret_20d`, `Ret_60d`는 개별 주식의 미래 보유수익률로 해석합니다.

산출물:
- [monthly20_labels.csv](../data_inventory/monthly20_labels.csv)
- [monthly20_labels_by_year.csv](../data_inventory/monthly20_labels_by_year.csv)
- [monthly20_data_check.md](../docs/monthly20_data_check.md)
