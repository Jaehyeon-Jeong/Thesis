# 2-I4 BTC Label/Split/Normalization Code

## English

Status: complete

Implemented:
- `src/stage2_btc/data/label_split.py`
- `scripts/check_stage2_label_split.py`

Verified `I20 / ohlc_ma_vb / R20` counts:
- Total eligible samples: `2519`
- Train: `727`, positive rate `54.33%`
- Validation: `311`, positive rate `58.84%`
- Test: `1441`, positive rate `54.13%`

Train-only normalization:
- pixel mean: `0.1197522638697845`
- pixel std: `0.32467161743498`
- pixel scale: `255.0`

## 한국어

상태: 완료

구현:
- `src/stage2_btc/data/label_split.py`
- `scripts/check_stage2_label_split.py`

확인한 `I20 / ohlc_ma_vb / R20` count:
- 전체 eligible sample: `2519`
- Train: `727`, positive rate `54.33%`
- Validation: `311`, positive rate `58.84%`
- Test: `1441`, positive rate `54.13%`

Train-only normalization:
- pixel mean: `0.1197522638697845`
- pixel std: `0.32467161743498`
- pixel scale: `255.0`
