# 2-I6 BTC Prediction and Classification Metric Export

## English

Status: complete

Implemented:
- `src/stage2_btc/evaluation/prediction.py`
- `scripts/evaluate_stage2_predictions.py`

Smoke output:
- Predictions: `outputs/stage2/predictions/stage2_i20_ohlc_ma_vb_r20/seed_42/test_predictions.csv`
- Metrics: `outputs/stage2/metrics/stage2_i20_ohlc_ma_vb_r20/seed_42/test_metrics.json`
- Accuracy on smoke subset: `0.265625`
- Number of smoke predictions: `64`

Note:
- `outputs/` is intentionally not published to GitHub.

## 한국어

상태: 완료

구현:
- `src/stage2_btc/evaluation/prediction.py`
- `scripts/evaluate_stage2_predictions.py`

Smoke output:
- Predictions: `outputs/stage2/predictions/stage2_i20_ohlc_ma_vb_r20/seed_42/test_predictions.csv`
- Metrics: `outputs/stage2/metrics/stage2_i20_ohlc_ma_vb_r20/seed_42/test_metrics.json`
- Smoke subset accuracy: `0.265625`
- Smoke prediction row 수: `64`

주의:
- `outputs/`는 GitHub에 올리지 않습니다.
