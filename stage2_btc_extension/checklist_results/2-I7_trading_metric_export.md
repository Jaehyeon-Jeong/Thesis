# 2-I7 BTC Trading Metric Export

## English

Status: complete

Implemented:
- `src/stage2_btc/evaluation/backtest.py`
- `scripts/evaluate_stage2_trading.py`

Smoke output:
- Trading metrics: `outputs/stage2/metrics/stage2_i20_ohlc_ma_vb_r20/seed_42/test_trading_metrics.json`

Important report sentence:
> 원 논문은 cross-sectional stock prediction이므로 H-L decile spread를 구성할 수 있지만, BTC는 단일 자산이므로 동일한 H-L 구조를 직접 적용하기 어렵다. 따라서 BTC 실험에서는 classification metric과 time-series trading strategy를 함께 사용한다.

## 한국어

상태: 완료

구현:
- `src/stage2_btc/evaluation/backtest.py`
- `scripts/evaluate_stage2_trading.py`

Smoke output:
- Trading metrics: `outputs/stage2/metrics/stage2_i20_ohlc_ma_vb_r20/seed_42/test_trading_metrics.json`

보고서에 반드시 넣을 문장:
> 원 논문은 cross-sectional stock prediction이므로 H-L decile spread를 구성할 수 있지만, BTC는 단일 자산이므로 동일한 H-L 구조를 직접 적용하기 어렵다. 따라서 BTC 실험에서는 classification metric과 time-series trading strategy를 함께 사용한다.
