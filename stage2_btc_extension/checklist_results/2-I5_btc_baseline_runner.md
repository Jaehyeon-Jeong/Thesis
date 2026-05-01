# 2-I5 BTC Baseline Runner and Model Variants

## English

Status: complete

Implemented:
- `src/stage2_btc/models/stock_cnn.py`
- `src/stage2_btc/training/loop.py`
- `src/stage2_btc/runners/btc_baseline.py`
- `scripts/run_stage2_btc_baseline.py`

Model checks:
- I5: params `155138`, flatten `15360`
- I20: params `708866`, flatten `46080`
- I60: params `2952962`, flatten `184320`

Smoke training:
- Experiment: `stage2_i20_ohlc_ma_vb_r20`
- Seed: `42`
- Rows: train `128`, validation `64`, test `64`
- Epochs: `2`
- Best epoch: `2`
- Best validation loss: `0.6888229846954346`

Note:
- This smoke run is only a code-path check, not a thesis result.

## 한국어

상태: 완료

구현:
- `src/stage2_btc/models/stock_cnn.py`
- `src/stage2_btc/training/loop.py`
- `src/stage2_btc/runners/btc_baseline.py`
- `scripts/run_stage2_btc_baseline.py`

모델 확인:
- I5: params `155138`, flatten `15360`
- I20: params `708866`, flatten `46080`
- I60: params `2952962`, flatten `184320`

Smoke training:
- Experiment: `stage2_i20_ohlc_ma_vb_r20`
- Seed: `42`
- Rows: train `128`, validation `64`, test `64`
- Epochs: `2`
- Best epoch: `2`
- Best validation loss: `0.6888229846954346`

주의:
- 이 smoke run은 코드 연결 확인용이며 논문 결과값이 아닙니다.
