# 2-I9 Local Smoke Test

## English

Status: complete

Smoke command sequence:
1. `scripts/run_stage2_btc_baseline.py`
2. `scripts/evaluate_stage2_predictions.py`
3. `scripts/evaluate_stage2_trading.py`
4. `scripts/generate_stage2_gradcam.py`
5. `scripts/check_stage2_outputs.py`

Smoke configuration:
- `I20 / ohlc_ma_vb / R20`
- seed `42`
- max epochs `2`
- train `128`, validation `64`, test `64`

Output check:
- checkpoint: exists
- train history: exists
- train metadata: exists
- prediction CSV: exists
- classification metrics: exists
- trading metrics: exists
- Grad-CAM figure: exists

## 한국어

상태: 완료

Smoke command 순서:
1. `scripts/run_stage2_btc_baseline.py`
2. `scripts/evaluate_stage2_predictions.py`
3. `scripts/evaluate_stage2_trading.py`
4. `scripts/generate_stage2_gradcam.py`
5. `scripts/check_stage2_outputs.py`

Smoke 설정:
- `I20 / ohlc_ma_vb / R20`
- seed `42`
- max epochs `2`
- train `128`, validation `64`, test `64`

Output check:
- checkpoint: 존재
- train history: 존재
- train metadata: 존재
- prediction CSV: 존재
- classification metrics: 존재
- trading metrics: 존재
- Grad-CAM figure: 존재
