# Stage 2 Scripts

## English

Scripts:
- `audit_btc_ohlcv.py`: audits the BTC OHLCV CSV before image generation.
- `check_stage2_data_loading.py`: verifies the shared BTC OHLCV loader.
- `check_stage2_image_generation.py`: generates local sample images under
  `reports/figures/sample_images/`.
- `check_stage2_label_split.py`: verifies label/split/normalization.
- `run_stage2_btc_baseline.py`: trains one BTC baseline experiment tuple.
- `evaluate_stage2_predictions.py`: exports prediction CSV and classification metrics.
- `evaluate_stage2_trading.py`: exports BTC time-series trading metrics.
- `generate_stage2_gradcam.py`: exports BTC Grad-CAM figures.
- `check_stage2_outputs.py`: checks required output files after a run.

## 한국어

현재 script:
- `audit_btc_ohlcv.py`: image generation 전에 BTC OHLCV CSV를 audit합니다.
- `check_stage2_data_loading.py`: 공통 BTC OHLCV loader를 확인합니다.
- `check_stage2_image_generation.py`: `reports/figures/sample_images/` 아래에
  local sample image를 생성합니다.
- `check_stage2_label_split.py`: label/split/normalization을 확인합니다.
- `run_stage2_btc_baseline.py`: BTC baseline experiment tuple 하나를 학습합니다.
- `evaluate_stage2_predictions.py`: prediction CSV와 classification metric을 저장합니다.
- `evaluate_stage2_trading.py`: BTC time-series trading metric을 저장합니다.
- `generate_stage2_gradcam.py`: BTC Grad-CAM figure를 저장합니다.
- `check_stage2_outputs.py`: 실행 후 필요한 output file이 있는지 확인합니다.
