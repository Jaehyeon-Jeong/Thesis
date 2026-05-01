# 2-I8 BTC Grad-CAM Export

## English

Status: complete

Implemented:
- `src/stage2_btc/interpretability/gradcam.py`
- `scripts/generate_stage2_gradcam.py`

Smoke Grad-CAM outputs:
- Output figure: `outputs/stage2/figures/stage2_i20_ohlc_ma_vb_r20/seed_42/gradcam/test/btc_gradcam_test_2perclass.png`
- Selected samples: `outputs/stage2/figures/stage2_i20_ohlc_ma_vb_r20/seed_42/gradcam/test/samples.csv`
- Report copy: `reports/figures/gradcam/stage2_i20_ohlc_ma_vb_r20_seed_42_test_gradcam.png`

Important:
- This is a Grad-CAM heatmap, not a raw feature map.
- The smoke figure uses `2` samples per class.

## 한국어

상태: 완료

구현:
- `src/stage2_btc/interpretability/gradcam.py`
- `scripts/generate_stage2_gradcam.py`

Smoke Grad-CAM output:
- Output figure: `outputs/stage2/figures/stage2_i20_ohlc_ma_vb_r20/seed_42/gradcam/test/btc_gradcam_test_2perclass.png`
- Selected samples: `outputs/stage2/figures/stage2_i20_ohlc_ma_vb_r20/seed_42/gradcam/test/samples.csv`
- Report copy: `reports/figures/gradcam/stage2_i20_ohlc_ma_vb_r20_seed_42_test_gradcam.png`

중요:
- 이 그림은 raw feature map이 아니라 Grad-CAM heatmap입니다.
- smoke figure는 class당 `2`개 sample을 사용했습니다.
