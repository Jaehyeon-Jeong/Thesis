# 2-I3 BTC Image Generator

## English

Status: complete

Implemented:
- `src/stage2_btc/imaging/ohlcv_image.py`
- `scripts/check_stage2_image_generation.py`

Generated local sample images:
- `reports/figures/sample_images/btc_i20_ohlc_sample.png`
- `reports/figures/sample_images/btc_i20_ohlc_vb_sample.png`
- `reports/figures/sample_images/btc_i20_ohlc_ma_sample.png`
- `reports/figures/sample_images/btc_i20_ohlc_ma_vb_sample.png`

Correction after visual review:
- `ohlc` and `ohlc_ma` now use the full image height for price scaling when
  volume bars are absent.
- `ohlc_vb` and `ohlc_ma_vb` keep the paper-style split between upper price
  area and lower volume area.
- Verified I20 no-volume sample nonzero rows: `0` to `63`.

Absolute folder:
- `/Users/jaehyeonjeong/Desktop/논문/stage2_btc_extension/reports/figures/sample_images/`

Sample:
- `I20/R20`
- sample date: `2018-02-08`

## 한국어

상태: 완료

구현:
- `src/stage2_btc/imaging/ohlcv_image.py`
- `scripts/check_stage2_image_generation.py`

생성된 로컬 sample image:
- `reports/figures/sample_images/btc_i20_ohlc_sample.png`
- `reports/figures/sample_images/btc_i20_ohlc_vb_sample.png`
- `reports/figures/sample_images/btc_i20_ohlc_ma_sample.png`
- `reports/figures/sample_images/btc_i20_ohlc_ma_vb_sample.png`

시각 확인 후 수정:
- `ohlc`와 `ohlc_ma`는 volume bar가 없을 때 price scaling이 전체 image height를
  사용하도록 수정했습니다.
- `ohlc_vb`와 `ohlc_ma_vb`는 기존처럼 상단 price area와 하단 volume area를 나눕니다.
- I20 no-volume sample의 nonzero row 범위가 `0`부터 `63`까지인 것을 확인했습니다.

절대 경로 폴더:
- `/Users/jaehyeonjeong/Desktop/논문/stage2_btc_extension/reports/figures/sample_images/`

Sample:
- `I20/R20`
- sample date: `2018-02-08`
