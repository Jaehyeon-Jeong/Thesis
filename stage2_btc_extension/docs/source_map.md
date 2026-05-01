# Stage 2 Source Map

## English

This file records sources that Stage 2 must check before implementation.

| Area | Source | Stage 2 use |
| --- | --- | --- |
| Stage 1 CNN core | `stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py` | Reuse the confirmed Stock_CNN-style baseline core where applicable. |
| Stage 1 data/label/eval patterns | `stage1_reimage_reproduction/src/stage1_reimage/` | Reuse code style, output schema, and leakage guards. |
| Re-image paper summary | `자료조사/Re-image 요약.md` | Image construction, label rule, CNN/training settings, Grad-CAM reporting style. |
| Re-image paper PDF | `자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | Page-level source check before implementing exact image/CNN choices. |
| Grad-CAM summary | `자료조사/Grad-CAM요약.md` | BTC Grad-CAM implementation and interpretation. |
| Grad-CAM PDF | `자료조사/Grad-CAM.pdf` | Original Grad-CAM method check before implementation. |
| BTC OHLCV data | `kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024` | Primary Stage 2 dataset. Must audit columns/date/frequency before coding. |

Initial decisions:
- Stage 2 starts with BTC OHLCV only.
- Batch size default remains paper value `128`.
- Stage 2 does not use stock cross-sectional H-L decile portfolios.
- Stage 2 must generate BTC Grad-CAM for baseline runs.

Open checks:
- Exact BTC CSV filename and columns inside Kaggle input.
- Whether the dataset is already daily or needs resampling.
- Date range after cleaning.
- Trading-cost assumption for adjusted returns.
- Train/validation/test date ranges.

## 한국어

이 파일은 Stage 2 구현 전에 확인해야 할 근거를 기록합니다.

| 영역 | source | Stage 2에서 쓰는 방식 |
| --- | --- | --- |
| Stage 1 CNN core | `stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py` | 확인된 Stock_CNN식 baseline core를 가능한 한 재사용합니다. |
| Stage 1 data/label/eval pattern | `stage1_reimage_reproduction/src/stage1_reimage/` | 코드 스타일, output schema, leakage guard를 재사용합니다. |
| Re-image 논문 요약 | `자료조사/Re-image 요약.md` | image construction, label rule, CNN/training setting, Grad-CAM reporting style 확인. |
| Re-image 논문 PDF | `자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | 정확한 image/CNN 선택 구현 전 page-level source 확인. |
| Grad-CAM 요약 | `자료조사/Grad-CAM요약.md` | BTC Grad-CAM 구현과 해석에 사용. |
| Grad-CAM PDF | `자료조사/Grad-CAM.pdf` | Grad-CAM 원전 방법 구현 전 확인. |
| BTC OHLCV 데이터 | `kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024` | Stage 2 primary dataset. 코딩 전에 columns/date/frequency audit 필수. |

초기 결정:
- Stage 2는 BTC OHLCV만으로 시작합니다.
- 기본 batch size는 논문값 `128`을 유지합니다.
- Stage 2에서는 stock cross-sectional H-L decile portfolio를 사용하지 않습니다.
- Stage 2 baseline run에서도 BTC Grad-CAM을 반드시 생성합니다.

열린 확인 항목:
- Kaggle input 안의 정확한 BTC CSV filename과 column.
- dataset이 이미 daily인지, resampling이 필요한지.
- cleaning 이후 date range.
- transaction-cost-adjusted return에 쓸 trading cost 가정.
- train/validation/test date range.
