# Stage 4 Source Map

## English

| Source | Local/reference path | Stage 4 use |
| --- | --- | --- |
| FiLM paper summary | `자료조사/FiLM요약.md` | FiLM equation, gamma/beta interpretation, BN-FiLM placement |
| FiLM paper PDF | `자료조사/FiLM.pdf` | Exact paper citation check before final code comments |
| FiLM reference repository | `https://github.com/ethanjperez/film`, commit `fe43ddf8a22b339dcca2efa07091ce9d498955cf` | Reference implementation for FiLM generator/layer conventions |
| Stage 2 BTC model | `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py` | Base CNN block variants for I5/I20/I60 |
| Stage 2 BTC pipeline | `stage2_btc_extension/src/stage2_btc/` | Fixed BTC data, labels, split, normalization, evaluation |
| Stage 3 Linear model | `stage3_linear_adapter/src/stage3_linear/models/linear_stock_cnn.py` | Comparison model only |
| Grad-CAM local summary | `자료조사/Grad-CAM요약.md` | Stage 4 Grad-CAM rule and interpretation |

Implementation-source distinction:
- Paper/reference reported: FiLM is feature-wise affine modulation with gamma
  and beta generated from a condition.
- Paper/reference reported: applying FiLM after BatchNorm is a known setting in
  the FiLM implementation notes.
- Implementation choice for this thesis: in Stock_CNN blocks, insert FiLM as
  `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`.
- Implementation choice for today: no News/LLM condition.

## 한국어

| Source | Local/reference path | Stage 4 사용 위치 |
| --- | --- | --- |
| FiLM 논문 요약 | `자료조사/FiLM요약.md` | FiLM 수식, gamma/beta 해석, BN-FiLM 위치 |
| FiLM 논문 PDF | `자료조사/FiLM.pdf` | 최종 코드 주석 전 정확한 paper citation 확인 |
| FiLM reference repository | `https://github.com/ethanjperez/film`, commit `fe43ddf8a22b339dcca2efa07091ce9d498955cf` | FiLM generator/layer convention 참고 |
| Stage 2 BTC model | `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py` | I5/I20/I60 base CNN block variant |
| Stage 2 BTC pipeline | `stage2_btc_extension/src/stage2_btc/` | 고정된 BTC data, label, split, normalization, evaluation |
| Stage 3 Linear model | `stage3_linear_adapter/src/stage3_linear/models/linear_stock_cnn.py` | 비교 모델로만 사용 |
| Grad-CAM local summary | `자료조사/Grad-CAM요약.md` | Stage 4 Grad-CAM 규칙과 해석 |

구현 근거 구분:
- Paper/reference reported: FiLM은 condition에서 생성한 gamma와 beta로 feature-wise
  affine modulation을 적용합니다.
- Paper/reference reported: FiLM implementation note에서 BatchNorm 뒤 FiLM 적용은
  중요한 설정입니다.
- 이 논문 구현 선택: Stock_CNN block에서는
  `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`로 삽입합니다.
- 오늘 구현 선택: News/LLM condition은 사용하지 않습니다.
