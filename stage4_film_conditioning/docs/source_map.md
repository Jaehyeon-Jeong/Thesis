# Stage 4 Source Map

## English

| Source | Local/reference path | Stage 4 use |
| --- | --- | --- |
| FiLM paper summary | `자료조사/FiLM요약.md` | FiLM equation, gamma/beta interpretation, BN-FiLM placement |
| FiLM paper PDF | `자료조사/FiLM.pdf` | Exact paper citation check before final code comments |
| Advisor direction note | `/Users/jaehyeonjeong/Desktop/film_chart_research_summary.md` | Stage 4 research framing: fixed chart-image baseline, structured context, MLP encoder, concat/gating/FiLM comparison, interpretability |
| FiLM reference repository | `https://github.com/ethanjperez/film`, commit `fe43ddf8a22b339dcca2efa07091ce9d498955cf` | Reference implementation for FiLM generator/layer conventions |
| Stage 2 BTC model | `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py` | Base CNN block variants for I5/I20/I60 |
| Stage 2 BTC pipeline | `stage2_btc_extension/src/stage2_btc/` | Fixed BTC data, labels, split, normalization, evaluation |
| Stage 3 Linear model | `stage3_linear_adapter/src/stage3_linear/models/linear_stock_cnn.py` | Comparison model only |
| Grad-CAM local summary | `자료조사/Grad-CAM요약.md` | Stage 4 Grad-CAM rule and interpretation |
| BTC news candidate dataset | `https://huggingface.co/datasets/edaschau/bitcoin_news` | Candidate second-phase news context source after source/date/leakage audit |

Implementation-source distinction:
- Paper/reference reported: FiLM is feature-wise affine modulation with gamma
  and beta generated from a condition.
- Paper/reference reported: applying FiLM after BatchNorm is a known setting in
  the FiLM implementation notes.
- Implementation choice for this thesis: in Stock_CNN blocks, insert FiLM as
  `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`.
- Advisor-direction mapping:
  - The advisor note says the task should not be treated as full VQA; the
    financial prediction query is effectively fixed.
  - It says structured metadata can replace the original FiLM question encoder.
  - It recommends comparing CNN-only, naive condition concatenation, FiLM, and
    optional attention-based fusion.
  - Therefore Stage 4 uses numeric context + MLP encoder and compares concat,
    gating, gamma-only FiLM, and full FiLM.
- Short excerpts from the advisor note that determine the model set:
  - "strong no-conditioning baseline"
  - "the prediction query is effectively fixed"
  - "compact structured metadata"
  - "an MLP or embedding-based condition encoder is a cleaner design than an RNN"
  - "CNN + naive condition concatenation"
  - "CNN + FiLM"
  - "Optional attention-based fusion"
- Implementation choice for the first Stage 4 main run: structured numeric
  context first; news context remains a second-phase track after audit.

## 한국어

| Source | Local/reference path | Stage 4 사용 위치 |
| --- | --- | --- |
| FiLM 논문 요약 | `자료조사/FiLM요약.md` | FiLM 수식, gamma/beta 해석, BN-FiLM 위치 |
| FiLM 논문 PDF | `자료조사/FiLM.pdf` | 최종 코드 주석 전 정확한 paper citation 확인 |
| 교수님 방향성 note | `/Users/jaehyeonjeong/Desktop/film_chart_research_summary.md` | Stage 4 연구 framing: 고정 chart-image baseline, structured context, MLP encoder, concat/gating/FiLM 비교, 해석력 |
| FiLM reference repository | `https://github.com/ethanjperez/film`, commit `fe43ddf8a22b339dcca2efa07091ce9d498955cf` | FiLM generator/layer convention 참고 |
| Stage 2 BTC model | `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py` | I5/I20/I60 base CNN block variant |
| Stage 2 BTC pipeline | `stage2_btc_extension/src/stage2_btc/` | 고정된 BTC data, label, split, normalization, evaluation |
| Stage 3 Linear model | `stage3_linear_adapter/src/stage3_linear/models/linear_stock_cnn.py` | 비교 모델로만 사용 |
| Grad-CAM local summary | `자료조사/Grad-CAM요약.md` | Stage 4 Grad-CAM 규칙과 해석 |
| BTC news 후보 dataset | `https://huggingface.co/datasets/edaschau/bitcoin_news` | source/date/leakage audit 이후 second-phase news context source 후보 |

구현 근거 구분:
- Paper/reference reported: FiLM은 condition에서 생성한 gamma와 beta로 feature-wise
  affine modulation을 적용합니다.
- Paper/reference reported: FiLM implementation note에서 BatchNorm 뒤 FiLM 적용은
  중요한 설정입니다.
- 이 논문 구현 선택: Stock_CNN block에서는
  `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`로 삽입합니다.
- 교수님 방향성 mapping:
  - 교수님 note는 이 task를 full VQA로 보지 말라고 정리합니다. 금융 예측 질문은
    사실상 고정되어 있습니다.
  - original FiLM의 question encoder 대신 structured metadata를 condition으로
    쓸 수 있다고 정리합니다.
  - CNN-only, naive condition concatenation, FiLM, optional attention-based
    fusion 비교를 권장합니다.
  - 따라서 Stage 4는 numeric context + MLP encoder를 사용하고 concat, gating,
    gamma-only FiLM, full FiLM을 비교합니다.
- 네 가지 model set을 결정한 교수님 note 원문 발췌:
  - "strong no-conditioning baseline"
  - "the prediction query is effectively fixed"
  - "compact structured metadata"
  - "an MLP or embedding-based condition encoder is a cleaner design than an RNN"
  - "CNN + naive condition concatenation"
  - "CNN + FiLM"
  - "Optional attention-based fusion"
- 첫 Stage 4 main run 구현 선택: structured numeric context를 먼저 사용합니다.
  news context는 audit 이후 second-phase track으로 유지합니다.
