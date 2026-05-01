# 2-I0 Implementation Readiness Review

## English

Status: complete.

Readiness verdict:
- Stage 2 is ready for implementation.
- The next item is `2-I1`, shared Stage 2 config/code scaffold.

What was checked:
- Planning checklist `2-0` through `2-8` is complete.
- BTC daily OHLCV data audit is complete.
- Image generation, label/split/normalization, model adaptation, evaluation,
  Grad-CAM, and Kaggle runner plans exist.
- Small schema/count/matrix CSV artifacts exist.

Main implementation constraints:
- Keep Stage 1/Stock_CNN-style CNN core.
- Use model variants by image window: I5, I20, I60.
- Use default Stage 2 `batch_size=128`.
- Do not use stock cross-sectional H-L decile portfolios for BTC.
- Generate BTC Grad-CAM for baseline runs.
- Keep actual logic in `src/` and `scripts/`, not inside the Kaggle notebook
  wrapper.

Detailed review:
- [Stage 2 implementation readiness review](../docs/stage2_implementation_readiness_review.md)

Task map:
- `stage2_btc_extension/reports/tables/stage2_implementation_task_map.csv`

## 한국어

상태: 완료.

Readiness 판정:
- Stage 2는 구현 단계로 넘어갈 준비가 됐습니다.
- 다음 항목은 `2-I1`, Stage 2 공통 config/code scaffold입니다.

확인한 것:
- planning checklist `2-0`부터 `2-8`까지 완료.
- BTC daily OHLCV data audit 완료.
- image generation, label/split/normalization, model adaptation, evaluation,
  Grad-CAM, Kaggle runner 계획 문서 존재.
- 작은 schema/count/matrix CSV artifact 존재.

주요 구현 제약:
- Stage 1/Stock_CNN식 CNN core를 유지합니다.
- image window별 model variant를 사용합니다: I5, I20, I60.
- Stage 2 기본 `batch_size=128`을 사용합니다.
- BTC에는 stock cross-sectional H-L decile portfolio를 사용하지 않습니다.
- BTC baseline run에서도 Grad-CAM을 생성합니다.
- 실제 로직은 Kaggle notebook wrapper가 아니라 `src/`와 `scripts/`에 둡니다.

상세 review:
- [Stage 2 implementation readiness review](../docs/stage2_implementation_readiness_review.md)

Task map:
- `stage2_btc_extension/reports/tables/stage2_implementation_task_map.csv`
