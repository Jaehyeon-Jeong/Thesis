# 4-I0 Implementation Readiness Review

## English

Status: complete.

Readiness decision:
- Stage 4 can move from planning to implementation.
- The next item is `4-I1`, shared Stage 4 config/code scaffold.
- The first implementation target is the structured numeric-context track, not
  the news/LLM extension.

What was checked:
- Planning checklist `4-0` through `4-8` is complete.
- The primary Stage 2 baseline is locked:
  `I60/R20/ohlc_ma_vb`, five-seed accuracy mean `0.5793`, ROC-AUC mean
  `0.5849`.
- The first context vector, context encoder, concat/gating/FiLM insertion
  points, Grad-CAM/export policy, and Kaggle backup contract are documented.
- Stage 2 code reuse is required. Stage 4 should import Stage 2 data/image/split
  and evaluation helpers instead of rewriting them.

Main implementation constraints:
- Do not edit Stage 2 image generation, split, normalization, evaluation, or
  trading metric code.
- Stage 4 must add a `stage2_dependency` config section so Kaggle/local scripts
  can import Stage 2 `src`.
- Local BTC OHLCV data exists, but local F&G data is not present in the active
  dataset folder. Full context feature construction is therefore a Kaggle-first
  path unless a local F&G CSV is supplied.
- Output completion must be based on the Stage 4 output checker, not checkpoint
  existence.

Detailed review:
- [Stage 4 implementation readiness review](../docs/stage4_implementation_readiness_review.md)

Task map:
- `stage4_film_conditioning/reports/tables/stage4_implementation_task_map.csv`

## 한국어

상태: 완료.

Readiness 결정:
- Stage 4는 planning에서 implementation으로 넘어갈 수 있습니다.
- 다음 항목은 `4-I1`, Stage 4 공통 config/code scaffold입니다.
- 첫 구현 대상은 structured numeric-context track이며, news/LLM 확장이 아닙니다.

확인한 것:
- Planning checklist `4-0`부터 `4-8`까지 완료.
- Primary Stage 2 baseline 고정:
  `I60/R20/ohlc_ma_vb`, five-seed accuracy mean `0.5793`, ROC-AUC mean
  `0.5849`.
- 첫 context vector, context encoder, concat/gating/FiLM 삽입 위치,
  Grad-CAM/export 정책, Kaggle backup 계약이 문서화되어 있습니다.
- Stage 2 code 재사용이 필요합니다. Stage 4는 Stage 2 data/image/split/evaluation
  helper를 재작성하지 않고 import해야 합니다.

주요 구현 제약:
- Stage 2 image generation, split, normalization, evaluation, trading metric
  code를 수정하지 않습니다.
- Stage 4 config에는 Kaggle/local script가 Stage 2 `src`를 import할 수 있도록
  `stage2_dependency` section이 필요합니다.
- 로컬 BTC OHLCV data는 있지만 active dataset folder에 F&G data는 없습니다.
  따라서 local F&G CSV를 제공하지 않는 한 full context feature construction은
  Kaggle-first path입니다.
- 완료 판정은 checkpoint 존재가 아니라 Stage 4 output checker 기준입니다.

상세 review:
- [Stage 4 implementation readiness review](../docs/stage4_implementation_readiness_review.md)

Task map:
- `stage4_film_conditioning/reports/tables/stage4_implementation_task_map.csv`
