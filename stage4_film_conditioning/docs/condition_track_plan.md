# Stage 4 Condition Track Plan

## English

The user's proposed classification is correct as an experiment progression, but
it should be written as condition-source tracks so the implementation does not
mix different research questions.

Track 4A: FiLM-only control
- Purpose: verify FiLM insertion, initialization, logging, Grad-CAM, and
  gamma/beta export.
- No external sentiment/news information.
- If gamma/beta are static learned parameters, this is a mechanism control, not
  a conditioning result.

Track 4B: F&G index + FiLM
- Purpose: test a compact daily market-sentiment numeric signal.
- Required before implementation:
  source audit, date coverage, BTC date alignment, missing-day policy,
  no-future-leakage rule.
- Likely encoder: small MLP from numeric condition to block-wise gamma/beta.

Track 4C: News + non-LLM encoder + FiLM
- Purpose: test whether BTC news text helps the chart CNN when the news is
  encoded by a non-LLM text encoder.
- This is the first news-conditioned track.
- Required before implementation:
  news source audit, publication-time alignment, duplicate handling, daily
  aggregation rule, no-future-leakage rule.
- Likely encoder: embedding/GRU or pooled text encoder closer to the original
  FiLM setup.

Track 4D: News + LLM encoder + FiLM
- Purpose: test whether BTC news text helps the chart CNN when the news is
  encoded by an LLM.
- Deferred.
- Required before implementation:
  model choice, cache/version rule, prompt or embedding specification,
  reproducibility manifest, and cost/runtime plan.

## 한국어

사용자가 제안한 분류는 실험 진행 순서로 맞습니다. 다만 구현에서 연구 질문이 섞이지
않도록 condition-source track으로 적는 것이 좋습니다.

Track 4A: FiLM-only control
- 목적: FiLM 삽입, 초기화, logging, Grad-CAM, gamma/beta export가 제대로 되는지
  확인합니다.
- 외부 sentiment/news 정보는 사용하지 않습니다.
- gamma/beta가 static learned parameter라면 이것은 mechanism control이지,
  conditioning result가 아닙니다.

Track 4B: F&G index + FiLM
- 목적: compact daily market-sentiment numeric signal을 테스트합니다.
- 구현 전 필수:
  source audit, date coverage, BTC date alignment, missing-day policy,
  no-future-leakage rule.
- 예상 encoder: numeric condition에서 block-wise gamma/beta를 만드는 작은 MLP.

Track 4C: News + non-LLM encoder + FiLM
- 목적: BTC news text를 LLM이 아닌 text encoder로 condition vector로 바꿨을 때
  chart CNN에 도움이 되는지 테스트합니다.
- 첫 news-conditioned track입니다.
- 구현 전 필수:
  news source audit, publication-time alignment, duplicate handling, daily
  aggregation rule, no-future-leakage rule.
- 예상 encoder: 원 FiLM 구조에 가까운 embedding/GRU 또는 pooled text encoder.

Track 4D: News + LLM encoder + FiLM
- 목적: BTC news text를 LLM encoder로 condition vector로 바꿨을 때 chart CNN에
  도움이 되는지 테스트합니다.
- 나중으로 미룹니다.
- 구현 전 필수:
  model choice, cache/version rule, prompt 또는 embedding specification,
  reproducibility manifest, cost/runtime plan.
