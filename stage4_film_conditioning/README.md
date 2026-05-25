# Stage 4: Market Context Fusion + FiLM Conditioning

## English

This folder is the Stage 4 workspace for testing whether structured market
context can improve the BTC chart-image CNN through conditional feature
modulation.

Stage 4 objective:
- Keep the Stage 2 BTC image-generation, label, split, normalization,
  evaluation, trading metric, and Grad-CAM pipeline fixed.
- Use the selected five-seed Stage 2 baseline as the primary image model:
  `I60/R20/ohlc_ma_vb`.
- Build a structured market context vector from information available at or
  before the image end date `t`.
- Compare simple context fusion against FiLM-style conditional modulation.
- Use the comparison to answer why FiLM is needed rather than only adding more
  parameters or appending side information.

Primary Stage 4 image input:
- `I60/R20/ohlc_ma_vb`
- This is selected because the Stage 2 selected five-seed check found it to be
  the strongest candidate:
  - accuracy mean `0.5793`
  - accuracy std `0.0182`
  - ROC-AUC mean `0.5849`
- The seed-42 Stage 2 run is useful for metadata and Grad-CAM inspection, but
  the Stage 4 baseline claim should use the five-seed mean.

Stage 3 dependency position:
- Stage 3 Linear is not a Stage 4 architecture dependency.
- It is kept as a negative/simple-parameter ablation.
- The preliminary matching seed-42 test dropped from Stage 2 accuracy
  `0.603053` and ROC-AUC `0.616950` to Stage 3 Linear accuracy `0.541291` and
  ROC-AUC `0.522101`.

Structured market context candidates:
- Fear & Greed score.
- Bollinger %B.
- Bollinger bandwidth.
- Money Flow Index.
- Realized volatility.

Important modeling rule:
- These context values are not drawn into the chart image in the main Stage 4
  experiment.
- The chart image remains the Stage 2 `ohlc_ma_vb` image.
- The context values are fed as a separate numeric vector.
- The numeric context vector is normalized with train-only statistics, encoded
  by an MLP, and then used by the fusion/modulation model.

Main Stage 4 ablation models:

| Track | Model | What changes | Interpretation |
|:---|:---|:---|:---|
| `4-A` | CNN + context concat | Append context embedding to CNN feature before classifier | Tests whether simple side-information fusion is enough |
| `4-B` | CNN + context gating | Use context to multiply CNN channels/features by learned gates | Tests a simpler modulation alternative |
| `4-C` | CNN + context FiLM gamma-only | Use context to generate block-wise `gamma`; apply `F' = gamma * F` | Tests FiLM scaling without additive shift |
| `4-D` | CNN + context FiLM full | Use context to generate block-wise `gamma` and `beta`; apply `F' = gamma * F + beta` | Main FiLM model and interpretability target |

Why this matches the advisor's direction:
- The chart-image CNN baseline is already strong.
- The research question is not simply whether chart images work.
- The research question is whether market context can make the visual feature
  extractor more adaptive to regime/state.
- FiLM is useful because the path from context to feature modulation is explicit:
  the model produces gates/gamma/beta values that can be analyzed by date,
  market regime, layer, channel, confidence, and correctness.

Advisor direction file mapping:
- Source: `/Users/jaehyeonjeong/Desktop/film_chart_research_summary.md`.
- It frames FiLM as conditional modulation for chart-image features, not as a
  full visual-question-answering task.
- It states that the prediction query is effectively fixed, so an RNN question
  encoder is not essential.
- It recommends compact structured metadata and an MLP/embedding-based condition
  encoder.
- It recommends comparing CNN-only, naive condition concatenation, FiLM, and an
  optional attention-based fusion alternative.
- This is why Stage 4 uses structured numeric context first and compares
  concat, gating, gamma-only FiLM, and full FiLM.

News-context position:
- News is not removed from the thesis.
- Candidate source: Hugging Face `edaschau/bitcoin_news`.
- Public metadata checked on 2026-05-25 shows `210,832` rows, date range
  2011-2025, and columns such as `date_time`, `title`, `source`, `url`, and
  `article_text`.
- News requires source audit, publication-time alignment, daily aggregation, and
  encoder/cache rules before model training.
- Therefore it is kept as a second-phase context track after the structured
  numeric-context ablation is stable.
- 4-3 decision: first news version should be headline-only, strict `t-1`, and
  encoded with train-fit non-LLM text features.

Main documents:
- [Checklist](checklist.md)
- [Workflow diagram](workflow_diagram.md)
- [Stage 4 pipeline](docs/stage4_pipeline.md)
- [Context fusion ablation plan](docs/condition_track_plan.md)
- [Professor meeting direction brief](docs/professor_meeting_stage4_direction_brief.md)
- [FiLM insertion design](docs/film_insertion_design.md)
- [Source map](docs/source_map.md)
- [Planning report](checklist_results/4-1_context_fusion_and_news_plan.md)
- [News dataset audit](checklist_results/4-3_news_dataset_audit_and_feasibility.md)
- [Stage 2/Stage 3 dependency review](checklist_results/4-4_stage2_stage3_dependency_and_baseline_output_review.md)

## 한국어

이 폴더는 structured market context가 BTC chart-image CNN을 개선할 수 있는지
확인하는 Stage 4 작업 공간입니다. 핵심은 단순히 이미지를 더 복잡하게 그리는 것이
아니라, 시장 맥락이 CNN feature를 조건부로 조절하게 만드는 것입니다.

Stage 4 목표:
- Stage 2의 BTC image generation, label, split, normalization, evaluation,
  trading metric, Grad-CAM 파이프라인은 고정합니다.
- primary image model은 Stage 2 selected five-seed best인
  `I60/R20/ohlc_ma_vb`로 둡니다.
- image end date `t`까지 이용 가능한 정보만 사용해서 structured market context
  vector를 만듭니다.
- 단순 context fusion과 FiLM-style conditional modulation을 비교합니다.
- FiLM이 단순 parameter 증가나 side information 추가보다 왜 필요한지 실험적으로
  방어합니다.

Stage 4 primary image input:
- `I60/R20/ohlc_ma_vb`
- Stage 2 selected five-seed check에서 가장 강한 후보였기 때문에 사용합니다:
  - accuracy mean `0.5793`
  - accuracy std `0.0182`
  - ROC-AUC mean `0.5849`
- seed 42 Stage 2 run은 metadata와 Grad-CAM 확인에 유용하지만, Stage 4 baseline
  claim은 five-seed mean을 기준으로 둡니다.

Stage 3 dependency 위치:
- Stage 3 Linear는 Stage 4 architecture dependency가 아닙니다.
- negative/simple-parameter ablation으로만 둡니다.
- 같은 조합의 seed 42 preliminary test에서 Stage 2 accuracy `0.603053`,
  ROC-AUC `0.616950`이 Stage 3 Linear accuracy `0.541291`, ROC-AUC `0.522101`로
  하락했습니다.

Structured market context 후보:
- Fear & Greed score.
- Bollinger %B.
- Bollinger bandwidth.
- Money Flow Index.
- Realized volatility.

중요한 모델링 규칙:
- 이 context 값들을 main Stage 4 실험에서 chart image 위에 추가로 그리지 않습니다.
- chart image는 Stage 2의 `ohlc_ma_vb` 이미지를 그대로 사용합니다.
- context 값은 별도 numeric vector로 모델에 들어갑니다.
- numeric context vector는 train split 통계로만 normalize하고, MLP로 encoding한 뒤
  fusion/modulation model에서 사용합니다.

Stage 4 주요 ablation model:

| Track | Model | 바뀌는 부분 | 해석 |
|:---|:---|:---|:---|
| `4-A` | CNN + context concat | classifier 직전에 CNN feature와 context embedding을 붙임 | 단순 side information 추가만으로 충분한지 확인 |
| `4-B` | CNN + context gating | context가 CNN channel/feature gate를 만들어 곱함 | 더 단순한 modulation 대안 |
| `4-C` | CNN + context FiLM gamma-only | context가 block별 `gamma`를 만들고 `F' = gamma * F` 적용 | additive shift 없는 FiLM scaling |
| `4-D` | CNN + context FiLM full | context가 block별 `gamma`, `beta`를 만들고 `F' = gamma * F + beta` 적용 | main FiLM model과 해석력 대상 |

교수님 방향성과 맞는 이유:
- chart-image CNN baseline은 이미 강합니다.
- 핵심 질문은 chart image 자체가 예측력이 있는지가 아닙니다.
- 핵심 질문은 market context가 visual feature extractor를 regime/state에 따라
  더 적응적으로 만들 수 있는지입니다.
- FiLM은 context에서 feature modulation으로 가는 경로가 명시적입니다. 따라서
  gate/gamma/beta 값을 date, regime, layer, channel, confidence, correctness별로
  분석할 수 있습니다.

교수님 방향성 파일과의 연결:
- Source: `/Users/jaehyeonjeong/Desktop/film_chart_research_summary.md`.
- 해당 note는 FiLM을 full VQA가 아니라 chart-image feature의 conditional
  modulation으로 재정의합니다.
- 금융 예측 질문은 사실상 고정되어 있으므로 RNN question encoder가 필수는 아니라고
  정리합니다.
- compact structured metadata와 MLP/embedding-based condition encoder를 권장합니다.
- CNN-only, naive condition concatenation, FiLM, optional attention-based fusion
  비교를 권장합니다.
- 그래서 Stage 4는 structured numeric context를 먼저 사용하고 concat, gating,
  gamma-only FiLM, full FiLM을 비교합니다.

뉴스 context 위치:
- 뉴스는 논문에서 제거하지 않습니다.
- 후보 source: Hugging Face `edaschau/bitcoin_news`.
- 2026-05-25 기준 공개 metadata 확인 결과 `210,832` rows, 2011-2025 date range,
  `date_time`, `title`, `source`, `url`, `article_text` columns를 포함합니다.
- 뉴스는 source audit, publication-time alignment, daily aggregation,
  encoder/cache rule이 필요합니다.
- 따라서 structured numeric-context ablation이 안정화된 뒤 second-phase context
  track으로 유지합니다.
- 4-3 결정: 첫 news version은 headline-only, strict `t-1`, train-fit non-LLM
  text feature encoder로 시작합니다.

주요 문서:
- [Checklist](checklist.md)
- [Workflow diagram](workflow_diagram.md)
- [Stage 4 pipeline](docs/stage4_pipeline.md)
- [Context fusion ablation plan](docs/condition_track_plan.md)
- [Professor meeting direction brief](docs/professor_meeting_stage4_direction_brief.md)
- [FiLM insertion design](docs/film_insertion_design.md)
- [Source map](docs/source_map.md)
- [Planning report](checklist_results/4-1_context_fusion_and_news_plan.md)
- [News dataset audit](checklist_results/4-3_news_dataset_audit_and_feasibility.md)
- [Stage 2/Stage 3 dependency review](checklist_results/4-4_stage2_stage3_dependency_and_baseline_output_review.md)
