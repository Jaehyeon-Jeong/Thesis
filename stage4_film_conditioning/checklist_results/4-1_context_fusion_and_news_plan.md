# 4-1 Context Fusion and News Plan

## English

Status: complete

Stage 4 is now defined as a **market-context fusion/modulation comparison** on
top of the fixed Stage 2 BTC chart-image CNN.

## Advisor Direction Mapping

The advisor direction file is:

`/Users/jaehyeonjeong/Desktop/film_chart_research_summary.md`

Key parts and how they map to the current Stage 4 plan:

| Advisor-direction point | Stage 4 decision |
|:---|:---|
| "The starting point is the chart-image paradigm..." | Keep Stage 2 chart-image CNN fixed instead of redesigning the image pipeline. |
| "The main open question is no longer whether chart images can be predictive..." | Use the selected Stage 2 `I60/R20/ohlc_ma_vb` result as the strong CNN-only baseline. |
| "Whether conditional modulation improves robustness, transfer, or regime sensitivity..." | Test market-context fusion/modulation rather than adding another plain CNN/Linear layer. |
| "The prediction query is effectively fixed..." | Do not treat the task as text-question VQA. The fixed question is R20 Up/Down. |
| "The conditioning signal can instead be represented by compact structured metadata..." | Use F&G, Bollinger %B, Bollinger bandwidth, MFI, and realized volatility as numeric context. |
| "An MLP or embedding-based condition encoder is a cleaner design than an RNN." | Encode numeric context with an MLP before concat/gating/FiLM. |
| "A clean experimental design should separate the value of chart images from the value of conditioning." | Keep CNN-only and Stage 3 Linear as baselines, then compare context concat/gating/FiLM. |
| "CNN + naive condition concatenation" | Add `4-A CNN + context concat`. |
| "Optional attention-based fusion" | Add `4-B CNN + context gating` as a simpler attention/gating alternative. |
| "CNN + FiLM" and feature-wise affine modulation | Add `4-C gamma-only FiLM` and `4-D full gamma/beta FiLM`. |
| "The path from condition input to visual feature modulation is explicit." | Save context features, gates, gamma/beta, Grad-CAM, and correctness/confidence metadata. |
| "The contribution is conditional visual modulation, not generic language-style reasoning." | News/LLM is deferred until leakage-safe daily news vectors are defined; it is not the first main run. |

## Main Stage 4 Ablation

The four immediate experiments are:

1. `4-A CNN + context concat`
2. `4-B CNN + context gating`
3. `4-C CNN + context FiLM gamma-only`
4. `4-D CNN + context FiLM full`

The baseline image/model is:

```text
I60 / R20 / ohlc_ma_vb
```

This is the selected five-seed Stage 2 best configuration:
- accuracy mean `0.5793`;
- ROC-AUC mean `0.5849`.

## Why News Is Deferred but Preserved

News context is not removed from the thesis. It is moved into a second-phase
context track because:

- news requires publication-time alignment;
- article text is large/noisy;
- daily aggregation must be fixed;
- LLM summaries or embeddings require cache, version, and prompt policy;
- numeric context is faster and cleaner for first validating the fusion method.

The candidate news dataset is `edaschau/bitcoin_news` on Hugging Face. Public
metadata checked on 2026-05-21 shows:

- about `210,832` rows;
- date range around 2011-06-22 to 2025-06-04;
- columns including `date_time`, `title`, `source`, `url`, and `article_text`;
- overlap with the BTC Stage 2 test period.

Recommended news sequence:

1. Audit the dataset.
2. Align publication time to BTC image end date.
3. Start with headline-only daily aggregation.
4. Use non-LLM encoding first: TF-IDF/SVD or trainable embedding + GRU.
5. Use LLM summary/embedding later only after cache and reproducibility are
   fixed.
6. Apply the same four fusion heads to news context:
   concat, gating, gamma-only FiLM, full FiLM.

## Files Updated

- `PLAN.md`
- `stage4_film_conditioning/README.md`
- `stage4_film_conditioning/checklist.md`
- `stage4_film_conditioning/workflow_diagram.md`
- `stage4_film_conditioning/docs/stage4_pipeline.md`
- `stage4_film_conditioning/docs/condition_track_plan.md`
- `stage4_film_conditioning/docs/news_context_plan.md`
- `stage4_film_conditioning/docs/source_map.md`

## 한국어

상태: 완료

Stage 4는 이제 고정된 Stage 2 BTC chart-image CNN 위에서 **market-context
fusion/modulation을 비교하는 단계**로 정리했습니다.

## 교수님 방향성 파일과의 연결

교수님 방향성 파일:

`/Users/jaehyeonjeong/Desktop/film_chart_research_summary.md`

핵심 내용과 현재 Stage 4 결정의 연결:

| 교수님 방향성 파일의 핵심 | Stage 4 결정 |
|:---|:---|
| "chart-image paradigm"에서 출발 | Stage 2 chart-image CNN을 고정하고 이미지 파이프라인을 다시 바꾸지 않음 |
| chart image가 예측력이 있는지는 이미 baseline이 보여줌 | selected Stage 2 `I60/R20/ohlc_ma_vb`를 강한 CNN-only baseline으로 사용 |
| 조건부 modulation이 robustness/regime sensitivity를 개선하는지가 질문 | 단순 CNN/Linear 추가가 아니라 context fusion/modulation을 비교 |
| prediction query는 사실상 고정 | text-question VQA로 보지 않음. 질문은 R20 Up/Down으로 고정 |
| condition은 compact structured metadata로 표현 가능 | F&G, Bollinger %B, Bollinger bandwidth, MFI, realized volatility를 numeric context로 사용 |
| MLP/embedding-based condition encoder가 RNN보다 clean | numeric context는 MLP로 encoding |
| chart image value와 conditioning value를 분리해서 실험해야 함 | CNN-only/Stage 3 Linear baseline 뒤에 concat/gating/FiLM 비교 |
| naive condition concatenation 필요 | `4-A CNN + context concat` 추가 |
| attention-based fusion 비교 필요 | `4-B CNN + context gating`을 attention/gating 대안으로 추가 |
| CNN + FiLM 필요 | `4-C gamma-only FiLM`, `4-D full gamma/beta FiLM` 추가 |
| condition input에서 visual feature modulation으로 가는 경로가 명시적 | context, gate, gamma/beta, Grad-CAM, correctness/confidence metadata 저장 |
| contribution은 generic language reasoning이 아니라 conditional visual modulation | News/LLM은 첫 main run이 아니라 leakage-safe daily vector가 정의된 뒤 second-phase로 진행 |

## Stage 4 main ablation

바로 진행할 네 가지 실험은 다음입니다.

1. `4-A CNN + context concat`
2. `4-B CNN + context gating`
3. `4-C CNN + context FiLM gamma-only`
4. `4-D CNN + context FiLM full`

기준 image/model은 다음입니다.

```text
I60 / R20 / ohlc_ma_vb
```

이는 Stage 2 selected five-seed best configuration입니다.
- accuracy mean `0.5793`;
- ROC-AUC mean `0.5849`.

## 뉴스는 왜 미루되 유지하는가

뉴스 context는 논문에서 제거하지 않습니다. 다만 second-phase context track으로 둡니다.

이유:
- publication-time alignment가 필요합니다.
- article text가 크고 noisy합니다.
- daily aggregation rule을 고정해야 합니다.
- LLM summary/embedding은 cache, version, prompt policy가 필요합니다.
- numeric context가 fusion method 검증에는 더 빠르고 leakage audit이 쉽습니다.

뉴스 후보 dataset은 Hugging Face의 `edaschau/bitcoin_news`입니다. 2026-05-21 기준
공개 metadata 확인 결과:

- 약 `210,832` rows;
- date range 약 2011-06-22 to 2025-06-04;
- `date_time`, `title`, `source`, `url`, `article_text` columns 포함;
- BTC Stage 2 test period와 겹칩니다.

추천 뉴스 진행 순서:

1. Dataset audit.
2. Publication time과 BTC image end date 정렬.
3. Headline-only daily aggregation부터 시작.
4. Non-LLM encoding first: TF-IDF/SVD 또는 trainable embedding + GRU.
5. Cache/reproducibility가 고정된 뒤 LLM summary/embedding.
6. News context에도 같은 네 가지 fusion head 적용:
   concat, gating, gamma-only FiLM, full FiLM.

## 업데이트한 파일

- `PLAN.md`
- `stage4_film_conditioning/README.md`
- `stage4_film_conditioning/checklist.md`
- `stage4_film_conditioning/workflow_diagram.md`
- `stage4_film_conditioning/docs/stage4_pipeline.md`
- `stage4_film_conditioning/docs/condition_track_plan.md`
- `stage4_film_conditioning/docs/news_context_plan.md`
- `stage4_film_conditioning/docs/source_map.md`
