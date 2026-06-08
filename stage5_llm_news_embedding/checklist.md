# Stage 5 Checklist

Stage 5 tests LLM-derived news representations as FiLM context. The main design
choice is fixed:

- Use LLM/news-derived numeric context features: OpenAI embeddings first,
  FinBERT sentiment second, and prompt/event labels only if needed.
- Use prompt labels only as an auxiliary interpretability feature.
- Preserve the Stage 2 visual baseline by loading pretrained checkpoints and
  freezing the visual CNN/classifier unless a specific ablation says otherwise.

## Fixed Baseline

- Primary visual baseline: Stage 2 `I60/R20/ohlc_ma_vb`.
- Baseline accuracy mean: `0.5793`.
- Baseline ROC-AUC mean: `0.5849`.
- Primary Stage 5 protocol: Stage2 frozen + bounded last-block FiLM.

## Planning And Scaffold

- [x] 5-0. Stage 5 scaffold
  - Folder, checklist, workflow, source map, design document, notebook/script
    placeholders.
  - Result: [5-0 scaffold](checklist_results/5-0_stage5_scaffold.md)

- [x] 5-1. Literature/method lock
  - Confirm Chen/Kelly/Xiu style embedding as the main representation path.
  - Confirm Lopez-Lira/Tang style prompt label as auxiliary interpretation.
  - Decide whether OpenAI embedding API or local open-source embedding is the
    first executable path.
  - Decision: first executable path is OpenAI `text-embedding-3-small`.
  - Result: [5-1 method/backend lock](checklist_results/5-1_literature_method_backend_lock.md)

- [x] 5-2. News coverage and leakage audit
  - Reuse Stage 4 news source audit where possible.
  - Confirm news date range covers Stage 4 train/validation/test sample dates.
  - Enforce `t-1` or explicit publication-time lag before each image end date.
  - Output: coverage table, missing-day table, leakage policy.
  - Decision: existing `edaschau/bitcoin_news` source covers all Stage 5
    7/20/60-day trailing windows under strict `t-1` alignment.
  - Result: [5-2 news coverage and leakage audit](checklist_results/5-2_news_coverage_leakage_audit.md)

- [x] 5-3. Embedding input construction
  - Build a deterministic input table for each deduplicated news item.
  - Primary unit: headline/news-item-level embedding.
  - Locked template: headline-only text; date/source/URL kept as metadata.
  - Usable range: `2018-02-28` to `2024-12-10`, matching 60-day
    trailing windows with strict `t-1` lag.
  - Result: [5-3 embedding input construction](checklist_results/5-3_embedding_input_construction.md)

- [x] 5-4. Embedding backend decision and cache policy
  - Primary: OpenAI `text-embedding-3-small`.
  - Anthropic-side comparison: Voyage AI `voyage-4` because Claude has no
    native embedding endpoint.
  - Optional stronger checks: OpenAI `text-embedding-3-large`, Voyage AI
    `voyage-4-large`.
  - Cache key: `embedding_input_hash + provider + model + dimensions`.
  - API keys: environment variables only; never written to configs or outputs.
  - Result: [5-4 backend and cache policy](checklist_results/5-4_embedding_backend_cache_policy.md)

- [x] 5-5. Build headline-level embedding table
  - Input: cleaned news item table.
  - Output:
    `stage5_news_embedding_items.parquet` or equivalent local/Kaggle artifact.
  - Required columns: item id, date, source, title hash, embedding model,
    embedding dimension, embedding vector path/key.
  - Completed OpenAI `text-embedding-3-small` run: `24,281 x 1,536` vectors.
  - Failed rows: `0`.
  - Result: [5-5 headline embedding table](checklist_results/5-5_headline_embedding_table.md)

- [x] 5-6. Build trailing-window embedding context
  - Aggregate item embeddings by sample date:
    mean, time-decay mean, count, and missing indicators for 7/20/60 windows.
  - Fit aggregation/scaler only on train where applicable.
  - Outputs: sample metadata CSV, window mean vectors, time-decay mean vectors,
    context manifest.
  - Count match vs Stage4 deduped headline windows: `1.000` for 7/20/60.
  - Result: [5-6 embedding context windows](checklist_results/5-6_embedding_context_windows.md)

- [x] 5-7. Train-only SVD/PCA dimension grid
  - Fit dimensionality reduction on train split only.
  - Initial grid: `8`, `16`, `32` dimensions.
  - Keep raw embedding dimension out of the model unless sample size supports it.
  - Result: [5-7 SVD/PCA dimension grid](checklist_results/5-7_train_only_svd_pca_dimension_grid.md)

## Model Experiments

- [x] 5-8A. Stage2 frozen + embedding-only bounded FiLM
  - Context: embedding SVD features + news counts.
  - Model: bounded last-block FiLM, scale `0.02`.
  - Seeds: `42,43,44,45,46`.
  - Goal: check whether semantic news vectors beat TF-IDF/SVD news context.
  - Prepared runner: [Kaggle 5-8A one-cell](notebooks/kaggle_stage5_2_embedding_film_ablation_one_cell.md).
  - Prepared context: `mean/dim16`, `51` context features
    (`3` news-count features + `48` embedding-SVD features).
  - Prepared input bundle:
    `stage5_llm_news_embedding_5_8_compact_context_bundle.zip`.
  - Preparation note: [5-8A prepared](checklist_results/5-8A_embedding_only_bounded_film_prepared.md)
  - Result: mean accuracy `0.5782`, ROC-AUC `0.5844`; essentially tied
    with but slightly below Stage2 `ohlc_ma_vb` baseline (`0.5793`, `0.5849`).
  - Result note: [5-8A results](checklist_results/5-8A_embedding_only_bounded_film_results.md)

- [x] 5-8B. Stage2 frozen + embedding + F&G bounded FiLM
  - Context: embedding SVD features + news counts + F&G-only features.
  - Decision: closed without a separate run. Embedding-only was below the
    baseline and the stronger combined regime test is now `5-9E`
    FinBERT + F&G, where the news signal is explicitly sentiment-aligned.

- [x] 5-8C. Scale and dimension sanity grid
  - Only if 5-8A/B are stable.
  - Candidate scales: `0.02`, `0.05`, `0.10`.
  - Candidate dimensions: best two from 5-7.
  - Partial result: `mean/dim32`, scale `0.05` completed. Mean accuracy
    `0.5768`, ROC-AUC `0.5847`. It slightly improves Brier/AP but worsens
    accuracy, F1, and trading metrics, so broad scale/dimension search is not
    the main next step.
  - Result note: [5-8C dim32 scale 0.05](checklist_results/5-8C_dim32_s0p05_embedding_film_results.md)
  - Closeout: no further embedding scale/dimension expansion before the
    FinBERT + F&G final comparison.

- [x] 5-9. FinBERT news sentiment regime proxy
  - Run this before GPT/Claude prompt labels.
  - Rationale: generic embeddings were not explicitly aligned to BTC
    direction/regime. FinBERT gives a cheaper, reproducible financial text tone
    signal.
  - Important distinction: FinBERT is headline-level financial tone, while F&G
    is market-level sentiment/regime. They are related but not the same source.
  - 5-9A: headline-level FinBERT sentiment extraction.
  - 5-9B: sampled output audit.
  - 5-9C: 7/20/60-day aggregation and news-only sentiment/F&G proxy.
  - 5-9D: Stage2 frozen + FinBERT-only bounded FiLM.
  - 5-9E: Stage2 frozen + FinBERT + F&G bounded FiLM.
  - 5-9F: compare against embedding, F&G-only, and Stage2 baseline.
  - Plan: [5-9 FinBERT sentiment proxy](checklist_results/5-9_finbert_sentiment_proxy_plan.md)
  - 5-9A prepared: [FinBERT extraction runner](checklist_results/5-9A_finbert_sentiment_prepared.md)
  - 5-9A result: [FinBERT sentiment extraction](checklist_results/5-9A_finbert_sentiment_results.md)
  - 5-9B result: [FinBERT sentiment output audit](checklist_results/5-9B_finbert_sentiment_audit_results.md)
  - 5-9C result: [FinBERT sentiment context features](checklist_results/5-9C_finbert_context_features_results.md)
    - Exported `91` FinBERT context features over `2,399` samples.
    - 7/20/60-day missing rates are all `0.0000`; count match rates are all
      `1.0000`.
    - Test-period sentiment is materially more positive than train/validation,
      so `5-9D` must verify whether this is useful regime signal or just a
      distribution shift.
  - 5-9D prepared: [FinBERT-only bounded FiLM runner](checklist_results/5-9D_finbert_only_bounded_film_prepared.md)
    - Context dimension after excluding diagnostics: `79`.
    - Kaggle one-cell:
      `notebooks/kaggle_stage5_9d_finbert_film_ablation_one_cell.md`.
    - Compact upload bundle:
      `stage5_llm_news_embedding_5_9d_finbert_context_bundle.zip`.
  - 5-9D result: [FinBERT-only bounded FiLM](checklist_results/5-9D_finbert_only_bounded_film_results.md)
    - Mean accuracy `0.578487`, ROC-AUC `0.586072`, AP `0.611943`.
    - It does not beat Stage2 `ohlc_ma_vb` accuracy (`0.579320`) or N8B
      F&G-only accuracy (`0.580291`), but it slightly improves ROC-AUC/AP/Brier.
    - Main issue: test-period FinBERT sentiment/news-count features are shifted
      positive and create Up-bias; mean predicted positive rate is `0.637196`
      vs true positive rate `0.541291`.
  - 5-9E prepared: [FinBERT + F&G bounded FiLM runner](checklist_results/5-9E_finbert_fg_bounded_film_prepared.md)
    - Context dimension verified locally: `83` = `79` FinBERT numeric features
      + `4` F&G raw regime features.
    - Kaggle one-cell:
      `notebooks/kaggle_stage5_9e_finbert_fg_film_ablation_one_cell.md`.
  - 5-9E result: [FinBERT + F&G bounded FiLM](checklist_results/5-9E_finbert_fg_bounded_film_results.md)
    - Mean accuracy `0.580569`, ROC-AUC `0.585843`, AP `0.611899`,
      Brier `0.272701`.
    - Versus Stage2 `ohlc_ma_vb`: accuracy `+0.001249`, ROC-AUC
      `+0.000981`, Brier `-0.001636`.
    - Versus N8B F&G-only: accuracy `+0.000278`, ROC-AUC `+0.000913`,
      Brier `-0.001304`.
    - Interpretation: small positive/near-tie result. It does not justify a
      strong performance-improvement claim, but it shows that explicit
      headline-level sentiment can be combined with F&G without destabilizing
      the frozen Stage2 + bounded FiLM protocol.
  - 5-9F comparison: [Stage2 vs F&G vs FinBERT vs FinBERT + F&G](checklist_results/5-9F_stage5_finbert_comparison.md)
    - Decision: close the first FinBERT branch. Continue with
      correction/regression analysis, or use prompt/event labels only if a
      richer interpretable news feature is still needed.

- [ ] 5-10. Prompt/event auxiliary features
  - Execute only if FinBERT sentiment is insufficient or if a richer
    interpretation layer is needed.
  - Fixed GPT/Claude prompt:
    BTC relevance, direction, event type, impact horizon, risk regime,
    confidence, and short reason.
  - Convert labels to fixed numerical features only after caching raw outputs.
  - Use as structured feature extraction, not as a free-form predictor.

## Interpretation And Reporting

- [x] 5-11. Correction/regression analysis
  - Compare Stage2 baseline vs Stage5 candidates.
  - Required buckets:
    Stage2 wrong -> Stage5 correct,
    Stage2 correct -> Stage5 wrong,
    high-news-count,
    high prompt-confidence,
    positive/negative FinBERT regimes,
    positive/negative/unknown prompt regimes.
  - 5-11A partial result: [Stage2 vs FinBERT + F&G correction counts](checklist_results/5-11A_finbert_fg_correction_counts.md)
    - Total matched decisions: `7,205`.
    - Corrections: `95`.
    - Regressions: `86`.
    - Net corrections: `+9`.
    - Changed predictions: `181`, or `2.51%`.
    - Interpretation: small but real net correction; bounded FiLM remains
      conservative and changes only a small subset of Stage2 decisions.
  - 5-11 full condition result: [FinBERT + F&G conditional correction analysis](checklist_results/5-11_finbert_fg_condition_analysis.md)
    - Stage2 uncertain 45-55 bucket: delta accuracy `+0.012484`.
    - F&G greed bucket: delta accuracy `+0.010849`.
    - Low 60-day news-count and F&G neutral buckets are negative.
    - Selected sample CSV for 5-12:
      `reports/tables/stage5_5_11_finbert_fg_condition_analysis_selected_samples_for_5_12.csv`.

- [ ] 5-12. Grad-CAM and FiLM modulation export
  - For selected correction/regression samples, export:
    Stage2 Grad-CAM,
    Stage5 Grad-CAM,
    gamma/beta summary,
    news window text,
    embedding/FinBERT/prompt features,
    probability change.

- [ ] 5-13. Final Stage 5 report and thesis-title decision
  - Decide whether the final thesis can claim LLM/news context contribution.
  - Separate claims:
    overall accuracy,
    conditional improvement,
    interpretability,
    limitations.

## 한국어

Stage 5는 LLM에서 만든 뉴스 표현을 FiLM context로 넣는 실험입니다. 현재
큰 방향은 고정합니다.

- OpenAI embedding을 먼저 평가했고, 이후 FinBERT sentiment를 평가합니다.
  Prompt/event label은 필요할 때만 해석 보조로 사용합니다.
- Prompt label은 해석 보조로만 먼저 사용합니다.
- Stage 2 visual baseline은 pretrained checkpoint를 load하고 freeze합니다.
- 별도 ablation이 아니면 CNN/classifier를 다시 흔들지 않습니다.

## 고정 Baseline

- Primary visual baseline: Stage 2 `I60/R20/ohlc_ma_vb`.
- Baseline accuracy mean: `0.5793`.
- Baseline ROC-AUC mean: `0.5849`.
- Primary Stage 5 protocol: Stage2 frozen + bounded last-block FiLM.

## 계획 및 Scaffold

- [x] 5-0. Stage 5 scaffold
  - Folder, checklist, workflow, source map, design document, notebook/script
    placeholder 생성.
  - 결과: [5-0 scaffold](checklist_results/5-0_stage5_scaffold.md)

- [x] 5-1. Literature/method/backend lock
  - Chen/Kelly/Xiu식 embedding representation을 main path로 고정.
  - Lopez-Lira/Tang식 prompt label은 auxiliary interpretation으로 고정.
  - 첫 실행 backend는 OpenAI `text-embedding-3-small` API로 고정.
  - 결과: [5-1 method/backend lock](checklist_results/5-1_literature_method_backend_lock.md)

- [x] 5-2. News coverage and leakage audit
  - Stage 4 news audit를 재사용할 수 있는지 확인.
  - 뉴스 기간이 Stage 4 train/validation/test sample 날짜를 커버하는지 확인.
  - 각 image end date 기준 `t-1` 또는 publication-time lag 적용.
  - 산출물: coverage table, missing-day table, leakage policy.
  - 결정: 기존 `edaschau/bitcoin_news` source는 strict `t-1` alignment에서
    Stage 5 7/20/60일 trailing window를 전부 커버함.
  - 결과: [5-2 news coverage and leakage audit](checklist_results/5-2_news_coverage_leakage_audit.md)

- [x] 5-3. Embedding input construction
  - 중복 제거된 뉴스 item마다 deterministic input table 생성.
  - 기본 단위는 headline/news-item-level embedding.
  - 고정 template: headline-only text; date/source/URL은 metadata로만 보관.
  - 사용 가능 기간: strict `t-1` lag와 60일 trailing window 기준
    `2018-02-28` to `2024-12-10`.
  - 결과: [5-3 embedding input construction](checklist_results/5-3_embedding_input_construction.md)

- [x] 5-4. Embedding backend/cache policy
  - Primary: OpenAI `text-embedding-3-small`.
  - Anthropic-side comparison: Claude native embedding이 없으므로 Voyage AI
    `voyage-4`.
  - Optional stronger checks: OpenAI `text-embedding-3-large`, Voyage AI
    `voyage-4-large`.
  - Cache key: `embedding_input_hash + provider + model + dimensions`.
  - API key는 코드/CSV/JSON/log에 저장하지 않고 환경변수로만 사용.
  - 결과: [5-4 backend and cache policy](checklist_results/5-4_embedding_backend_cache_policy.md)

- [x] 5-5. Headline-level embedding table 생성
  - 입력: cleaned news item table.
  - 출력: `stage5_news_embedding_items.parquet` 또는 Kaggle/local artifact.
  - 필수 columns: item id, date, source, title hash, embedding model,
    embedding dimension, embedding vector path/key.
  - OpenAI `text-embedding-3-small` 실행 완료: `24,281 x 1,536` vectors.
  - 실패 row: `0`.
  - 결과: [5-5 headline embedding table](checklist_results/5-5_headline_embedding_table.md)

- [x] 5-6. Trailing-window embedding context 생성
  - 7/20/60일 window별 item embedding aggregation:
    mean, time-decay mean, count, missing indicator.
  - 필요한 scaler/aggregation fit은 train split에서만 수행.
  - 출력: sample metadata CSV, window mean vectors, time-decay mean vectors,
    context manifest.
  - Stage4 deduped headline window count와 7/20/60 모두 `1.000` 일치.
  - 결과: [5-6 embedding context windows](checklist_results/5-6_embedding_context_windows.md)

- [x] 5-7. Train-only SVD/PCA dimension grid
  - SVD/PCA는 train split에만 fit하고 validation/test는 transform만 수행.
  - 초기 grid: `8`, `16`, `32`.
  - 현재 BTC sample 수에서는 raw 1536차원을 모델에 직접 넣지 않음.
  - 결과: [5-7 SVD/PCA dimension grid](checklist_results/5-7_train_only_svd_pca_dimension_grid.md)

## 모델 실험

- [x] 5-8A. Stage2 frozen + embedding-only bounded FiLM
  - Context: embedding SVD features + news counts.
  - Model: bounded last-block FiLM, scale `0.02`.
  - Seeds: `42,43,44,45,46`.
  - 목적: semantic news vector가 기존 TF-IDF/SVD news context보다 나은지 확인.
  - 실행 준비: [Kaggle 5-8A one-cell](notebooks/kaggle_stage5_2_embedding_film_ablation_one_cell.md).
  - 준비된 context: `mean/dim16`, context feature `51`개
    (`news count 3개 + embedding SVD 48개`).
  - Kaggle 업로드용 bundle:
    `stage5_llm_news_embedding_5_8_compact_context_bundle.zip`.
  - 준비 메모: [5-8A prepared](checklist_results/5-8A_embedding_only_bounded_film_prepared.md)
  - 결과: mean accuracy `0.5782`, ROC-AUC `0.5844`; Stage2
    `ohlc_ma_vb` baseline (`0.5793`, `0.5849`)과 거의 같지만 약간 낮음.
  - 결과 메모: [5-8A results](checklist_results/5-8A_embedding_only_bounded_film_results.md)

- [x] 5-8B. Stage2 frozen + embedding + F&G bounded FiLM
  - Context: embedding SVD features + news counts + F&G-only features.
  - 결정: 별도 실행 없이 닫습니다. Embedding-only가 baseline보다 낮았고,
    더 의미 있는 결합 실험은 sentiment-aligned news signal인 FinBERT와
    F&G를 합치는 `5-9E`로 진행합니다.

- [x] 5-8C. Scale and dimension sanity grid
  - 5-8A/B가 안정적일 때만 실행.
  - 후보 scale: `0.02`, `0.05`, `0.10`.
  - 후보 dimension: 5-7에서 좋은 두 개.
  - 부분 결과: `mean/dim32`, scale `0.05` 완료. Mean accuracy `0.5768`,
    ROC-AUC `0.5847`. Brier/AP는 아주 조금 좋아졌지만 accuracy, F1,
    trading metric은 나빠졌으므로 넓은 scale/dimension grid를 계속
    확장하는 것은 우선순위가 낮음.
  - 결과 메모: [5-8C dim32 scale 0.05](checklist_results/5-8C_dim32_s0p05_embedding_film_results.md)
  - Closeout: FinBERT + F&G 최종 비교 전에는 embedding scale/dimension
    확장을 더 진행하지 않습니다.

- [x] 5-9. FinBERT news sentiment regime proxy
  - GPT/Claude prompt label 전에 먼저 실행.
  - 이유: generic embedding은 BTC 방향성/regime에 직접 정렬된 feature가
    아니었음. FinBERT는 더 저렴하고 재현 가능한 금융 텍스트 tone signal.
  - 중요한 구분: FinBERT는 headline-level 금융 텍스트 tone이고, F&G는
    market-level sentiment/regime index. 둘은 관련 있지만 같은 정보는 아님.
  - 5-9A: headline-level FinBERT sentiment extraction.
  - 5-9B: sampled output audit.
  - 5-9C: 7/20/60일 aggregation 및 news-only sentiment/F&G proxy 생성.
  - 5-9D: Stage2 frozen + FinBERT-only bounded FiLM.
  - 5-9E: Stage2 frozen + FinBERT + F&G bounded FiLM.
  - 5-9F: embedding, F&G-only, Stage2 baseline과 비교.
  - 계획: [5-9 FinBERT sentiment proxy](checklist_results/5-9_finbert_sentiment_proxy_plan.md)
  - 5-9A 준비 완료: [FinBERT extraction runner](checklist_results/5-9A_finbert_sentiment_prepared.md)
  - 5-9A 결과: [FinBERT sentiment extraction](checklist_results/5-9A_finbert_sentiment_results.md)
  - 5-9B 결과: [FinBERT sentiment output audit](checklist_results/5-9B_finbert_sentiment_audit_results.md)
  - 5-9C 결과: [FinBERT sentiment context features](checklist_results/5-9C_finbert_context_features_results.md)
    - `2,399`개 sample에 대해 FinBERT context feature `91`개 생성.
    - 7/20/60일 window missing rate는 모두 `0.0000`, count match rate는
      모두 `1.0000`.
    - test 구간 sentiment가 train/validation보다 뚜렷하게 positive라서,
      이게 유용한 regime signal인지 단순 분포 이동인지는 `5-9D` 모델
      실험에서 확인해야 함.
  - 5-9D 준비 완료: [FinBERT-only bounded FiLM runner](checklist_results/5-9D_finbert_only_bounded_film_prepared.md)
    - 진단용 컬럼 제외 후 context dimension: `79`.
    - Kaggle one-cell:
      `notebooks/kaggle_stage5_9d_finbert_film_ablation_one_cell.md`.
    - compact upload bundle:
      `stage5_llm_news_embedding_5_9d_finbert_context_bundle.zip`.
  - 5-9D 결과: [FinBERT-only bounded FiLM](checklist_results/5-9D_finbert_only_bounded_film_results.md)
    - 평균 accuracy `0.578487`, ROC-AUC `0.586072`, AP `0.611943`.
    - Stage2 `ohlc_ma_vb` accuracy `0.579320`, N8B F&G-only accuracy
      `0.580291`은 넘지 못했지만 ROC-AUC/AP/Brier는 아주 작게 개선.
    - 핵심 문제: test 구간의 FinBERT sentiment/news-count feature가 positive
      방향으로 shift되어 Up-bias를 만듦. 평균 predicted positive rate는
      `0.637196`, 실제 positive rate는 `0.541291`.
  - 5-9E 준비 완료: [FinBERT + F&G bounded FiLM runner](checklist_results/5-9E_finbert_fg_bounded_film_prepared.md)
    - 로컬 검증 context dimension: `83` = FinBERT numeric feature `79`개
      + F&G raw regime feature `4`개.
    - Kaggle one-cell:
      `notebooks/kaggle_stage5_9e_finbert_fg_film_ablation_one_cell.md`.
  - 5-9E 결과: [FinBERT + F&G bounded FiLM](checklist_results/5-9E_finbert_fg_bounded_film_results.md)
    - 평균 accuracy `0.580569`, ROC-AUC `0.585843`, AP `0.611899`,
      Brier `0.272701`.
    - Stage2 `ohlc_ma_vb` 대비 accuracy `+0.001249`, ROC-AUC `+0.000981`,
      Brier `-0.001636`.
    - N8B F&G-only 대비 accuracy `+0.000278`, ROC-AUC `+0.000913`,
      Brier `-0.001304`.
    - 해석: small positive/near-tie result. 강한 성능 개선이라고 주장하기는
      어렵지만, headline-level sentiment를 F&G와 결합해도 frozen Stage2 +
      bounded FiLM 구조는 무너지지 않고 아주 작은 개선을 보였다.
  - 5-9F 비교: [Stage2 vs F&G vs FinBERT vs FinBERT + F&G](checklist_results/5-9F_stage5_finbert_comparison.md)
    - 결정: 첫 FinBERT branch는 닫는다. 다음은 correction/regression 분석,
      또는 더 풍부한 해석형 뉴스 feature가 필요할 때만 prompt/event label로
      넘어간다.

- [ ] 5-10. Prompt/event auxiliary features
  - FinBERT sentiment가 부족하거나 더 풍부한 해석 layer가 필요할 때 실행.
  - GPT/Claude 고정 prompt:
    BTC relevance, direction, event type, impact horizon, risk regime,
    confidence, short reason.
  - Raw output을 cache한 뒤 fixed rule로 numeric feature 변환.
  - 자유로운 예측기가 아니라 structured feature extraction으로 사용.

## 해석 및 보고

- [x] 5-11. Correction/regression analysis
  - Stage2 baseline과 Stage5 candidate 비교.
  - 필수 bucket:
    Stage2 wrong -> Stage5 correct,
    Stage2 correct -> Stage5 wrong,
    high-news-count,
    high prompt-confidence,
    positive/negative FinBERT regime,
    positive/negative/unknown prompt regime.
  - 5-11A 부분 결과: [Stage2 vs FinBERT + F&G correction counts](checklist_results/5-11A_finbert_fg_correction_counts.md)
    - 전체 matched decision: `7,205`.
    - Corrections: `95`.
    - Regressions: `86`.
    - Net corrections: `+9`.
    - Changed predictions: `181`, 즉 `2.51%`.
    - 해석: 작지만 실제 net correction이 있으며, bounded FiLM은 Stage2
      decision 중 일부만 보수적으로 바꾸는 방식으로 작동했다.
  - 5-11 전체 조건 분석 결과: [FinBERT + F&G conditional correction analysis](checklist_results/5-11_finbert_fg_condition_analysis.md)
    - Stage2 uncertain 45-55 bucket: delta accuracy `+0.012484`.
    - F&G greed bucket: delta accuracy `+0.010849`.
    - low 60-day news-count와 F&G neutral bucket은 negative.
    - 5-12 입력 sample:
      `reports/tables/stage5_5_11_finbert_fg_condition_analysis_selected_samples_for_5_12.csv`.

- [ ] 5-12. Grad-CAM and FiLM modulation export
  - 선택된 correction/regression sample에 대해 export:
    Stage2 Grad-CAM,
    Stage5 Grad-CAM,
    gamma/beta summary,
    news window text,
    embedding/FinBERT/prompt features,
    probability change.

- [ ] 5-13. Final Stage 5 report and thesis-title decision
  - 최종 논문에서 LLM/news context contribution을 주장할 수 있는지 결정.
  - Claim은 분리해서 작성:
    overall accuracy,
    conditional improvement,
    interpretability,
    limitations.
