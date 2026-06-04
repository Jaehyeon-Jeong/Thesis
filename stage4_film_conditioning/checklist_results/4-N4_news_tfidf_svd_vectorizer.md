# 4-N4. Train-Only TF-IDF/SVD News Vectorizer

## English

Status: complete.

Purpose:
- Convert the 4-N3 headline-window documents into fixed-size numeric news
  vectors.
- Keep the first news track headline-only, non-LLM, and leakage-safe.
- Fit all text preprocessing only on train split documents.

Command:

```bash
python -u scripts/build_stage4_news_tfidf_svd.py \
  --config configs/env_local.yaml \
  --window-days 7 20 60 \
  --svd-dim 32 \
  --output-prefix stage4_news_tfidf_svd
```

Fit policy:
- TF-IDF vocabulary, IDF weights, and SVD are fit only on train split headline
  windows.
- Fit documents: `2,013` = `671` train samples x `3` windows.
- Fit sample date range: `2018-04-29` to `2020-12-10`.
- Validation/test are transform-only.

Vectorizer:
- `TfidfVectorizer`
- `ngram_range=(1, 2)`
- `stop_words="english"`
- `min_df=2`
- `max_df=0.95`
- `max_features=10,000`
- vocabulary size: `10,000`

SVD:
- requested components: `32`
- actual components: `32`
- explained variance ratio sum: `0.5856`

Output vectors:

```text
news_svd_7d_00 ... news_svd_7d_31
news_svd_20d_00 ... news_svd_20d_31
news_svd_60d_00 ... news_svd_60d_31
news_count_7d / 20d / 60d
unique_source_count_7d / 20d / 60d
log1p count variants
```

Generated full artifacts:
- `outputs/stage4/news/stage4_news_tfidf_svd_i60_r20/news_tfidf_svd_features.parquet`
- `outputs/stage4/news/stage4_news_tfidf_svd_i60_r20/artifacts/tfidf_vectorizer.joblib`
- `outputs/stage4/news/stage4_news_tfidf_svd_i60_r20/artifacts/truncated_svd.joblib`

Report tables:
- `reports/tables/stage4_news_tfidf_svd_manifest.json`
- `reports/tables/stage4_news_tfidf_svd_summary.csv`
- `reports/tables/stage4_news_tfidf_svd_feature_summary.csv`
- `reports/tables/stage4_news_tfidf_svd_top_terms.csv`
- `reports/tables/stage4_news_tfidf_svd_examples.csv`

Result summary:
- Output sample rows: `2,399`.
- Output feature table columns: `120`.
- All 7/20/60 windows have `100.00%` coverage by construction from 4-N3.
- Top SVD terms show recognizable BTC/crypto news themes such as price,
  Bitcoin Cash, market wrap, blockchain bites, DeFi, halving, Libra, and
  MicroStrategy.

Decision:
- Proceed to `4-N5`.
- `4-N5` should normalize count/log-count features with train-only statistics
  and expose the final sample-level news context vector for model runners.

## 한국어

상태: 완료.

목적:
- 4-N3 headline-window document를 고정 길이 numeric news vector로 변환합니다.
- 첫 news track은 headline-only, non-LLM, leakage-safe로 유지합니다.
- 모든 text preprocessing은 train split document에만 fit합니다.

실행 command:

```bash
python -u scripts/build_stage4_news_tfidf_svd.py \
  --config configs/env_local.yaml \
  --window-days 7 20 60 \
  --svd-dim 32 \
  --output-prefix stage4_news_tfidf_svd
```

Fit policy:
- TF-IDF vocabulary, IDF weight, SVD는 train split headline window에만 fit합니다.
- Fit document 수: `2,013` = train sample `671`개 x window `3`개.
- Fit sample date range: `2018-04-29` to `2020-12-10`.
- Validation/test는 transform-only입니다.

Vectorizer:
- `TfidfVectorizer`
- `ngram_range=(1, 2)`
- `stop_words="english"`
- `min_df=2`
- `max_df=0.95`
- `max_features=10,000`
- vocabulary size: `10,000`

SVD:
- requested components: `32`
- actual components: `32`
- explained variance ratio sum: `0.5856`

생성 vector:

```text
news_svd_7d_00 ... news_svd_7d_31
news_svd_20d_00 ... news_svd_20d_31
news_svd_60d_00 ... news_svd_60d_31
news_count_7d / 20d / 60d
unique_source_count_7d / 20d / 60d
log1p count variants
```

생성한 full artifact:
- `outputs/stage4/news/stage4_news_tfidf_svd_i60_r20/news_tfidf_svd_features.parquet`
- `outputs/stage4/news/stage4_news_tfidf_svd_i60_r20/artifacts/tfidf_vectorizer.joblib`
- `outputs/stage4/news/stage4_news_tfidf_svd_i60_r20/artifacts/truncated_svd.joblib`

Report table:
- `reports/tables/stage4_news_tfidf_svd_manifest.json`
- `reports/tables/stage4_news_tfidf_svd_summary.csv`
- `reports/tables/stage4_news_tfidf_svd_feature_summary.csv`
- `reports/tables/stage4_news_tfidf_svd_top_terms.csv`
- `reports/tables/stage4_news_tfidf_svd_examples.csv`

결과 요약:
- Output sample rows: `2,399`.
- Output feature table columns: `120`.
- 7/20/60 window coverage는 4-N3에서 이미 `100.00%`로 고정됐습니다.
- SVD top term은 price, Bitcoin Cash, market wrap, blockchain bites, DeFi,
  halving, Libra, MicroStrategy 등 BTC/crypto news theme를 포착합니다.

판단:
- `4-N5`로 진행합니다.
- `4-N5`에서는 count/log-count feature를 train-only statistics로 normalize하고,
  model runner가 사용할 최종 sample-level news context vector를 노출합니다.
