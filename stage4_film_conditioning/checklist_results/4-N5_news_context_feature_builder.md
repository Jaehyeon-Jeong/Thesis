# 4-N5 News Context Feature Builder

Status: completed

## Goal

Convert the 4-N4 headline TF-IDF/SVD table into the normalized Stage 4 context
artifact format used by later model experiments.

## Input

- Source table:
  `outputs/stage4/news/stage4_news_tfidf_svd_i60_r20/news_tfidf_svd_features.parquet`
- Fixed sample universe:
  `I60/R20/ohlc_ma_vb`
- Alignment policy:
  strict `t-1`; same-day news remains excluded.

## Context Vector

The first news-context vector has `102` dimensions:

- `96` headline TF-IDF/SVD features:
  `news_svd_7d`, `news_svd_20d`, `news_svd_60d`
  with `32` components per window.
- `6` log-count features:
  `news_count_log1p_7d/20d/60d` and
  `unique_source_count_log1p_7d/20d/60d`.

Raw news counts, unique-source counts, headline hashes, and SVD norms are kept
as metadata columns, but the model input uses the normalized feature columns
listed in `context_scaler.json`.

## Normalization

- Fit split: train only.
- Imputation: train median.
- Clipping: train `0.01` and `0.99` quantiles.
- Scaling: train z-score.
- Validation/test use the train-fitted scaler without refitting.

## Result

- Rows: `2,399`
- Split counts:
  - train: `671`
  - validation: `287`
  - test: `1,441`
- Missing warnings: none.
- Context artifact:
  `outputs/stage4/context/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60/seed_42/`

## Report Tables

- [audit](../reports/tables/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60_seed42_news_context_feature_audit.json)
- [summary](../reports/tables/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60_seed42_news_context_feature_summary.csv)
- [manifest](../reports/tables/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60_seed42_news_context_manifest.json)

## Next Step

Proceed to 4-N6:

1. Run visual-only control on the same news-aligned sample universe.
2. Run `CNN + news concat` five-seed.
3. Only after that compare `CNN + news bounded last-block FiLM`.
