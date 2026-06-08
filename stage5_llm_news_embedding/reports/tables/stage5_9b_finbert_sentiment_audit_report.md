# 5-9B FinBERT Sentiment Output Audit

Status: `ok`.

## Purpose

This audit checks whether the 5-9A FinBERT headline labels are usable for
7/20/60-day news sentiment aggregation. It does not evaluate market prediction
performance.

## Label Distribution

- negative: `6307` (`0.2598`)
- neutral: `11231` (`0.4625`)
- positive: `6743` (`0.2777`)

## Confidence And Quality

- Mean confidence: `0.813328`
- Median confidence: `0.865923`
- Low-confidence rows `< 0.50`: `771`
- Low-confidence share `< 0.50`: `0.0318`
- Mean sentiment score: `0.030385`
- Sentiment std: `0.572671`

Quality warnings:

- none

## Year-Level Pattern

- 2018: sentiment_mean `-0.1092`, positive `0.1853`, negative `0.3182`
- 2019: sentiment_mean `-0.0282`, positive `0.2004`, negative `0.2501`
- 2020: sentiment_mean `0.0514`, positive `0.2771`, negative `0.2303`
- 2021: sentiment_mean `0.0719`, positive `0.2900`, negative `0.2295`
- 2022: sentiment_mean `-0.0302`, positive `0.2757`, negative `0.3313`
- 2023: sentiment_mean `0.0855`, positive `0.3430`, negative `0.2571`
- 2024: sentiment_mean `0.1024`, positive `0.3216`, negative `0.2205`

## Interpretation

The output is suitable for the next aggregation step if label distribution is
not collapsed, confidence is generally high, and low-confidence samples are
plausibly ambiguous. The generated audit sample table should be reviewed before
treating the feature as final.

## Outputs

- Audit samples: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_9b_finbert_sentiment_audit_audit_samples.csv`
- Label distribution: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_9b_finbert_sentiment_audit_label_distribution.csv`
- Year summary: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_9b_finbert_sentiment_audit_year_summary.csv`
- Daily summary: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_9b_finbert_sentiment_audit_daily_summary.csv`
- Quantiles: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_9b_finbert_sentiment_audit_quantiles.csv`
- Manifest: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/data_inventory/stage5_9b_finbert_sentiment_audit_manifest.json`
