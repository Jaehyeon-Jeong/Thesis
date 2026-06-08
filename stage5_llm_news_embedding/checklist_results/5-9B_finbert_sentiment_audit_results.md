# 5-9B Result: FinBERT Sentiment Output Audit

## Status

Completed successfully.

## Inputs

- Source artifact: `5-9A` FinBERT sentiment item CSV
- Rows: `24,281`
- Audit sample size per bucket: `60`
- Buckets:
  - random
  - high positive
  - high negative
  - high neutral
  - low confidence
  - ambiguous score or confidence

## Distribution Check

| label | rows | share |
|---|---:|---:|
| negative | 6,307 | 0.2598 |
| neutral | 11,231 | 0.4625 |
| positive | 6,743 | 0.2777 |

Quality summary:

- Mean confidence: `0.813328`
- Median confidence: `0.865923`
- Low-confidence rows `< 0.50`: `771`
- Low-confidence share `< 0.50`: `0.0318`
- Mean sentiment score: `0.030385`
- Sentiment score std: `0.572671`
- Quality warnings: none

## Year-Level Pattern

- `2018`: sentiment mean `-0.1092`, negative share `0.3182`
- `2022`: sentiment mean `-0.0302`, negative share `0.3313`
- `2023`: sentiment mean `0.0855`, positive share `0.3430`
- `2024`: sentiment mean `0.1024`, positive share `0.3216`

This year-level pattern is directionally plausible for BTC market regimes.

## Sample Audit Findings

High-positive examples are mainly revenue increases, mining/security
improvements, hashrate increases, and outperformance headlines.

High-negative examples are mainly revenue drops, futures volume declines,
mining profitability declines, and BTC price declines.

Low-confidence examples are genuinely mixed or ambiguous, such as headlines
combining BTC price strength with macro uncertainty. This supports using
`finbert_confidence` and low-confidence flags in later window aggregation.

## Outputs

- `reports/tables/stage5_9b_finbert_sentiment_audit_audit_samples.csv`
- `reports/tables/stage5_9b_finbert_sentiment_audit_label_distribution.csv`
- `reports/tables/stage5_9b_finbert_sentiment_audit_year_summary.csv`
- `reports/tables/stage5_9b_finbert_sentiment_audit_daily_summary.csv`
- `reports/tables/stage5_9b_finbert_sentiment_audit_quantiles.csv`
- `reports/tables/stage5_9b_finbert_sentiment_audit_report.md`

## Decision

Proceed to `5-9C`: aggregate FinBERT item-level sentiment into strict `t-1`
7/20/60-day news sentiment context features.
