# 5-9C Result: FinBERT Sentiment Context Features

## Status

Completed successfully.

## Purpose

Convert headline-level FinBERT sentiment from `5-9A` into leakage-safe
sample-level context features for Stage2 frozen + bounded FiLM experiments.

For each Stage4 sample ending at date `t`, news is included only when:

```text
t-window <= news_date <= t-1
```

Same-day news is excluded.

## Inputs

- FinBERT headline rows: `24,281`
- Stage4 samples: `2,399`
- Windows: `7`, `20`, `60` days
- Sample table: Stage4 deduplicated headline-window parquet

## Outputs

- Context features:
  `outputs/stage5/finbert_context/finbert_prosusai_headline_v1/stage5_finbert_context_features.csv`
- Scaler:
  `outputs/stage5/finbert_context/finbert_prosusai_headline_v1/stage5_finbert_context_scaler.json`
- Split summary:
  `reports/tables/stage5_9c_finbert_sentiment_context_split_summary.csv`
- Feature summary:
  `reports/tables/stage5_9c_finbert_sentiment_context_feature_summary.csv`
- Report:
  `reports/tables/stage5_9c_finbert_sentiment_context_report.md`

## Coverage Check

| window | missing rate | mean count | count match |
|---:|---:|---:|---:|
| 7d | 0.0000 | 69.38 | 1.0000 |
| 20d | 0.0000 | 197.81 | 1.0000 |
| 60d | 0.0000 | 589.57 | 1.0000 |

The FinBERT aggregation exactly matches the Stage4 deduplicated headline-window
counts for all three windows.

## Sentiment Pattern

| split | sentiment 7d | sentiment 20d | sentiment 60d | news-FG proxy 60d |
|---|---:|---:|---:|---:|
| train | -0.0318 | -0.0305 | -0.0344 | 50.16 |
| validation | -0.0359 | -0.0354 | -0.0352 | 49.92 |
| test | 0.0594 | 0.0575 | 0.0556 | 79.50 |

The test period is clearly more positive than train/validation. This can be a
real regime signal, but it can also behave like a distribution shift, so model
performance must be verified in `5-9D` rather than assumed.

## Decision

Proceed to `5-9D`: Stage2 frozen + FinBERT-only bounded FiLM. Use the exported
context feature table as the source for Stage4 prebuilt context artifacts.
