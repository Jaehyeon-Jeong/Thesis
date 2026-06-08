# 5-9A Result: FinBERT Headline Sentiment Extraction

## Status

Completed successfully.

## Input

- Input rows: `24,281` deduplicated headline items
- News date range: `2018-02-28` to `2024-12-10`
- Model: `ProsusAI/finbert`
- Device: `cuda`
- Text column: `embedding_input_text`

## Label Distribution

| label | rows | share |
|---|---:|---:|
| neutral | 11,231 | 0.4625 |
| positive | 6,743 | 0.2777 |
| negative | 6,307 | 0.2598 |

Summary statistics:

- Mean positive probability: `0.291660`
- Mean negative probability: `0.261275`
- Mean neutral probability: `0.447065`
- Mean sentiment score (`positive - negative`): `0.030385`
- Sentiment score std: `0.572670`
- Mean confidence: `0.813328`
- Low-confidence rows `< 0.50`: `771` (`3.17%`)

## Year-Level Pattern

The year-level sentiment pattern is plausible:

- `2018`: negative-leaning, mean sentiment `-0.1092`
- `2022`: negative-leaning, mean sentiment `-0.0302`
- `2023`: positive-leaning, mean sentiment `0.0855`
- `2024`: positive-leaning, mean sentiment `0.1024`

This suggests FinBERT is not producing a degenerate label distribution.

## Sample Audit

High-confidence positive examples are mostly revenue, mining, adoption, or
outperformance headlines. High-confidence negative examples are mostly revenue
drops, futures volume drops, mining profitability drops, or BTC price declines.

Low-confidence examples are genuinely ambiguous, such as headlines combining
positive price movement with macro uncertainty or mixed analyst language. This
supports keeping `finbert_confidence` and low-confidence flags in later feature
aggregation.

## Decision

Proceed to `5-9B/5-9C`:

1. Build sampled audit tables for positive/negative/neutral/low-confidence
   examples.
2. Aggregate FinBERT item-level sentiment into strict `t-1` 7/20/60-day
   windows.
3. Use the resulting news-only sentiment regime features in Stage2-frozen
   bounded FiLM.

This is a stronger next step than continuing generic embedding scale grids,
because FinBERT outputs an interpretable financial-tone signal rather than an
opaque semantic vector.
