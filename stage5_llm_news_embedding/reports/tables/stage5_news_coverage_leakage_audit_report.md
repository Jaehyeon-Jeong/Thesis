# 5-2 News Coverage and Leakage Audit

Status: completed.

## Decision

Stage 5 can proceed with the existing Bitcoin news source under a strict
`t-1` leakage policy.

## Dataset

- Source: `edaschau/bitcoin_news`
- Rows: `30,626`
- News date range: `2011-06-22` to `2025-06-03`
- Stage sample date range: `2018-04-29` to `2024-12-11`
- Title coverage: `1.000`
- Article-text coverage: `1.000`

## Leakage Policy

For every sample ending at image date `t`, Stage 5 may use only:

```text
news_date <= t - 1 calendar day
```

Same-day news and all later news are excluded. This matches the Stage 4 news
alignment policy and avoids relying on undefended intraday publication-time
assumptions.

## Coverage Summary

- Total samples: `2,399`
- Train/validation/test split counts: `671 / 287 / 1,441`
- 7-day trailing window coverage: `1.000`
- 20-day trailing window coverage: `1.000`
- 60-day trailing window coverage: `1.000`
- Single prior-day `t-1` coverage: `0.960`
- Single prior-day missing samples: `95`
- Test 7/20/60-day coverage: `1.000` / `1.000` / `1.000`

## Implication for Embeddings

The 7/20/60-day windows have full sample coverage, so Stage 5 can build
embedding context for every Stage 2/Stage 4 sample.

However, whole-window headline text can be long:

- Mean 7-day headline chars: `5405.2`
- Mean 20-day headline chars: `15414.8`
- Mean 60-day headline chars: `45904.6`

Therefore the first Stage 5 embedding design should use news-item-level
embeddings followed by window aggregation. Whole-window concatenated embeddings
should be kept as a later ablation only.

## Required Stage 5 Inputs

- `stage5_news_coverage_by_split.csv`
- `stage5_news_headline_window_summary.csv`
- `stage5_news_alignment_sample_counts.csv`
- `stage5_news_leakage_policy.json`

## Next Step

Proceed to `5-3`: deterministic embedding input construction.
