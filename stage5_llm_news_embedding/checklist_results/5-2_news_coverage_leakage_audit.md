# 5-2 News Coverage and Leakage Audit

Status: completed.

## Decision

Stage 5 can proceed with the existing `edaschau/bitcoin_news` source.

The Stage 5 leakage policy is fixed as:

```text
news_date <= image_end_date - 1 calendar day
```

Same-day news is excluded because the dataset is date-based and the exact
publication cutoff relative to the BTC daily candle close is not defended.

## Coverage Result

- News source date range: `2011-06-22` to `2025-06-03`.
- Stage sample date range: `2018-04-29` to `2024-12-11`.
- Stage samples: `2,399` total.
- Train/validation/test: `671 / 287 / 1,441`.
- 7-day trailing window coverage: `1.000`.
- 20-day trailing window coverage: `1.000`.
- 60-day trailing window coverage: `1.000`.
- Single prior-day `t-1` coverage: `0.960`.
- Single prior-day missing samples: `95`.

The single-day `t-1` gaps do not block Stage 5 because the actual embedding
context uses 7/20/60-day windows, and all of those windows have full sample
coverage.

## Embedding Design Implication

Whole-window headline text can be long:

- Mean 7-day headline text length: `5,405` characters.
- Mean 20-day headline text length: `15,415` characters.
- Mean 60-day headline text length: `45,905` characters.
- Test split mean 60-day headline text length: `55,282` characters.

Therefore Stage 5 should not start by embedding the whole 60-day window as one
large text. The first design remains:

```text
news item -> embedding -> 7/20/60-day aggregation
```

Whole-window embedding can remain a later ablation only.

## Generated Outputs

- [coverage by split](../reports/tables/stage5_news_coverage_by_split.csv)
- [headline window summary](../reports/tables/stage5_news_headline_window_summary.csv)
- [missing t-1 dates](../reports/tables/stage5_news_missing_t_minus_1_dates.csv)
- [alignment sample counts](../reports/tables/stage5_news_alignment_sample_counts.csv)
- [leakage policy](../reports/tables/stage5_news_leakage_policy.json)
- [source inventory](../data_inventory/stage5_news_source_inventory.json)
- [audit report](../reports/tables/stage5_news_coverage_leakage_audit_report.md)

## Next Step

Proceed to `5-3`: deterministic embedding input construction.

