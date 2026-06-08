# 5-6 Trailing-Window Embedding Context

Status: completed.

## Inputs

- Embedding model: `text-embedding-3-small`
- Headline embedding matrix: `(24281, 1536)`
- Sample table: Stage 4 deduplicated headline-window table
- Sample count: `2,399`
- Windows: `7`, `20`, `60`

## Aggregation Rule

For a sample ending at image date `t`, use only:

```text
t-window <= news_date <= t-1
```

Same-day news is excluded.

Two vector aggregations are exported:

- simple arithmetic mean
- exponential time-decay mean with half-life `window / 2`

## Verification

- 7d missing rate: `0.0000`
- 20d missing rate: `0.0000`
- 60d missing rate: `0.0000`
- 7d count match rate vs Stage4 deduped headline windows: `1.0000`
- 20d count match rate vs Stage4 deduped headline windows: `1.0000`
- 60d count match rate vs Stage4 deduped headline windows: `1.0000`

## Outputs

- Metadata:
  `outputs/stage5/embedding_context/openai_text_embedding_3_small/stage5_news_embedding_context_metadata.csv`
- Mean vectors:
  `outputs/stage5/embedding_context/openai_text_embedding_3_small/stage5_news_embedding_window_mean_vectors.npy`
- Time-decay mean vectors:
  `outputs/stage5/embedding_context/openai_text_embedding_3_small/stage5_news_embedding_window_decay_mean_vectors.npy`
- Summary:
  `reports/tables/stage5_news_embedding_context_summary.csv`
- Manifest:
  `data_inventory/stage5_news_embedding_context_manifest.json`

## Next

Proceed to `5-7`: train-only SVD/PCA dimension grid over the 7/20/60
embedding context vectors.
