# 5-6 Trailing-Window Embedding Context

Status: completed.

## Inputs

- Embedding model: `text-embedding-3-small`
- Embedding matrix shape: `[24281, 1536]`
- Sample count: `2399`
- Windows: `[7, 20, 60]`

## Aggregation

For each sample ending at date `t`, the script uses news dates:

```text
t-window <= news_date <= t-1
```

Same-day news is excluded. Two vector aggregations are exported:

- simple mean
- exponential time-decay mean with half-life `window / 2`

## Coverage

- 7d missing rate: `0.0000`
- 20d missing rate: `0.0000`
- 60d missing rate: `0.0000`
- 7d count match rate vs Stage4 audit: `1.0000`
- 20d count match rate vs Stage4 audit: `1.0000`
- 60d count match rate vs Stage4 audit: `1.0000`

## Outputs

- Metadata: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/outputs/stage5/embedding_context/openai_text_embedding_3_small/stage5_news_embedding_context_metadata.csv`
- Mean vectors: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/outputs/stage5/embedding_context/openai_text_embedding_3_small/stage5_news_embedding_window_mean_vectors.npy`
- Time-decay mean vectors: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/outputs/stage5/embedding_context/openai_text_embedding_3_small/stage5_news_embedding_window_decay_mean_vectors.npy`
- Summary: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_news_embedding_context_summary.csv`
- Manifest: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/data_inventory/stage5_news_embedding_context_manifest.json`

## Next

Proceed to `5-7`: train-only SVD/PCA dimension grid over the 7/20/60
embedding context vectors.
