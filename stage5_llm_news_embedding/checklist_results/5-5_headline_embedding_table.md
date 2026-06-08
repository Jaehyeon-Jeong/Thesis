# 5-5 Headline-Level Embedding Table

Status: completed.

## Backend

- Provider: `openai`
- Model: `text-embedding-3-small`
- Requested dimensions: `1536`
- Input template: `headline_only_v1`

## Result

- Items requested: `24,281`
- Items completed: `24,281`
- Vector shape: `(24281, 1536)`
- Vector dtype: `float32`
- Failed rows: `0`
- Duplicate input hashes: `0`
- API prompt tokens: `390,148`
- API total tokens: `390,148`

## Local Result Source

Downloaded/expanded result folder:

```text
/Users/jaehyeonjeong/Desktop/논문/5_openai_embed_small
```

The generated embedding artifacts were also overlaid into:

```text
stage5_llm_news_embedding/outputs/stage5/embeddings/openai_text_embedding_3_small
```

## Outputs

- Embedding matrix:
  `outputs/stage5/embeddings/openai_text_embedding_3_small/embedding_vectors.npy`
- Item manifest:
  `outputs/stage5/embeddings/openai_text_embedding_3_small/embedding_item_manifest.csv`
- Failure log:
  `outputs/stage5/embeddings/openai_text_embedding_3_small/embedding_failures.csv`
- Cache manifest:
  `data_inventory/openai_text_embedding_3_small_cache_manifest.json`
- Report:
  `reports/tables/stage5_openai_text_embedding_3_small_primary_embedding_report.md`

## Next

Proceed to `5-6`: aggregate headline-level embeddings into 7/20/60-day
sample-level context windows.
