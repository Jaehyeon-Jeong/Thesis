# 5-5 Headline-Level Embedding Table

Status: ok.

## Backend

- Provider: `openai`
- Model: `text-embedding-3-small`
- Requested dimensions: `1536`
- Input template: `headline_only_v1`

## Rows

- Items requested: `24,281`
- Items completed: `24,281`
- Vector shape: `[24281, 1536]`

## Outputs

- Embedding matrix: `/kaggle/working/stage5_llm_news_embedding/outputs/stage5/embeddings/openai_text_embedding_3_small/embedding_vectors.npy`
- Item manifest: `/kaggle/working/stage5_llm_news_embedding/outputs/stage5/embeddings/openai_text_embedding_3_small/embedding_item_manifest.csv`
- Cache manifest: `/kaggle/working/stage5_llm_news_embedding/data_inventory/openai_text_embedding_3_small_cache_manifest.json`
- Failure log: `/kaggle/working/stage5_llm_news_embedding/outputs/stage5/embeddings/openai_text_embedding_3_small/embedding_failures.csv`

## Notes

Embedding matrices are generated artifacts and should not be committed to
GitHub. Keep them in local/Kaggle output bundles.
