# 5-5 Headline-Level Embedding Table

Status: prepared; actual API run pending `OPENAI_API_KEY`.

## Prepared

Implemented:

```text
scripts/build_stage5_news_embeddings.py
```

The script builds resumable OpenAI headline embeddings for:

```text
openai_text_embedding_3_small_primary
```

It stores each successful batch before assembling the final matrix, so Kaggle
or local interruption can resume without re-embedding completed batches.

## Dry-Run Result

- Provider: `openai`
- Model: `text-embedding-3-small`
- Requested dimensions: `1536`
- Input template: `headline_only_v1`
- Items: `24,281`
- Batch size: `128`
- Planned batches: `190`
- Artifact root: `outputs/stage5/embeddings/openai_text_embedding_3_small`
- Cache manifest: `data_inventory/openai_text_embedding_3_small_cache_manifest.json`

## Actual Run Command

Set the API key in the shell or Kaggle secret environment, then run:

```bash
python3 stage5_llm_news_embedding/scripts/build_stage5_news_embeddings.py \
  --run-id openai_text_embedding_3_small_primary
```

Optional smoke run:

```bash
python3 stage5_llm_news_embedding/scripts/build_stage5_news_embeddings.py \
  --run-id openai_text_embedding_3_small_primary \
  --limit 256
```

## Expected Outputs After Actual Run

- `outputs/stage5/embeddings/openai_text_embedding_3_small/embedding_vectors.npy`
- `outputs/stage5/embeddings/openai_text_embedding_3_small/embedding_item_manifest.csv`
- `outputs/stage5/embeddings/openai_text_embedding_3_small/embedding_failures.csv`
- `data_inventory/openai_text_embedding_3_small_cache_manifest.json`
- `reports/tables/openai_text_embedding_3_small_primary_cache_manifest.json`
- `reports/tables/stage5_openai_text_embedding_3_small_primary_embedding_report.md`

## Current Blocker

The current local shell does not have `OPENAI_API_KEY`, so the paid/API
embedding call was not executed in this run.
