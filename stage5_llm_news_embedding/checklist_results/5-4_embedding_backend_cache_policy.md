# 5-4 Embedding Backend And Cache Policy

Status: completed.

## Decision

Stage 5 embedding execution order is fixed as:

1. OpenAI `text-embedding-3-small`
2. Voyage AI `voyage-4`
3. Optional: OpenAI `text-embedding-3-large`
4. Optional: Voyage AI `voyage-4-large`

Claude is not used directly for embeddings because Anthropic documentation
states that Anthropic does not provide a native embedding model. Voyage AI is
therefore used as the Anthropic-side embedding comparison.

## Input Scale

- Embedding items: `24,281`
- Input template: `headline_only_v1`
- Total input characters: `1,721,517`
- Rough token estimate: `430,380`
- Max input characters: `279`

## Cost/Size Planning

Current rough OpenAI estimates, checked on `2026-06-08`:

- `text-embedding-3-small`: about `$0.0086`
- `text-embedding-3-large`: about `$0.0559`

Voyage pricing must be verified immediately before execution.

## Cache Policy

Cache key:

```text
embedding_input_hash + provider + model + requested_dimensions
```

Rules:

- API keys are environment variables only.
- API keys are never written to configs, CSVs, JSON manifests, logs, or reports.
- Reruns skip already cached matching rows.
- Failed items are written to a failure log and retried separately.
- GitHub tracks only small manifests/reports.
- Embedding matrices remain in local/Kaggle artifacts.

## Outputs

- Backend plan: `reports/tables/stage5_embedding_backend_plan.csv`
- Local fallbacks: `reports/tables/stage5_embedding_local_fallbacks.csv`
- Cache policy: `data_inventory/stage5_embedding_cache_policy.json`
- Report: `reports/tables/stage5_embedding_backend_cache_policy_report.md`
- Config: `configs/stage5_embedding_backends.yaml`

## Next

Proceed to `5-5`: generate the headline-level embedding table for
OpenAI `text-embedding-3-small`.
