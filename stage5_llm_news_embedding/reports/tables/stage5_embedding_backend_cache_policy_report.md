# 5-4 Embedding Backend And Cache Policy

Status: completed.

## Locked Execution Order

1. `text-embedding-3-small` via OpenAI.
2. `voyage-4` via Voyage AI as the Anthropic-side comparison.
3. `text-embedding-3-large` as optional OpenAI upper-bound comparison.
4. `voyage-4-large` as optional stronger Voyage comparison.

Claude itself is not used for embeddings because Anthropic documentation states
that Anthropic does not offer a native embedding model. Voyage AI is used as
the Claude/Anthropic-side embedding comparison.

## Input Scale

- Embedding items: `24,281`
- Input template: `headline_only_v1`
- Total input characters: `1,721,517`
- Rough token estimate: `430,380`
- Max input characters: `279`

## Cache Policy

- API keys are read only from environment variables.
- API keys are never written to configs, CSVs, JSON manifests, logs, or reports.
- Every embedding row is keyed by `input_hash + provider + model + requested_dimensions`.
- Reruns must skip already cached matching rows.
- Failed items must be written to a failure log and retried separately.
- GitHub tracks only small manifests/reports; embedding matrices remain in local/Kaggle artifacts.

## Required Cache Manifest Fields

```text
provider
model
requested_dimensions
input_manifest_path
input_manifest_sha256
input_template
input_hash_column
item_count
created_utc
embedding_array_path
item_manifest_path
failure_log_path
```

## Next Step

Proceed to `5-5`: generate the headline-level embedding table for
`text-embedding-3-small`.
