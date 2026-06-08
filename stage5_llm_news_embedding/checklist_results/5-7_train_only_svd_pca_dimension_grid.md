# 5-7 Train-Only SVD/PCA Dimension Grid

Status: completed.

## Input

- Source context: `outputs/stage5/embedding_context/openai_text_embedding_3_small`
- Rows: `2,399`
- Train rows used to fit reducers: `671`
- Windows: `7`, `20`, `60`
- Aggregations tested:
  - `mean`
  - `decay_mean`
- Dimensions tested: `8`, `16`, `32`

## Protocol

Reducers are fit on the train split only. Validation and test rows are
transformed using the train-fitted mean vector and SVD/PCA components.

The raw `1536`-dimensional embedding vectors are not fed directly into Stage 4.

## Result Summary

Mean explained variance across 7/20/60 windows:

| aggregation | dim 8 | dim 16 | dim 32 |
|---|---:|---:|---:|
| mean | `0.6182` | `0.7483` | `0.8520` |
| decay_mean | `0.5924` | `0.7214` | `0.8283` |

Window-level detail:

- `mean/7d`: dim32 explains `0.6873`
- `mean/20d`: dim32 explains `0.8837`
- `mean/60d`: dim32 explains `0.9850`
- `decay_mean/7d`: dim32 explains `0.6607`
- `decay_mean/20d`: dim32 explains `0.8524`
- `decay_mean/60d`: dim32 explains `0.9716`

## Practical Reading

`mean/dim16` is the conservative first candidate because it keeps the context
feature count smaller while preserving about `75%` of the average window
variance.

`mean/dim32` is the stronger representation candidate because it preserves about
`85%` of the average window variance, but it adds more context features and may
overfit with only `671` train samples.

## Outputs

- Summary:
  `reports/tables/stage5_news_embedding_svd_grid_summary.csv`
- Combined summary:
  `reports/tables/stage5_news_embedding_svd_grid_combined_summary.csv`
- Report:
  `reports/tables/stage5_news_embedding_svd_grid_report.md`
- Manifest:
  `data_inventory/stage5_news_embedding_svd_manifest.json`
- Feature tables:
  `outputs/stage5/embedding_svd/openai_text_embedding_3_small/features`
- Reducers:
  `outputs/stage5/embedding_svd/openai_text_embedding_3_small/reducers`

## Next

Proceed to `5-8A`: Stage2 frozen + embedding-only bounded FiLM.

Suggested first pass:

1. `mean/dim16`, scale `0.02`
2. `mean/dim32`, scale `0.02` if dim16 is stable or weak
