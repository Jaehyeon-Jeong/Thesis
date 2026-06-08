# 5-3 Embedding Input Construction

Status: completed.

## Decision

Stage 5 will embed each deduplicated news headline as one item. The embedding
text is headline-only:

```text
<headline>
```

Date, source, and URL are kept as metadata but are not included in the embedding
input. This keeps the first embedding experiment content-based and avoids
encoding calendar/source artifacts into the vector.

## Counts

- Raw news rows: `30626`
- Deduplicated news rows: `29208`
- Stage 5 usable news rows: `24281`
- Usable news date range: `2018-02-28` to `2024-12-10`
- Unique usable news dates: `2371`
- Mean input chars: `70.9`
- Max input chars: `279`
- Duplicate embedding hashes after dedupe: `0`

## Outputs

- Full input table: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/outputs/stage5/embedding_inputs/stage5_embedding_input_items.csv`
- Sample preview: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_embedding_input_examples.csv`
- Summary table: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_embedding_input_summary.csv`
- Manifest: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/data_inventory/stage5_embedding_input_manifest.json`

## Next Step

Proceed to `5-4/5-5`: define cache policy and call the embedding backend.
