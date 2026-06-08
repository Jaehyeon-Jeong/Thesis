# 5-3 Embedding Input Construction

Status: completed.

## Decision

Stage 5 will embed one deduplicated news headline at a time.

The locked embedding input template is:

```text
<headline>
```

Date, source, and URL are retained as metadata columns but excluded from the
embedding input text. This keeps the first LLM embedding experiment focused on
news content and reduces the risk of encoding calendar/source artifacts into
the vector representation.

## Counts

- Raw news rows: `30,626`
- Deduplicated rows: `29,208`
- Stage 5 usable news rows: `24,281`
- Usable news date range: `2018-02-28` to `2024-12-10`
- Unique usable news dates: `2,371`
- Mean input chars: `70.9`
- Max input chars: `279`
- Duplicate embedding hashes after dedupe: `0`

## Outputs

- Full input table: `outputs/stage5/embedding_inputs/stage5_embedding_input_items.csv`
- Manifest: `data_inventory/stage5_embedding_input_manifest.json`
- Summary: `reports/tables/stage5_embedding_input_summary.csv`
- Examples: `reports/tables/stage5_embedding_input_examples.csv`
- Report: `reports/tables/stage5_embedding_input_construction_report.md`

The full input table is not intended for GitHub tracking because it is a
large generated artifact. The manifest, report, summary, and sample preview are
small enough to track.

## Next

Proceed to `5-4/5-5`: embedding cache policy and OpenAI embedding generation.
