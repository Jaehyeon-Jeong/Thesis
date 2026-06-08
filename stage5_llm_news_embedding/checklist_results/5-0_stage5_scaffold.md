# 5-0 Stage 5 Scaffold

Status: completed.

Created Stage 5 as a separate experiment track for LLM-derived news embedding
context.

## Files

- `README.md`
- `checklist.md`
- `workflow_diagram.md`
- `docs/stage5_llm_embedding_design.md`
- `docs/stage5_source_map.md`
- `notebooks/README.md`
- `scripts/README.md`
- `configs/README.md`
- `data_inventory/README.md`
- `reports/README.md`

## Locked Direction

Stage 5 will use:

- embedding-based news representation as the main performance feature,
- prompt-based GPT/Claude labels as auxiliary interpretability features,
- Stage 2 pretrained/frozen chart CNN,
- bounded last-block FiLM as the first conditioning model.

## Next Step

Proceed to `5-1` and lock the first executable path:

- OpenAI embedding API first, or
- local open-source embedding model first.

