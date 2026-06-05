# 4-N11 LLM Summary/Embedding Decision Deferred

Status: deferred.

Decision:

```text
Do not add LLM summaries or LLM embeddings before the headline-only,
no-leakage frozen-FiLM track is finalized.
```

Reason:

- N8/N9/N10 already show that the main unresolved issue is not just context
  richness, but where and how FiLM should intervene in the frozen Stage 2
  visual baseline.
- Adding an LLM encoder now would introduce a new variable before the
  headline-only TF-IDF/SVD baseline has been fully interpreted.
- N10-B indicates that useful corrections occur near the Stage 2 decision
  boundary, so the next controlled experiment should test gated FiLM before
  changing the text encoder.

If LLM context is used later, record:

- model name,
- version/date,
- prompt,
- cache hash,
- input time-window rule,
- leakage rule,
- runtime and cost notes.

Next step:

```text
Proceed to 4-N12-A uncertainty-gated news FiLM.
```
