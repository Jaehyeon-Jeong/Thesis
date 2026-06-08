# Stage 5: LLM News Embedding Context

Stage 5 tests whether LLM-derived news representations can provide a richer
market-context signal than the hand-built numeric contexts tested in Stage 4.

This stage does not replace the Stage 2/Stage 4 pipeline. It keeps the strongest
visual protocol fixed:

- Visual baseline: Stage 2 `I60/R20/ohlc_ma_vb`.
- Current baseline metric: accuracy mean `0.5793`, ROC-AUC mean `0.5849`.
- Conditioning protocol: Stage 2 pretrained visual CNN loaded and frozen,
  bounded last-block FiLM trained from context only.
- Primary question: can news embeddings improve or conditionally correct the
  fixed chart-image model?

## Scope

Main experiment:

```text
BTC news headlines
-> pretrained text embedding model
-> headline-level vectors
-> 7/20/60-day trailing window aggregation
-> train-only SVD/PCA
-> MLP
-> bounded last-block FiLM
-> Stage 2 chart CNN feature modulation
```

This follows the representation-extraction direction of Chen, Kelly, and Xiu:
the LLM is used to convert financial text into numerical representations. It is
not asked to directly predict BTC returns.

Auxiliary interpretability:

```text
BTC news headlines
-> FinBERT sentiment or GPT/Claude fixed prompt
-> positive/negative/neutral tone, or richer event/regime labels
-> cached label features and qualitative explanation
```

This follows the prompt-classification direction of Lopez-Lira and Tang, but is
used as an interpretation layer, not as the main prediction model.

## Non-Goals

- This stage is not a general LLM benchmark.
- It will not compare many LLMs just for ranking.
- It will not use GPT/Claude as the final predictor.
- It will not overwrite the Stage 2 baseline or Stage 4 numeric-context results.

## Key Documents

- [Checklist](checklist.md)
- [Workflow diagram](workflow_diagram.md)
- [LLM embedding design](docs/stage5_llm_embedding_design.md)
- [Source map](docs/stage5_source_map.md)

## Current Status

- `5-5`: OpenAI `text-embedding-3-small` headline embedding table completed
  (`24,281 x 1,536`, failed rows `0`).
- `5-6`: 7/20/60-day trailing-window embedding context completed with strict
  `t-1` news alignment and full count match against the Stage 4 deduped
  headline-window table.
- `5-7`: train-only SVD/PCA grid completed for `8/16/32` dimensions.
  Practical first candidates for Stage 5 FiLM are `mean/dim16` and
  `mean/dim32`.
- `5-8A`: Kaggle runner prepared for Stage2 frozen + embedding-only bounded
  FiLM using `mean/dim16`, scale `0.02`, and seed `42-46`. The compact Kaggle
  upload bundle is `stage5_llm_news_embedding_5_8_compact_context_bundle.zip`.
- `5-8A` result: completed. Mean accuracy `0.5782`, ROC-AUC `0.5844`.
  This is essentially tied with but slightly below the Stage2 `ohlc_ma_vb`
  baseline (`0.5793`, `0.5849`), so it is not yet a performance improvement.
- `5-8C` partial result: `mean/dim32`, scale `0.05` completed. Mean accuracy
  `0.5768`, ROC-AUC `0.5847`. It improves Brier/AP only marginally but worsens
  accuracy/F1/trading, so the next useful direction is prompt/event features
  rather than a broad embedding scale grid.
- `5-9` plan locked: run FinBERT headline sentiment first as a news-only
  sentiment regime proxy. If this is insufficient, move to richer GPT/Claude
  prompt-based event/horizon/regime extraction.
- `5-9A` prepared: FinBERT headline sentiment extraction script and Kaggle
  one-cell are ready.
- `5-9A` result: completed on `24,281` headlines. Label shares are neutral
  `46.25%`, positive `27.77%`, negative `25.98%`; mean confidence is `0.8133`.
  Proceed to FinBERT window aggregation.
- `5-9B` result: FinBERT output audit completed. Label distribution is not
  collapsed, low-confidence share is only `3.18%`, and sample buckets are
  qualitatively plausible. Proceed to `5-9C` aggregation.
- `5-9C` result: FinBERT 7/20/60-day context aggregation completed. The output
  contains `91` context features for `2,399` samples, with missing rates `0`
  and count match rates `1.0000` against Stage4 deduplicated headline windows.
  Proceed to `5-9D` Stage2 frozen + FinBERT-only bounded FiLM.
- `5-9D` result: FinBERT-only bounded FiLM completed. Mean accuracy
  `0.578487`, ROC-AUC `0.586072`, AP `0.611943`, Brier `0.272739`.
  It does not beat Stage2/F&G-only accuracy, but it slightly improves
  ranking/calibration.
- `5-9E` result: FinBERT + F&G bounded FiLM completed. Mean accuracy
  `0.580569`, ROC-AUC `0.585843`, AP `0.611899`, Brier `0.272701`.
  This is a small positive/near-tie result: `+0.001249` accuracy over Stage2
  `ohlc_ma_vb` and `+0.000278` over F&G-only, but the margin is too small for a
  strong performance-improvement claim.
- Current reading: F&G remains the strongest compact external regime source;
  FinBERT sentiment can be combined with it without destabilizing the frozen
  Stage2 + bounded FiLM setup, but headline-level sentiment is still too weak
  to be the headline contribution. Next useful work is correction/regression
  analysis or prompt/event labels if a richer interpretable news feature is
  required.

## Output Policy

Tracked in GitHub:

- plans, checklists, small summary CSVs, reports, small figures

Not tracked:

- raw news dumps
- paid API responses if large
- embedding matrix files
- checkpoints
- large predictions
- Kaggle result bundles

Large Stage 5 artifacts should be saved as Kaggle or local zip bundles and
listed in `reports/tables` or `data_inventory` manifests.
