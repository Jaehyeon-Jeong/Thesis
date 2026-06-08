# 5-9D Prepared: FinBERT-Only Bounded FiLM

## Status

Prepared for Kaggle execution.

## Experiment

```text
image: I60/R20/ohlc_ma_vb
visual baseline: Stage2 pretrained checkpoint
freeze: CNN frozen + classifier frozen
context: FinBERT headline sentiment 7/20/60-day features
model: bounded last-block FiLM
scale: 0.02
seeds: 42,43,44,45,46
```

## Stage4 Names

- Context name:
  `stage5_finbert_context_i60_ohlc_ma_vb_r20_finbert_sentiment_v1`
- Feature set:
  `stage5_finbert_sentiment_v1`
- Expected Stage4 experiment:
  `stage4_film_full_bounded_last_block_i60_ohlc_ma_vb_r20_c60_stage5_finbert_sentiment_v1_pretrained_frozen_s0p02`

## Prepared Artifacts

- Kaggle one-cell:
  `notebooks/kaggle_stage5_9d_finbert_film_ablation_one_cell.md`
- Compact Kaggle upload bundle:
  `stage5_llm_news_embedding_5_9d_finbert_context_bundle.zip`
- Local Stage4 prebuilt context artifact:
  `stage4_film_conditioning/outputs/stage4/context/stage5_finbert_context_i60_ohlc_ma_vb_r20_finbert_sentiment_v1/seed_*/context_features.csv`

## Context Features

- Context dimension: `79`
- Sample count: `2,399`
- Train/validation/test split counts: `671 / 287 / 1,441`
- Source: `5-9C` FinBERT context feature table.

Diagnostic columns such as count-match flags and missing flags are excluded
from training. The model receives only numeric FinBERT count, ratio,
probability, sentiment, and news-FG proxy features.

## How To Run

Attach these Kaggle datasets:

- latest `stage4_film_conditioning`
- latest `stage2_btc_extension`
- Stage2 `I60/R20/ohlc_ma_vb` checkpoint bundle
- `stage5_llm_news_embedding_5_9d_finbert_context_bundle.zip`

Then copy the Python cell from:

```text
stage5_llm_news_embedding/notebooks/kaggle_stage5_9d_finbert_film_ablation_one_cell.md
```

## Next

Run `5-9D` on Kaggle and report mean/std results against:

- Stage2 `I60/R20/ohlc_ma_vb` baseline
- `5-8A` embedding-only bounded FiLM
- F&G-only Stage2 frozen bounded FiLM
