# 5-9E FinBERT + F&G Bounded FiLM Prepared

## Objective

Test whether headline-level FinBERT sentiment becomes more useful when combined
with the market-level Fear & Greed regime signal.

## Model

- Image: `I60/R20/ohlc_ma_vb`
- Visual baseline: Stage2 seed `42-46` checkpoints
- Frozen parts: Stage2 CNN backbone and classifier
- Trainable parts: context MLP and bounded last-block FiLM
- Context method: `film_full_bounded_last_block`
- FiLM bound: `gamma = 1 + 0.02 * tanh(raw_gamma)`,
  `beta = 0.02 * tanh(raw_beta)`
- Seeds: `42,43,44,45,46`

## Context

The runner appends actual F&G regime features to the FinBERT context table:

- FinBERT numeric context features after excluding diagnostics: `79`
- F&G raw features: `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`
- Final context dimension verified locally: `83`

The F&G context is built automatically in Kaggle with
`scripts/build_stage4_context_features.py` if it is missing after a reset.

## Artifacts

- Kaggle one-cell:
  `notebooks/kaggle_stage5_9e_finbert_fg_film_ablation_one_cell.md`
- Updated context builder:
  `scripts/build_stage5_stage4_prebuilt_context.py`

## Interpretation Target

Compare against:

- Stage2 `ohlc_ma_vb` visual baseline
- N8B F&G-only bounded FiLM
- 5-9D FinBERT-only bounded FiLM

The key question is whether FinBERT adds incremental information beyond F&G, or
whether the positive-shifted FinBERT news tone remains noisy when combined with
the stronger market sentiment index.
