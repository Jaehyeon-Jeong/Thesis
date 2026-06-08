# 5-8A Embedding-Only Bounded FiLM Prepared

Status: prepared, not yet trained on Kaggle.

## Experiment

```text
Stage2 baseline: I60/R20/ohlc_ma_vb
Context: OpenAI text-embedding-3-small SVD features + news counts
Aggregation/dimension: mean / dim16
Context features: 51
Model: Stage2 CNN + classifier frozen
FiLM: bounded last-block FiLM
Scale: 0.02
Seeds: 42,43,44,45,46
```

## Prepared Artifacts

- Kaggle runner:
  `notebooks/kaggle_stage5_2_embedding_film_ablation_one_cell.md`
- Kaggle upload bundle:
  `stage5_llm_news_embedding_5_8_compact_context_bundle.zip`
- Stage 4 prebuilt context name:
  `stage5_embedding_context_i60_ohlc_ma_vb_r20_embedding_mean_dim16`
- Stage 4 experiment name:
  `stage4_film_full_bounded_last_block_i60_ohlc_ma_vb_r20_c60_stage5_embedding_mean_dim16_pretrained_frozen_s0p02`

## Local Context Verification

The Stage5 compact feature table was converted to Stage4 prebuilt context
artifacts for seeds `42` to `46`.

Expected per-seed artifacts:

```text
outputs/stage4/context/stage5_embedding_context_i60_ohlc_ma_vb_r20_embedding_mean_dim16/seed_*/context_features.csv
outputs/stage4/context/stage5_embedding_context_i60_ohlc_ma_vb_r20_embedding_mean_dim16/seed_*/context_scaler.json
outputs/stage4/context/stage5_embedding_context_i60_ohlc_ma_vb_r20_embedding_mean_dim16/seed_*/context_feature_audit.json
outputs/stage4/context/stage5_embedding_context_i60_ohlc_ma_vb_r20_embedding_mean_dim16/seed_*/context_feature_summary.csv
```

`context_scaler.json` uses `51` normalized context columns:

- `3` news-count features: 7/20/60-day counts.
- `48` embedding features: 7/20/60-day windows x 16 SVD dimensions.

## Next Step

Upload/attach the compact Stage5 bundle, Stage4 code snapshot, Stage2 code
snapshot, and Stage2 `I60/R20/ohlc_ma_vb` checkpoint bundle in Kaggle, then run
the one-cell notebook.
