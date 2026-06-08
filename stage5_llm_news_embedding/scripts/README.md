# Stage 5 Scripts

Planned scripts:

- `build_stage5_news_coverage_audit.py`
  - Implemented in `5-2`.
  - Reuses Stage 4 news audit artifacts and writes Stage 5 coverage/leakage
    tables.

- `build_stage5_news_embedding_inputs.py`
  - Implemented in `5-3`.
  - Creates deterministic news-item text inputs and hashes.
  - Does not call the embedding API.
  - Writes the full generated input table under `outputs/stage5` and small
    manifests/summaries under `data_inventory` and `reports/tables`.

- `build_stage5_embedding_backend_cache_policy.py`
  - Implemented in `5-4`.
  - Writes the OpenAI/Voyage execution order and cache policy.
  - Does not call OpenAI, Voyage, or any local embedding model.

- `build_stage5_news_embeddings.py`
  - Implemented in `5-5`.
  - Calls the selected embedding backend or loads cached vectors.
  - Currently supports OpenAI embedding runs.
  - Writes resumable batch `.npy` files first, then assembles the final
    embedding matrix and item manifest.

- `build_stage5_embedding_context_features.py`
  - Implemented in `5-6`.
  - Aggregates headline-level embeddings into 7/20/60-day windows.
  - Writes sample metadata plus mean and time-decay mean vector arrays.
  - Uses Stage 4 deduplicated headline-window samples when available.

- `build_stage5_embedding_svd.py`
  - Implemented in `5-7`.
  - Fits train-only SVD/PCA and transforms all splits.
  - Writes dimension-grid summaries, reducer arrays, and compact context CSVs
    for `mean` and `decay_mean` embedding windows.

- `build_stage5_stage4_prebuilt_context.py`
  - Implemented for `5-8A`.
  - Converts one compact Stage5 feature table into Stage4 `prebuilt` context
    artifacts.
  - Supports embedding-SVD context and FinBERT context through
    `--feature-source`.
  - Writes seed-specific `context_features.csv`, `context_scaler.json`,
    audit, and summary files under `stage4_film_conditioning/outputs/stage4`.

- `build_stage5_finbert_sentiment.py`
  - Implemented for `5-9A`.
  - Converts each deduplicated headline into FinBERT positive/neutral/negative
    probabilities, confidence, and `positive - negative` sentiment score.
  - Writes a headline-level sentiment table plus summary/report manifests.

- `build_stage5_finbert_sentiment_audit.py`
  - Implemented for `5-9B`.
  - Creates label distribution, year/daily summaries, confidence/sentiment
    quantiles, and sampled audit buckets from the 5-9A output.

- `build_stage5_finbert_context_features.py`
  - Implemented for `5-9C`.
  - Aggregates headline-level FinBERT probabilities/sentiment scores into
    strict t-1 7/20/60-day sample-level context features.
  - Writes context features, train-fit sentiment scaler, split summary, feature
    summary, report, and manifest.

- `run_stage5_embedding_film.py`
  - Planned only.
  - Current executable path is the Kaggle one-cell runner in `notebooks/`.

- `analyze_stage5_corrections.py`
  - Compares Stage2 baseline with Stage5 predictions.

Until these are implemented, Stage 5 should not be treated as a completed
experiment track.
