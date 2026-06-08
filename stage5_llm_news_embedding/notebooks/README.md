# Stage 5 Notebooks

Kaggle one-cell runners will be added here.

Planned notebooks:

- `kaggle_stage5_1_build_llm_embedding_features_one_cell.md`
  - Build/copy news input table.
  - Create or load cached embeddings.
  - Build 7/20/60-day embedding context features.

- `kaggle_stage5_2_embedding_film_ablation_one_cell.md`
  - Implemented for `5-8A`.
  - Train/evaluate Stage2 frozen + bounded last-block FiLM using embedding
    context.
  - Run seeds `42,43,44,45,46`.
  - Default candidate: OpenAI `text-embedding-3-small`, `mean/dim16`,
    scale `0.02`.

- `kaggle_stage5_9a_finbert_sentiment_one_cell.md`
  - Implemented for `5-9A`.
  - Runs headline-level FinBERT sentiment extraction and bundles the output for
    sampled audit and 7/20/60-day aggregation.

- `kaggle_stage5_9d_finbert_film_ablation_one_cell.md`
  - Implemented for `5-9D`.
  - Trains/evaluates Stage2 frozen + bounded last-block FiLM using FinBERT
    7/20/60-day sentiment context.
  - Run seeds `42,43,44,45,46`.
  - Default candidate: FinBERT-only context, scale `0.02`.

- `kaggle_stage5_3_prompt_label_interpretability_one_cell.md`
  - Use cached GPT/Claude prompt labels.
  - Build interpretation buckets and correction/regression report.

Do not put large output directly in GitHub. Save Kaggle result bundles and list
their manifest paths in reports.
