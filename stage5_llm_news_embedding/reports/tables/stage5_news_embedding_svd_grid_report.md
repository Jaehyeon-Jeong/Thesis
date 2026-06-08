# 5-7 Train-Only SVD/PCA Dimension Grid

Status: completed.

## Protocol

- Reducer fit split: `train` only.
- Validation/test are transformed with the train-fitted mean and components.
- Method: train-centered PCA implemented with NumPy SVD.
- Windows: `[7, 20, 60]`.
- Aggregations: `['mean', 'decay_mean']`.
- Dimensions: `[8, 16, 32]`.

## Mean Explained Variance Across 7/20/60 Windows

| aggregation   |   dimension |   mean_window_explained_variance_ratio |   min_window_explained_variance_ratio |   max_window_explained_variance_ratio |
|:--------------|------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|
| decay_mean    |           8 |                                 0.5924 |                                0.3968 |                                0.8006 |
| decay_mean    |          16 |                                 0.7214 |                                0.5223 |                                0.9220 |
| decay_mean    |          32 |                                 0.8283 |                                0.6607 |                                0.9716 |
| mean          |           8 |                                 0.6182 |                                0.4174 |                                0.8300 |
| mean          |          16 |                                 0.7483 |                                0.5456 |                                0.9476 |
| mean          |          32 |                                 0.8520 |                                0.6873 |                                0.9850 |

## Outputs

- Summary: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_news_embedding_svd_grid_summary.csv`
- Combined summary: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_news_embedding_svd_grid_combined_summary.csv`
- Manifest: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/data_inventory/stage5_news_embedding_svd_manifest.json`
- Feature tables root: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/outputs/stage5/embedding_svd/openai_text_embedding_3_small/features`
- Reducer root: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/outputs/stage5/embedding_svd/openai_text_embedding_3_small/reducers`

## Next

Use the best compact feature tables in `5-8A`: Stage2 frozen + embedding-only bounded FiLM.
