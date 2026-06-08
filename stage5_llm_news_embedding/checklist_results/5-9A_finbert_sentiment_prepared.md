# 5-9A Prepared: FinBERT Headline Sentiment Extraction

## Purpose

Prepare a cheaper and more interpretable news-tone feature before moving to
GPT/Claude event prompts. This tests whether headline-level financial sentiment
contains more directionally useful information than generic semantic embeddings.

## Prepared Artifacts

- Script: `scripts/build_stage5_finbert_sentiment.py`
- Kaggle one-cell: `notebooks/kaggle_stage5_9a_finbert_sentiment_one_cell.md`

## Default Configuration

- Model: `ProsusAI/finbert`
- Input: `stage5_embedding_input_items.csv`
- Text column: `embedding_input_text`
- Unit: one deduplicated headline
- Batch size: `128`
- Max token length: `128`

## Output Columns

- `finbert_positive_prob`
- `finbert_negative_prob`
- `finbert_neutral_prob`
- `finbert_label`
- `finbert_confidence`
- `finbert_sentiment_score = positive_prob - negative_prob`

## Next Step

Run the Kaggle one-cell. If output label distributions look reasonable, proceed
to `5-9B/5-9C`: sampled audit and 7/20/60-day FinBERT sentiment aggregation.
