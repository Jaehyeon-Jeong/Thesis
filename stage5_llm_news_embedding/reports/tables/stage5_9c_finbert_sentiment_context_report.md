# 5-9C FinBERT Sentiment Context Features

Status: `ok`.

## Purpose

This step converts headline-level FinBERT sentiment into sample-level context
features aligned to the Stage4 image samples. For a sample ending at date `t`,
news is included only when:

```text
t-window <= news_date <= t-1
```

Same-day news is excluded, matching the Stage5 leakage policy.

## Inputs

- FinBERT items: `/Users/jaehyeonjeong/Desktop/논문/5_9a_results/outputs/stage5/finbert_sentiment/finbert_prosusai_headline_v1/stage5_finbert_sentiment_items.csv`
- Sample table: `/Users/jaehyeonjeong/Desktop/논문/stage4_film_conditioning/outputs/stage4/news/stage4_news_headline_windows_i60_r20/sample_headline_windows.parquet`
- Windows: `[7, 20, 60]`
- Headlines: `24281`
- Samples: `2399`

## Coverage

- 7d: missing `0.0000`, mean count `69.38`, count match `1.0000`
- 20d: missing `0.0000`, mean count `197.81`, count match `1.0000`
- 60d: missing `0.0000`, mean count `589.57`, count match `1.0000`

## Sentiment Regime Pattern

- 7d: train sentiment `-0.0318`, test sentiment `0.0594`, test news-FG proxy `69.61`
- 20d: train sentiment `-0.0305`, test sentiment `0.0575`, test news-FG proxy `73.25`
- 60d: train sentiment `-0.0344`, test sentiment `0.0556`, test news-FG proxy `79.50`

## Exported Context

- Feature columns: `91`
- Output CSV: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/outputs/stage5/finbert_context/finbert_prosusai_headline_v1/stage5_finbert_context_features.csv`
- Scaler: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/outputs/stage5/finbert_context/finbert_prosusai_headline_v1/stage5_finbert_context_scaler.json`
- Split summary: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_9c_finbert_sentiment_context_split_summary.csv`
- Feature summary: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_9c_finbert_sentiment_context_feature_summary.csv`

## Interpretation

The feature is usable for `5-9D` if coverage is complete, count alignment is
stable, and the train/test sentiment distribution is not collapsed. This step
does not prove predictive value; it only prepares a leakage-safe sentiment
regime context.
