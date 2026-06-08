# 5-9 Plan: FinBERT News Sentiment Regime Proxy

## Motivation

The OpenAI embedding experiments showed that generic semantic headline vectors
do not robustly improve the frozen Stage 2 chart baseline. The likely issue is
that generic embeddings are not explicitly aligned to BTC market direction or
regime.

FinBERT is a useful intermediate step before GPT/Claude prompt-based event
labeling:

- It is cheaper and easier to reproduce than API prompt labeling.
- It provides a direct financial text tone signal: positive, neutral, negative.
- It can be converted into an interpretable news-only sentiment regime proxy.

FinBERT is not the same as F&G. F&G is a market-level sentiment/regime index;
FinBERT is headline-level financial text tone. The experiment tests whether
news text tone, aggregated over 7/20/60-day windows, can provide context that
the visual chart baseline does not already contain.

## Experiment Sequence

### 5-9A. FinBERT headline sentiment extraction

- Input: Stage 5 deduplicated headline table.
- Model: fixed FinBERT sentiment classifier.
- Unit: one headline/news item.
- Output columns:
  - `finbert_positive_prob`
  - `finbert_neutral_prob`
  - `finbert_negative_prob`
  - `finbert_label`
  - `finbert_confidence`
  - `finbert_sentiment_score = positive_prob - negative_prob`
  - `headline_hash`
  - model/cache metadata

### 5-9B. FinBERT output audit

Audit only a sample, not the full dataset:

- random headlines
- high-confidence positive headlines
- high-confidence negative headlines
- low-confidence/neutral headlines
- high-news-density dates

The goal is not perfect manual labeling. The goal is to verify that the model
is not systematically misreading obvious financial tone.

### 5-9C. Window aggregation and news-only F&G proxy

Aggregate item-level sentiment into 7/20/60-day trailing windows with strict
`t-1` alignment.

Primary features:

- `finbert_news_count_{7,20,60}d`
- `finbert_positive_ratio_{7,20,60}d`
- `finbert_negative_ratio_{7,20,60}d`
- `finbert_neutral_ratio_{7,20,60}d`
- `finbert_sentiment_mean_{7,20,60}d`
- `finbert_sentiment_sum_{7,20,60}d`
- `finbert_confidence_mean_{7,20,60}d`
- `finbert_high_conf_positive_count_{7,20,60}d`
- `finbert_high_conf_negative_count_{7,20,60}d`

Optional interpretability proxy:

```text
news_sentiment_z = train_zscore(finbert_sentiment_mean_window)
news_fear_greed_proxy = 50 + 50 * tanh(news_sentiment_z)
```

This is not the official Alternative.me F&G index. It is a news-only sentiment
regime proxy for analysis.

### 5-9D. Stage2 frozen + FinBERT-only bounded FiLM

- Visual model: Stage2 `I60/R20/ohlc_ma_vb` checkpoint, CNN/classifier frozen.
- Context: FinBERT aggregate features only.
- Model: bounded last-block FiLM.
- Scale: start with `0.02`.
- Seeds: `42,43,44,45,46`.
- Purpose: test whether financial news tone alone improves/ties the visual
  baseline more meaningfully than generic embeddings.

### 5-9E. Stage2 frozen + FinBERT + F&G bounded FiLM

- Context: FinBERT aggregate features + F&G-only features.
- Purpose: test whether article tone plus market-level sentiment regime is
  stronger than either source alone.

### 5-9F. Compare against embedding and prior Stage 4 candidates

Compare:

- Stage2 `ohlc_ma_vb`
- Stage5 embedding dim16 scale0.02
- Stage5 embedding dim32 scale0.05
- FinBERT-only
- FinBERT + F&G
- existing F&G-only frozen candidate

Metrics:

- accuracy
- ROC-AUC
- average precision
- F1
- Brier score
- predicted positive rate
- correction/regression counts
- conditional buckets: high positive sentiment, high negative sentiment,
  high-confidence sentiment, F&G extreme fear/greed

## Decision Rule

If FinBERT-only or FinBERT+F&G improves or nearly ties the baseline while
improving calibration/correction behavior, continue to interpretability export.

If FinBERT does not help, move to prompt-based event/horizon extraction:

- BTC relevance
- direction
- event type
- impact horizon
- risk regime
- confidence

That next step is more expressive than FinBERT because it separates sentiment
from event type and expected impact horizon.
