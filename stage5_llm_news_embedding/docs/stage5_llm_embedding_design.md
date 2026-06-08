# Stage 5 LLM Embedding Design

## Research Question

Stage 4 showed that many hand-built numeric contexts were either redundant with
the chart image or too weak to robustly improve the Stage 2 baseline. Stage 5
therefore tests whether financial news text contains additional market-regime
information that can condition the same chart image through FiLM.

The key question is not:

```text
Can GPT directly predict Bitcoin?
```

The key question is:

```text
Can an LLM-derived news representation help a fixed chart CNN interpret the same
visual pattern differently under different market news contexts?
```

## Main Method

Use the LLM as a text representation extractor.

```text
news text -> embedding vector -> window aggregation -> SVD/PCA -> FiLM context
```

This is closer to Chen, Kelly, and Xiu's embedding-based representation approach
than to prompt-based direct return prediction.

## 5-1 Backend Lock

First executable backend:

```text
OpenAI text-embedding-3-small
```

Reason:

- It is a dedicated embedding model, not a chat/reasoning model.
- It returns a floating-point vector for each input text.
- It is cheaper and lighter than the larger OpenAI embedding model, which is
  appropriate for a first pass over headline-level BTC news.
- The official OpenAI embedding guide reports default vector length `1536` for
  `text-embedding-3-small` and supports a `dimensions` parameter for
  third-generation embedding models.

Stage 5 first pass will keep the API output at the default `1536` dimensions,
then fit train-only SVD/PCA to `8`, `16`, and `32` dimensions before feeding
the features into FiLM.

If cost/storage or rate limits become a blocker, a second pass may use the
OpenAI `dimensions` parameter or a local open-source embedding model. That is a
fallback, not the primary Stage 5 path.

API key policy:

```text
OPENAI_API_KEY environment variable only
```

No API key should be written into configs, notebooks, CSVs, logs, or committed
files.

## 5-4 Backend Comparison Lock

Execution order:

```text
1. OpenAI text-embedding-3-small
2. Voyage AI voyage-4
3. Optional: OpenAI text-embedding-3-large
4. Optional: Voyage AI voyage-4-large
```

Claude is not listed as an embedding model because Anthropic does not provide a
native embedding endpoint. Voyage AI is used as the Anthropic-side comparison
provider.

All embedding outputs are cache-keyed by:

```text
embedding_input_hash + provider + model + requested_dimensions
```

This makes reruns deterministic at the data-artifact level even if Kaggle
working storage is reset.

## Auxiliary Method

Use GPT/Claude prompt labels only for interpretation.

```text
news text -> POSITIVE / NEGATIVE / UNKNOWN + confidence + reason
```

This is closer to Lopez-Lira and Tang's prompt-based news classification
approach. In Stage 5 it should be used to explain why a news window may be
positive, negative, or uncertain, not as the main predictive representation.

## Input Construction

Primary unit: one news item.

Locked 5-3 text template:

```text
<headline>
```

Date, source, URL, and other metadata are retained in the input manifest but
are not included in the embedding text. This keeps the representation
content-based and avoids encoding calendar/source artifacts into the embedding
vector. Summary/full-text inputs remain a later ablation, not the first run.

## Leakage Policy

For a sample ending at date `t`, news may only use information available before
the prediction is made.

Default:

```text
news_date <= image_end_date - 1 day
```

If exact publication time is available, use timestamp-level filtering. If only
date is available, keep the conservative one-day lag.

## Embedding Granularity

Preferred:

```text
news item -> embedding
window -> aggregate item embeddings
```

Reason:

- avoids long-window text truncation,
- preserves news count and window density,
- allows later inspection of which items are in each window,
- gives stable input construction across 7/20/60 windows.

Backup/ablation:

```text
whole news window text -> one embedding
```

This is simpler, but harder to interpret and more sensitive to input length.

## Window Aggregation

Use three windows because they match the current chart/regime discussion:

- 7 days: short news shock.
- 20 days: return horizon-aligned context.
- 60 days: chart lookback-aligned regime context.

Candidate aggregation features:

- mean embedding per window,
- time-decay mean embedding per window,
- news count per window,
- missing/no-news indicator per window.

5-6 implementation:

```text
headline embeddings -> 7/20/60-day sample windows
```

For each image sample ending at date `t`, Stage 5 uses `t-window` through
`t-1` only. Same-day news remains excluded. The exported arrays are:

- simple mean vectors,
- exponential time-decay mean vectors with half-life `window / 2`.

The 5-6 count verification matches the Stage 4 deduplicated headline-window
audit at `1.0000` for all 7/20/60 windows.

## Dimensionality Reduction

Do not feed 1536/3072-dimensional embeddings directly into the current Stage 4
FiLM model. The BTC sample size is too small.

Initial grid:

```text
SVD/PCA dimensions: 8, 16, 32
```

Fit SVD/PCA on train split only, then transform validation/test. This avoids
leakage.

5-7 implementation:

```text
7/20/60-day embedding windows -> train-centered PCA via NumPy SVD
```

Reducers are fit only on the `train` split (`671` rows), then applied to all
`2,399` Stage 5 samples. Both simple mean and time-decay mean windows are
reduced.

Mean explained variance across 7/20/60 windows:

| aggregation | dim 8 | dim 16 | dim 32 |
|---|---:|---:|---:|
| mean | 0.6182 | 0.7483 | 0.8520 |
| decay_mean | 0.5924 | 0.7214 | 0.8283 |

Practical first candidates:

- `mean/dim16`: conservative first pass, fewer context features.
- `mean/dim32`: stronger representation pass, higher variance coverage but
  more overfitting risk.

5-8A starts with `mean/dim16`.

The Stage4 prebuilt context contains `51` features:

```text
3 news-count features
+ 7-day mean embedding SVD16
+ 20-day mean embedding SVD16
+ 60-day mean embedding SVD16
```

This is intentionally smaller than the full `1536`-dimensional embedding and
keeps the first FiLM test conservative.

## Model Protocol

Primary protocol:

```text
Stage2 I60/R20/ohlc_ma_vb checkpoint
-> load visual CNN/classifier
-> freeze visual weights
-> train context MLP + bounded last-block FiLM
```

FiLM:

```text
gamma = 1 + scale * tanh(raw_gamma)
beta  = scale * tanh(raw_beta)
```

Start with `scale = 0.02`. Increase only after the embedding context is stable.

## First Experiment Set

1. Embedding-only context:

```text
embedding_svd_7d/20d/60d + news_count_7d/20d/60d
```

2. Embedding + F&G:

```text
embedding context + F&G compact regime features
```

3. Prompt-label auxiliary:

```text
POSITIVE/NEGATIVE/UNKNOWN score + confidence
```

Prompt labels are mainly for interpretation unless embedding-only results are
clearly weak and the labels show stronger signal.

## Required Outputs

- `embedding_input_items.csv`
- `embedding_manifest.json`
- `openai_embedding_cache_manifest.json`
- `stage5_news_embedding_context_features.csv`
- `stage5_news_embedding_svd_context_features.csv`
- seed-level metrics
- mean/std summary
- correction/regression table
- Grad-CAM and FiLM modulation exports for selected samples
- final Stage 5 report

## Interpretation Standard

A useful Stage 5 result does not have to be a large overall accuracy jump if it
shows a clear conditional mechanism. However, the claim must be separated:

- overall performance improvement,
- same-image baseline improvement,
- conditional improvement in high-news/high-confidence regimes,
- qualitative interpretability through news text, prompt label, Grad-CAM, and
  gamma/beta modulation.
