# 5-1 Literature/Method/Backend Lock

Status: completed.

## Decision

Stage 5 will use LLM-derived news representation in two different roles:

1. Main performance feature:
   - Chen/Kelly/Xiu-style text representation.
   - News text is converted into embedding vectors.
   - The vectors become FiLM context after aggregation and train-only SVD/PCA.

2. Auxiliary interpretability feature:
   - Lopez-Lira/Tang-style prompt label.
   - GPT/Claude label news windows as `POSITIVE`, `NEGATIVE`, or `UNKNOWN`.
   - These labels explain regimes and correction/regression samples; they are
     not the main predictive representation.

## First Executable Backend

Use OpenAI API first:

```text
model: text-embedding-3-small
endpoint: /v1/embeddings
api key source: OPENAI_API_KEY
```

Official OpenAI documentation describes embeddings as vectors of floating-point
numbers and lists `text-embedding-3-small` as a current embedding model. The
embedding guide reports a default vector length of `1536` for
`text-embedding-3-small` and `3072` for `text-embedding-3-large`; it also
documents the `dimensions` parameter for reducing embedding size in third-
generation embedding models.

Sources:

- OpenAI embeddings guide:
  `https://platform.openai.com/docs/guides/embeddings`
- OpenAI `text-embedding-3-small` model page:
  `https://platform.openai.com/docs/models/text-embedding-3-small`

## Why OpenAI First

- It lets Stage 5 test a real LLM-derived semantic representation quickly.
- It avoids turning this thesis into an open-source embedding benchmark.
- `text-embedding-3-small` is the conservative first pass because it is smaller
  and cheaper than `text-embedding-3-large`.
- If the result is promising, later ablations can compare dimensions, local
  embedding models, or prompt-label features.

## Locked Stage 5 First Pass

```text
news item text
-> OpenAI text-embedding-3-small
-> headline-level embedding cache
-> 7/20/60-day window aggregation
-> train-only SVD/PCA: 8, 16, 32
-> Stage2 frozen + bounded last-block FiLM
```

First FiLM scale:

```text
scale = 0.02
```

The raw 1536-dimensional embedding will not be fed directly into the FiLM model
in the first pass because the BTC sample size is small.

## Reproducibility Rules

- Do not store API keys in files.
- Cache every input text with a stable hash.
- Cache model id, dimensions, API output shape, created timestamp, and output
  artifact path.
- Store raw embedding artifacts outside Git when they are large.
- Fit SVD/PCA only on the train split.

