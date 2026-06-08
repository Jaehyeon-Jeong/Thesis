# 5-8C Partial Result: Embedding dim32, FiLM scale 0.05

## Setup

- Visual baseline: Stage 2 `I60/R20/ohlc_ma_vb`
- Context: OpenAI `text-embedding-3-small` headline embeddings
- Aggregation: trailing-window embedding mean + news counts
- SVD dimension: `32`
- Model: Stage2 pretrained CNN/classifier frozen + bounded last-block FiLM
- FiLM scale: `0.05`
- Seeds: `42,43,44,45,46`

## Five-Seed Comparison

| model | acc mean | acc std | ROC-AUC mean | AP mean | F1 mean | Brier mean | pred positive mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage2 `ohlc_ma_vb` | 0.579320 | 0.018218 | 0.584862 | 0.611256 | 0.651071 | 0.274337 | 0.664400 |
| Stage5 dim16 scale 0.02 | 0.578210 | 0.018784 | 0.584401 | 0.611036 | 0.647212 | 0.273893 | 0.654129 |
| Stage5 dim32 scale 0.05 | 0.576822 | 0.020804 | 0.584749 | 0.611454 | 0.637770 | 0.272929 | 0.626648 |

## Interpretation

Increasing both embedding dimension and FiLM scale does not improve the main
classification metric. Accuracy decreases by `0.002498` vs Stage2 and by
`0.001388` vs the dim16/scale0.02 Stage5 run.

The useful signal is limited to ranking/calibration:

- ROC-AUC is almost tied with Stage2.
- Average precision improves only slightly.
- Brier score improves modestly.
- Predicted positive rate drops from `0.6644` to `0.6266`, so the model becomes
  less up-biased, but this hurts F1 and trading Sharpe.

Seed-level behavior is stable enough to avoid a collapse diagnosis, but not
strong enough for a performance claim. Seeds `43`, `44`, and `45` lose accuracy
relative to Stage2; only seed `46` improves.

## Decision

Do not expand a broad embedding dimension/scale grid unless a narrow diagnostic
is needed. Generic headline embeddings are not yet giving a robust directional
signal over the strong visual baseline. The next more useful direction is to
test lower-dimensional, interpretable prompt/event features rather than only
larger embedding vectors.
