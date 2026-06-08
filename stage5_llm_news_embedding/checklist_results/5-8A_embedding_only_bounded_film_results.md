# 5-8A Embedding-Only Bounded FiLM Results

Status: completed.

Local result folder:

```text
/Users/jaehyeonjeong/Desktop/논문/5_8a_results
```

## Setup

```text
Visual baseline: Stage2 I60/R20/ohlc_ma_vb
Context: OpenAI text-embedding-3-small headline embeddings
Aggregation/dimension: mean / dim16
Context features: 51
Model: Stage2 CNN + classifier frozen
FiLM: bounded last-block FiLM
Scale: 0.02
Seeds: 42,43,44,45,46
```

Context audit:

- Train/validation/test rows: `671 / 287 / 1441`.
- Missing rate: `0.0` for all context features.
- Warnings: none.

## Main Result

Compared with the Stage2 `ohlc_ma_vb` baseline, 5-8A is essentially tied but
slightly worse on the main metrics.

| model | acc mean | acc std | ROC-AUC mean | AP mean | F1 mean | pred. positive rate |
|---|---:|---:|---:|---:|---:|---:|
| Stage2 `ohlc_ma_vb` | 0.579320 | 0.018218 | 0.584862 | 0.611256 | 0.651071 | 0.664400 |
| 5-8A embedding FiLM | 0.578210 | 0.018784 | 0.584401 | 0.611036 | 0.647212 | 0.654129 |

Mean deltas:

```text
accuracy: -0.001110
ROC-AUC:  -0.000461
AP:       -0.000220
F1:       -0.003859
Brier:    -0.000444
```

The small Brier improvement suggests a tiny calibration benefit, but it is not
enough to offset the accuracy/F1/trading metric decline.

## Seed-Level Interpretation

Only seed `42` improved, and only by `+0.000694` accuracy. Seeds `43-46` were
all slightly below the matching Stage2 baseline.

```text
seed42 accuracy delta: +0.000694
seed43 accuracy delta: -0.001388
seed44 accuracy delta: -0.001388
seed45 accuracy delta: -0.001388
seed46 accuracy delta: -0.002082
```

There is no seed collapse: predicted-positive rate stays between `0.613` and
`0.713`. However, the model still has an upward prediction bias relative to the
test positive rate `0.541`.

## Conclusion

This specific embedding setup does not provide a clear performance gain over
the strongest Stage2 visual baseline.

The experiment is still useful because it shows:

- OpenAI headline embeddings can be integrated cleanly into the frozen FiLM
  pipeline.
- The semantic embedding context does not collapse the model.
- `mean/dim16 + scale 0.02` is likely too conservative or not discriminative
  enough to change decisions meaningfully.

## Recommended Follow-Up

Run one focused follow-up, not a large grid:

1. `mean/dim32`, scale `0.02`.
2. If still stable but weak, `mean/dim32`, scale `0.05`.
3. Add correction/regression analysis only for the best of 5-8A and the
   dim32 follow-up.

If dim32 does not improve, Stage5 should shift from generic embedding vectors
to more targeted sentiment/event features.
