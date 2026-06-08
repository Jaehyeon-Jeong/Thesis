# 4-N13-7D F&G Classifier-Unfreeze FiLM

## Status

Completed and reviewed.

## Purpose

N13-7A showed that simply increasing the bounded FiLM scale does not improve the
Stage 2 frozen protocol. Larger scale produced more corrections, but also more
regressions.

N13-7D therefore tests a different hypothesis:

```text
The visual chart filters should stay fixed, but the final decision boundary may
need to adapt to the FiLM-modulated feature map.
```

This keeps the core Stage 2 visual representation intact while allowing the
classifier to learn how to use the small F&G-conditioned correction.

## Fixed Setup

```text
image: I60/R20/ohlc_ma_vb
context: F&G-only
context features: fg_value, fg_mean_60, fg_delta_60, fg_std_60
model: film_full_bounded_last_block
FiLM scale: 0.02
visual CNN: frozen Stage 2 checkpoint
classifier: trainable, initialized from Stage 2 checkpoint
trainable modules: final classifier + F&G context encoder + final-block FiLM heads
seeds: 42, 43, 44, 45, 46
```

## Why This Comes Before N13-7B/C

N13-7B/C change the FiLM constraint or equation. That can make the experiment
harder to interpret because it changes how gamma/beta are parameterized.

N13-7D changes only the freeze policy:

```text
N8-B: visual CNN frozen, classifier frozen
N13-7D: visual CNN frozen, classifier trainable
```

So if performance improves, the interpretation is clear: the F&G FiLM branch was
not necessarily too weak; the fixed Stage 2 classifier was too rigid to use the
context-modulated features.

## Kaggle Runner

Use:

[kaggle_stage4_n13_7d_fg_classifier_unfreeze_one_cell.md](../notebooks/kaggle_stage4_n13_7d_fg_classifier_unfreeze_one_cell.md)

The runner writes:

```text
reports/tables/stage4_n13_7d_fg_classifier_unfreeze_seed_results.csv
reports/tables/stage4_n13_7d_fg_classifier_unfreeze_mean_std_results.csv
reports/tables/stage4_n13_7d_fg_classifier_unfreeze_correction_summary.csv
reports/tables/stage4_n13_7d_fg_classifier_unfreeze_s0p02_stage2_vs_film_correction_analysis_*.csv
```

It also creates:

```text
/kaggle/working/stage4_n13_7d_fg_classifier_unfreeze_result_bundle.zip
```

## Readout

Compare against:

```text
Stage 2 baseline
N8-B F&G-only scale 0.02
N13-7A scale 0.10/0.20
```

Key metrics:

```text
accuracy
ROC-AUC
Brier score
predicted_positive_rate
Stage2 wrong -> FiLM correct
Stage2 correct -> FiLM wrong
net correction
gamma/beta magnitude summary
```

## Decision Rule

Keep N13-7D only if it improves N8-B `scale=0.02` or gives a clearly better
correction/regression tradeoff without class collapse.

If N13-7D fails, the likely conclusion is:

```text
F&G FiLM can preserve the strong visual baseline, but the current compact F&G
context does not provide enough incremental signal to materially improve the
I60/R20/ohlc_ma_vb Stage 2 model.
```

## Result

| Model | Accuracy mean | ROC-AUC mean | Brier mean | Predicted positive rate | Trainable params |
| --- | ---: | ---: | ---: | ---: | ---: |
| N8-B F&G `scale=0.02`, classifier frozen | `0.580291` | `0.584930` | `0.274004` | `0.660652` | `35,008` |
| N13-7D F&G `scale=0.02`, classifier trainable | `0.574323` | `0.584220` | `0.280218` | `0.684108` | `403,650` |

Stage2-vs-N13-7D correction summary:

```text
corrections: 255
regressions: 291
net correction: -36
accuracy delta vs Stage 2: -0.004997
```

Seed-level behavior:

| Seed | Stage2 acc | N13-7D acc | Delta | Corrections | Regressions | Pred+ rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `42` | `0.603053` | `0.579459` | `-0.023595` | `49` | `83` | `0.524636` |
| `43` | `0.574601` | `0.577377` | `+0.002776` | `37` | `33` | `0.732130` |
| `44` | `0.562804` | `0.564885` | `+0.002082` | `46` | `43` | `0.696044` |
| `45` | `0.562804` | `0.564885` | `+0.002082` | `95` | `92` | `0.802915` |
| `46` | `0.593338` | `0.585010` | `-0.008328` | `28` | `40` | `0.664816` |

## Interpretation

Classifier unfreezing made the model much more active: correction/regression
counts jumped from the small-scale frozen-classifier regime into hundreds of
changed decisions. However, the extra flexibility did not produce a useful
decision boundary. It over-adjusted the Stage 2 classifier, especially in seeds
`42` and `46`, and increased the average predicted-positive rate to `0.684108`.

This weakens the hypothesis that the fixed Stage 2 classifier was the main
bottleneck. The stronger reading is:

```text
The Stage 2 visual representation is strong, and F&G-only context can make small
bounded corrections, but giving the classifier more freedom lets noise/regime
signals overwrite too many correct visual decisions.
```

## Decision

Do not keep N13-7D as the final model.

The best F&G model remains N8-B `scale=0.02`, because it is the only F&G
configuration here that preserves Stage 2 while giving positive net correction.
