# 4-N7 News-Context Bounded FiLM SVD8

Status: completed on Kaggle

## Goal

Test whether bounded last-block FiLM can use the SVD8 headline-news vector as
conditional market-regime modulation while preserving the strong Stage 2 visual
path.

## Why SVD8

N6.1 compared lower-dimensional news vectors before FiLM:

| Vector | Context dim | Accuracy mean | ROC-AUC mean |
| --- | ---: | ---: | ---: |
| SVD32 concat | `102` | `0.5478` | `0.5644` |
| SVD16 concat | `54` | `0.5348` | `0.5608` |
| SVD8 concat | `30` | `0.5407` | `0.5817` |

SVD8 is selected because it preserves the strongest threshold-independent
ranking signal and keeps the FiLM generator input small.

## Fixed Setup

- Visual input: `I60/R20/ohlc_ma_vb`
- News alignment: strict `t-1`
- Headline windows: `7d`, `20d`, `60d`
- News vector: train-only TF-IDF/SVD, SVD dim `8`
- Context dim: `30`
- Method: `film_full_bounded_last_block`
- FiLM formula:

```text
gamma = 1 + 0.05 * tanh(raw_gamma)
beta  =     0.05 * tanh(raw_beta)
```

- Seeds: `42, 43, 44, 45, 46`
- Grad-CAM/context modulation export: enabled

## Prepared Kaggle Cell

- [kaggle_stage4_news_context_n7_bounded_film_svd8_one_cell.md](../notebooks/kaggle_stage4_news_context_n7_bounded_film_svd8_one_cell.md)

The output bundle is:

```text
/kaggle/working/stage4_news_context_n7_bounded_film_svd8_result_bundle.zip
```

## Result

Five-seed result:

| Metric | N7 value |
| --- | ---: |
| Accuracy mean | `0.5591` |
| Accuracy std | `0.0116` |
| ROC-AUC mean | `0.5642` |
| ROC-AUC std | `0.0281` |
| F1 mean | `0.6387` |
| Predicted-Up-rate mean | `0.6952` |
| Long/flat Sharpe net mean | `3.1372` |
| Long/short Sharpe net mean | `2.0629` |

Seed-level summary:

| Seed | Accuracy | ROC-AUC | Predicted-Up rate |
| ---: | ---: | ---: | ---: |
| 42 | `0.5656` | `0.5746` | `0.6218` |
| 43 | `0.5552` | `0.5769` | `0.5461` |
| 44 | `0.5718` | `0.5912` | `0.5989` |
| 45 | `0.5614` | `0.5602` | `0.7092` |
| 46 | `0.5413` | `0.5180` | `1.0000` |

## Interpretation

N7 reduced seed-level collapse relative to SVD8 news concat, but it did not
beat the Stage 2 selected visual baseline (`accuracy=0.5793`,
`ROC-AUC=0.5849`). The bounded FiLM values stayed conservative:
`gamma` was close to `1` and `beta` close to `0`, so the failure is not simply
an uncontrolled gamma/beta explosion.

The important design correction is that N7 still uses the Stage 2 CNN
architecture from scratch. It does not load and preserve the Stage 2 learned
weights. Therefore N7 does not yet test the intended hypothesis:

```text
strong pretrained chart CNN + market/news context as bounded FiLM correction
```

Next step: move to `4-N8`, Stage 2 pretrained baseline-preserving FiLM, before
adding more context sources such as News + F&G combined context.
