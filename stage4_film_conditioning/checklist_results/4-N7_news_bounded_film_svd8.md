# 4-N7 News-Context Bounded FiLM SVD8

Status: prepared for Kaggle execution

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

## Decision Rule

Promote N7 only if it satisfies both:

1. Accuracy or ROC-AUC is at least competitive with Stage 2 baseline
   (`accuracy=0.5793`, `ROC-AUC=0.5849`), or it clearly improves over SVD8
   concat while preserving ranking signal.
2. Seed-level predicted positive rate is less collapsed than N6.1 SVD8.

If N7 still collapses, the next step should be news-vector calibration or a
different text representation, not another arbitrary gamma/beta search.
