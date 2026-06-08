# 5-13 Final Stage5 Report

Stage5 is closed for the first thesis draft.

The final Stage5 candidate is FinBERT headline sentiment aggregates + F&G raw
regime features with Stage2 `I60/R20/ohlc_ma_vb` frozen and bounded last-block
FiLM at scale `0.02`.

## Core Result

| Model | Accuracy | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|
| Stage2 visual baseline | `0.579320` | `0.584862` | `0.611256` | `0.274337` |
| F&G-only bounded FiLM | `0.580291` | `0.584930` | `0.611272` | `0.274004` |
| FinBERT-only bounded FiLM | `0.578487` | `0.586072` | `0.611943` | `0.272739` |
| FinBERT+F&G bounded FiLM | `0.580569` | `0.585843` | `0.611899` | `0.272701` |

FinBERT+F&G is the best Stage5 row by accuracy, but the improvement is small:
`+0.001249` accuracy over Stage2 and `+0.000278` over F&G-only.

## Conditional Result

Against Stage2 over `7,205` matched decisions:

- Corrections: `95`.
- Regressions: `86`.
- Net corrections: `+9`.
- Changed prediction rate: `2.5121%`.

The strongest positive buckets are Stage2 uncertainty `45-55`, F&G greed,
high 7-day news count, and high 60-day F&G mean. Weak/negative buckets include
F&G neutral, F&G extreme fear, and low 60-day news count.

## Interpretability Result

The 5-12 targeted export shows small final-block modulation:

- Correction gamma mean: `1.000334`.
- Regression gamma mean: `1.000349`.
- Correction beta mean: `0.000079`.
- Regression beta mean: `0.000082`.

This supports the interpretation that FinBERT+F&G acts as a conservative
calibration layer rather than a strong visual feature rewrite.

## Thesis Claim

The Stage5 result supports a modest and defensible claim:

> News-derived sentiment and F&G context can be attached to a strong chart CNN
> through bounded FiLM without destabilizing the visual baseline. The benefit is
> small overall, but it is interpretable as conditional calibration, especially
> when the chart-only model is uncertain or sentiment/news context is active.

## Title Decision

Recommended title:

```text
Context-Conditioned FiLM for Bitcoin Direction Prediction from Price Charts
```

Recommended subtitle:

```text
Market Sentiment and News-Derived Signals as Bounded Calibration Context
```

Alternative:

```text
News- and Sentiment-Conditioned FiLM for Bitcoin Direction Prediction from Price Charts
```

Avoid the stronger original title unless the final scope changes:

```text
LLM-to-FiLM Multimodal Fusion for Market Regime Classification from Price Charts and News
```
