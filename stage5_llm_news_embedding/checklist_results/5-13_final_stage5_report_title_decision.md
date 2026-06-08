# 5-13 Final Stage5 Report and Thesis Title Decision

## Status

Completed.

## Purpose

Close the first Stage5 news-context branch and decide how it should be used in
the bachelor thesis. This step does not train a new model. It consolidates the
embedding, FinBERT, FinBERT+F&G, correction/regression, and Grad-CAM/modulation
results into thesis-ready claims.

## Final Stage5 Model Decision

The best Stage5 candidate is:

```text
Stage2 I60/R20/ohlc_ma_vb visual CNN/classifier frozen
+ FinBERT headline sentiment aggregates
+ F&G raw regime features
+ bounded last-block FiLM
+ scale 0.02
```

Experiment id:

```text
stage4_film_full_bounded_last_block_i60_ohlc_ma_vb_r20_c60_stage5_finbert_fg_sentiment_v1_pretrained_frozen_s0p02
```

This is the main Stage5 result for the thesis, but it should be presented as a
small positive / near-tie result, not as a strong SOTA improvement.

## Mean Metric Summary

All context-FiLM rows use the same core protocol: Stage2 `I60/R20/ohlc_ma_vb`
checkpoint loaded, visual CNN/classifier frozen, bounded last-block FiLM, scale
`0.02`, seeds `42-46`.

| Model | Context | Accuracy | ROC-AUC | AP | Brier | Predicted positive rate |
|---|---|---:|---:|---:|---:|---:|
| Stage2 baseline | none | `0.579320` | `0.584862` | `0.611256` | `0.274337` | `0.664400` |
| N8B F&G-only | F&G raw regime | `0.580291` | `0.584930` | `0.611272` | `0.274004` | `0.660652` |
| 5-9D FinBERT-only | FinBERT sentiment aggregates | `0.578487` | `0.586072` | `0.611943` | `0.272739` | `0.637196` |
| 5-9E FinBERT+F&G | FinBERT + F&G | `0.580569` | `0.585843` | `0.611899` | `0.272701` | `0.639278` |

## Delta Interpretation

Versus Stage2 baseline:

- Accuracy: `+0.001249`.
- ROC-AUC: `+0.000981`.
- AP: `+0.000643`.
- Brier: `-0.001636`.

Versus F&G-only:

- Accuracy: `+0.000278`.
- ROC-AUC: `+0.000913`.
- AP: `+0.000627`.
- Brier: `-0.001304`.

Interpretation: FinBERT+F&G is the best Stage5 row by mean accuracy among the
tested news/sentiment candidates, but the margin is very small. The thesis must
not claim a large performance gain.

## Conditional Correction Result

Against the Stage2 baseline over `7,205` matched decisions:

| Quantity | Value |
|---|---:|
| Corrections | `95` |
| Regressions | `86` |
| Net corrections | `+9` |
| Changed predictions | `181` |
| Changed prediction rate | `2.5121%` |
| Mean probability-up delta | `-0.016158` |

Positive condition buckets:

- Stage2 uncertainty `0.45 <= prob_up <= 0.55`: delta accuracy `+0.012484`.
- F&G greed regime: delta accuracy `+0.010849`, net corrections `+23`.
- High 7-day news count: delta accuracy `+0.007509`.
- High 60-day F&G mean: delta accuracy `+0.008247`.

Negative condition buckets:

- F&G neutral regime: delta accuracy `-0.006751`.
- F&G extreme fear: delta accuracy `-0.004138`.
- Low 60-day news count: delta accuracy `-0.008276`.

This supports a conditional claim: FinBERT+F&G helps most when the visual model
is uncertain or when sentiment/news context is active, and it is weaker in
neutral or low-news regimes.

## Interpretability Result

The targeted 5-12 Grad-CAM/modulation export selected `40` correction/regression
samples and exported `30` Grad-CAM/report artifacts.

Panel-level FiLM modulation:

| Panel | block4 gamma mean | block4 delta-gamma mean | block4 beta mean |
|---|---:|---:|---:|
| Correction | `1.000334` | `0.000334` | `0.000079` |
| Regression | `1.000349` | `0.000349` | `0.000082` |

Probability-level pattern:

| Panel | Mean Stage2 prob_up | Mean Stage5 prob_up | Mean prob_up delta |
|---|---:|---:|---:|
| Correction | `0.523529` | `0.486073` | `-0.037456` |
| Regression | `0.517489` | `0.478454` | `-0.039035` |

Interpretation: FinBERT+F&G does not strongly rewrite the chart representation.
It applies a small bounded final-block calibration. This calibration tends to
lower `prob_up`; it corrects some false-Up cases, but the same mechanism causes
some true-Up regressions.

## What Stage5 Can Claim

Allowed thesis claims:

1. Headline embeddings alone are not enough to beat the strong visual baseline.
2. FinBERT-only sentiment improves ranking/calibration more than hard accuracy.
3. Combining FinBERT sentiment with F&G gives the best Stage5 news-context row,
   but only by a small margin.
4. The main value of news/sentiment context is conditional calibration and
   interpretability, not a large global accuracy jump.
5. The frozen Stage2 + bounded last-block FiLM protocol is stable and preserves
   the strong visual baseline better than scratch-trained context models.

Claims to avoid:

1. Do not claim that LLM/news context strongly outperforms the chart-only model.
2. Do not claim that GPT/Claude reasoning is the final model; the final Stage5
   branch uses OpenAI embeddings as an earlier negative/near-tie result and
   FinBERT/F&G as the strongest tested news/sentiment context.
3. Do not call the task market-regime classification as the primary supervised
   target; the actual target is BTC R20 Up/Down direction.
4. Do not claim that FiLM radically changes visual attention; the measured
   gamma/beta changes are small.

## Thesis Title Decision

The previous title direction, `LLM-to-FiLM Multimodal Fusion for Market Regime
Classification from Price Charts and News`, is too strong for the final
evidence:

- The supervised target is BTC `R20` Up/Down, not a direct market-regime label.
- The strongest Stage5 model is FinBERT+F&G numeric context, not free-form LLM
  reasoning.
- The empirical contribution is bounded context calibration, not large
  multimodal dominance.

Recommended thesis title:

```text
Context-Conditioned FiLM for Bitcoin Direction Prediction from Price Charts
```

Recommended subtitle:

```text
Market Sentiment and News-Derived Signals as Bounded Calibration Context
```

Alternative title if the supervisor wants news/LLM visible in the title:

```text
News- and Sentiment-Conditioned FiLM for Bitcoin Direction Prediction from Price Charts
```

Avoid as final title unless the scope changes:

```text
LLM-to-FiLM Multimodal Fusion for Market Regime Classification from Price Charts and News
```

## Stage5 Closure Decision

Stage5 is closed for the first thesis draft.

Do not run more broad embedding/FinBERT grids before drafting. If the supervisor
asks for one more news experiment later, the only defensible extension is a
small prompt/event-label branch that explicitly separates:

- BTC relevance,
- expected direction,
- event type,
- impact horizon,
- confidence.

For now, the next thesis work should be:

1. Start the English thesis draft v0 using the current Stage1-Stage5 evidence.
2. Run Stage4 `4-N14B2-B6` only as a supplementary derivatives/leverage
   conditional analysis, not as a blocker for drafting.

## Main Evidence Artifacts

- 5-9F comparison:
  `checklist_results/5-9F_stage5_finbert_comparison.md`
- 5-11 conditional analysis:
  `checklist_results/5-11_finbert_fg_condition_analysis.md`
- 5-12 Grad-CAM/modulation export:
  `checklist_results/5-12_gradcam_modulation_export.md`
- 5-11 bucket table:
  `reports/tables/stage5_5_11_finbert_fg_condition_analysis_bucket_summary.csv`
- 5-12 modulation table:
  `reports/tables/stage5_5_12_finbert_fg_targeted_gradcam_modulation_by_panel.csv`
