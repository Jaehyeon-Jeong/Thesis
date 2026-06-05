# 4-N9 News Pretrained/Frozen FiLM Design

Status: N9-A completed; SVD/scale grid runner prepared

## Goal

Use the stable N8-B structure to test whether headline-news context can add
incremental signal to the selected Stage 2 visual baseline.

The fixed baseline remains:

```text
Stage 2 I60/R20/ohlc_ma_vb
five-seed accuracy mean = 0.579320
five-seed ROC-AUC mean  = 0.584863
```

N8-B showed that the baseline-preserving structure is stable:

```text
Stage 2 CNN/classifier frozen
F&G-only context -> MLP -> bounded last-block FiLM
scale 0.02 accuracy mean = 0.580291
scale 0.02 ROC-AUC mean  = 0.584930
```

N9 now asks:

```text
Does richer external news context produce a stronger correction than F&G-only
when attached through the same pretrained/frozen FiLM path?
```

## Prepared Runner

Kaggle one-cell runner:

```text
notebooks/kaggle_stage4_n9_news_pretrained_frozen_bounded_film_one_cell.md
```

Default setting:

```text
N9_VARIANT = "N9A"
news SVD8-only
Stage 2 CNN frozen
Stage 2 classifier frozen
bounded last-block FiLM scale = 0.02
seeds = 42, 43, 44, 45, 46
```

## N9-A Result and Grid Follow-Up

N9-A completed with five seeds:

```text
news SVD8-only
Stage 2 CNN frozen
Stage 2 classifier frozen
bounded last-block FiLM scale = 0.02
accuracy mean = 0.579459
ROC-AUC mean  = 0.585670
```

Interpretation:

```text
N9-A is stable and preserves the Stage 2 baseline, but the correction is very
small. It slightly improves ROC-AUC versus the reloaded Stage 2 baseline, while
accuracy remains almost unchanged.
```

Prepared follow-up grid:

```text
notebooks/kaggle_stage4_n9_news_pretrained_frozen_svd_scale_grid_one_cell.md
```

Grid points:

```text
SVD8  / scale 0.05
SVD16 / scale 0.02
SVD16 / scale 0.05
SVD32 / scale 0.02
SVD32 / scale 0.05
```

Excluded:

```text
SVD8 / scale 0.02
```

Reason:

```text
This point is already covered by N9-A. The grid tests whether N9-A was too
conservative because the FiLM scale was too small or because SVD8 compressed
headline context too aggressively.
```

Grid policy:

```text
Keep Stage 2 CNN and classifier frozen.
Keep bounded last-block FiLM.
Disable Grad-CAM by default during the grid.
Run Grad-CAM only for the selected best grid configuration afterward.
```

The same runner can be reused for follow-ups by changing only `N9_VARIANT`:

```text
N9B = news-only, classifier frozen, scale 0.05
N9C = news-only, classifier trainable, scale 0.02
N9D = news-only, classifier trainable, scale 0.05
```

N9-E, news + F&G combined context, is intentionally left as a follow-up after
news-only results. Running N9-E first would mix two questions: whether news
helps and whether F&G helps next to news.

## Important Gamma/Beta Rule

Do not manually set sample-specific gamma/beta values.

Wrong experimental logic:

```text
This date has negative news, so manually lower gamma for channel 10.
This date has extreme greed, so manually raise beta for channel 3.
```

That would turn the experiment into hand-built rules and would weaken the
thesis claim.

Correct experimental logic:

```text
context vector -> MLP -> raw_gamma/raw_beta
gamma = 1 + scale * tanh(raw_gamma)
beta  =     scale * tanh(raw_beta)
```

The model learns the mapping from news/F&G context to gamma/beta. The researcher
only controls:

- which context vector is used,
- where FiLM is inserted,
- which pretrained weights are frozen,
- and how large the correction is allowed to be through `scale`.

Interpretation comes after training:

```text
same sample
Stage 2 Grad-CAM vs N9 Grad-CAM
news/F&G context values
learned gamma/beta summary
prediction change
```

## Fixed N9 Architecture Family

Use the same baseline-preserving family as N8-B:

```text
chart image -> Stage 2 pretrained CNN visual blocks frozen -> chart feature
context vector -> MLP -> gamma/beta
chart feature -> bounded last-block FiLM correction
classifier -> Up/Down logits
```

The CNN visual blocks stay frozen in all N9 primary experiments. This preserves
the Stage 2 chart representation and avoids the scratch-FiLM failure mode.

Weak correction setting:

- Stage 2 classifier.
- Frozen BatchNorm/dropout behavior is kept in eval mode during Stage 4
  training.

Trainable modules in weak correction:

- context encoder,
- `gamma_head`,
- `beta_head`.

Medium correction setting:

- Stage 2 visual CNN blocks stay frozen.
- The final classifier is trainable.
- Context encoder, `gamma_head`, and `beta_head` are trainable.

Rationale:

```text
The Stage 2 classifier learned to read unmodulated chart features. If news-FiLM
slightly changes those features, a frozen classifier may be too restrictive.
Opening only the classifier lets the model learn how to read FiLM-adjusted
features without changing the visual chart filters.
```

Strong correction setting:

```text
final CNN block partial-unfreeze or full scratch training
```

This is not today's main path because previous strong/scratch FiLM variants
were unstable and often collapsed.

## Context Candidates

### N9-A. News-only, SVD8, weak correction, scale 0.02

Primary run.

```text
context = news_svd_7d/20d/60d + log news-count features
news vector = SVD8 per window
context dim = 30
CNN = frozen
classifier = frozen
scale = 0.02
seeds = 42, 43, 44, 45, 46
```

Why first:

```text
N6.1 showed SVD8 had the best news ROC-AUC signal among SVD dimensions.
N8-B showed scale 0.02 is the safest F&G-only correction.
```

### N9-B. News-only, SVD8, weak correction, scale 0.05

Run only if N9-A is stable but too conservative.

Purpose:

```text
Test whether slightly larger news correction improves accuracy/ROC-AUC without
causing predicted-positive-rate collapse.
```

### N9-C. News-only, SVD8, medium correction, scale 0.02

Run if N9-A/B preserve the baseline but do not improve enough.

```text
context = news_svd_7d/20d/60d + log news-count features
CNN = frozen
classifier = trainable
scale = 0.02
seeds = 42, 43, 44, 45, 46
```

Purpose:

```text
Test whether the FiLM-adjusted feature needs a lightly adapted classifier while
still preserving the Stage 2 visual filters.
```

### N9-D. News-only, SVD8, medium correction, scale 0.05

Run only if N9-C is stable but too weak.

```text
context = news_svd_7d/20d/60d + log news-count features
CNN = frozen
classifier = trainable
scale = 0.05
seeds = 42, 43, 44, 45, 46
```

### N9-E. News + F&G combined, weak or medium correction, scale 0.02

Run after news-only results, preferably only if N9-A/B/C/D show that the
pretrained/frozen news path is stable or if an advisor-facing combined-context
comparison is needed.

```text
context = news SVD8 features + news count features + F&G-only features
F&G features = fg_value, fg_mean_60, fg_delta_60, fg_std_60
scale = 0.02
seeds = 42, 43, 44, 45, 46
```

Purpose:

```text
Test whether F&G works as a compact market-regime summary next to richer
headline-news context.
```

## Evaluation Criteria

Primary comparison:

```text
Stage 2 baseline I60/R20/ohlc_ma_vb
vs
N8-B F&G-only scale 0.02
vs
N9 news-only weak scale 0.02/0.05
vs
N9 news-only medium scale 0.02/0.05
vs
N9 news+F&G scale 0.02
```

Metrics:

- accuracy mean/std,
- ROC-AUC mean/std,
- average precision,
- F1,
- Brier score,
- predicted-positive-rate mean/std,
- long/flat Sharpe net,
- long/short Sharpe net.

Collapse guard:

```text
Flag any seed where predicted_positive_rate is extremely low or high.
Use this as a stability diagnostic, not as a post-hoc threshold-tuning trick.
```

Promotion rule:

```text
Promote N9 only if it is competitive with Stage 2 and avoids collapse.
Small accuracy improvement alone is not enough; ROC-AUC and seed stability must
also be defensible.
```

## Interpretability Plan

Interpretability should use matched samples:

```text
same date
same chart image
same true label
Stage 2 prediction and Grad-CAM
N8-B/N9 prediction and Grad-CAM
context vector
gamma/beta summary
```

Most important sample group:

```text
Stage 2 wrong -> N9 correct
```

For those samples, inspect:

- headline window terms/titles,
- news-count features,
- F&G features when combined context is used,
- gamma/beta magnitude and direction,
- whether Grad-CAM attention shifts to a defensible chart region.

This supports the thesis argument:

```text
FiLM does not make the whole CNN transparent, but it provides an explicit path
showing how market/news context modulates learned visual chart features.
```

## Today Decision

Run order for the final experiment pass:

```text
1. N9-A news-only SVD8, weak correction, scale 0.02, five seeds
2. N9-B news-only SVD8, weak correction, scale 0.05 if N9-A is stable but weak
3. N9-C news-only SVD8, medium correction, classifier trainable, scale 0.02
4. N9-D news-only SVD8, medium correction, classifier trainable, scale 0.05
   only if N9-C is stable but weak
5. N9-E news + F&G scale 0.02 only if news-only is promising or needed for the
   advisor-facing comparison
6. Generate Stage 2 vs N8-B/N9 interpretability comparison
7. Finalize Stage 4 report and advisor update
```

## Optional Follow-Ups After Interpretation

These are not first-pass N9 runs. They should be triggered only by the N9/N10
interpretation results.

### 4-N12. Uncertainty-Gated FiLM

Run only if matched-sample analysis suggests that context helps mostly when the
Stage 2 chart model is uncertain.

Idea:

```text
high-confidence Stage 2 chart decision -> very small context correction
ambiguous Stage 2 chart decision       -> larger context correction allowed
```

Candidate uncertainty:

```text
uncertainty = 4 * prob_up_stage2 * (1 - prob_up_stage2)
```

Candidate modulation:

```text
gamma = 1 + uncertainty * scale * tanh(raw_gamma)
beta  =     uncertainty * scale * tanh(raw_beta)
```

This keeps the experiment model-driven. The researcher does not manually change
gamma/beta for individual samples; the uncertainty gate is a general rule based
on the frozen Stage 2 baseline confidence.

### 4-N13. Sentiment/Event Feature Extension

Run only if headline TF-IDF/SVD is too weak or difficult to interpret.

Candidate context additions:

- headline sentiment score,
- positive/negative/neutral headline counts,
- regulation/exchange/ETF/macro event tags,
- cached headline-level sentiment/event labels.

Leakage rule:

```text
Use only headlines available by strict t-1.
Record encoder/model/version/date/cache hash.
```

Purpose:

```text
Test whether explicit news polarity or event type is more useful for FiLM
correction than unsupervised TF-IDF/SVD vectors.
```
