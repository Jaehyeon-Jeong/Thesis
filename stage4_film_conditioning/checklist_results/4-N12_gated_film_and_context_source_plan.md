# 4-N12 Gated FiLM And Context-Source Plan

Status: N12-A completed, N12-B prepared.

## Why N12

N10-B showed that the N10 news-FiLM model mostly applies a small Down/risk-off
correction near the Stage 2 decision boundary:

```text
Stage2 wrong -> N10 correct: 27
Stage2 correct -> N10 wrong: 24
Net correction: +3 / 7205
```

This means the next experiment should not randomly increase FiLM scale. The
model should use the interpretation: context correction should depend on the
frozen Stage 2 baseline confidence.

## N12-A. Uncertainty-Gated News FiLM

Purpose:

```text
Let context-FiLM intervene more when the Stage 2 chart model is uncertain.
```

Candidate gate:

```text
uncertainty = 4 * prob_up_stage2 * (1 - prob_up_stage2)
```

Candidate FiLM:

```text
gamma = 1 + uncertainty * scale * tanh(raw_gamma)
beta  =     uncertainty * scale * tanh(raw_beta)
```

Initial setup:

```text
image: I60/R20/ohlc_ma_vb
backbone: Stage 2 CNN frozen
classifier: Stage 2 classifier frozen
context: news SVD32 + news-count features
scale candidates: 0.02, 0.05
seeds: 42, 43, 44, 45, 46
```

## N12-B. Confidence-Gated News FiLM

Purpose:

```text
Test whether context can strengthen high-confidence Stage 2 visual evidence.
```

Candidate gate:

```text
confidence = abs(2 * prob_up_stage2 - 1)
```

Candidate FiLM:

```text
gamma = 1 + confidence * scale * tanh(raw_gamma)
beta  =     confidence * scale * tanh(raw_beta)
```

Risk:

```text
If Stage 2 is confidently wrong, confidence gating may strengthen the wrong
class. Evaluate correction/regression counts, not accuracy alone.
```

## N12-C. Technical-Only Frozen FiLM Ablation

Purpose:

```text
Separate image-derived technical context from image-external context.
```

Candidate features:

```text
bb_percent_b_60
bb_bandwidth_60
mfi_60
rv_60
```

Expected interpretation:

```text
The gain may be small because ohlc_ma_vb already contains MA/volume/shape
information, but this ablation makes the Stage 4 context-source comparison
cleaner.
```

## N12-D. Context-Source Comparison

Compare under the same frozen Stage 2 bounded-FiLM protocol:

```text
F&G-only
news-only
technical-only
news + F&G
```

Metrics:

- accuracy,
- ROC-AUC,
- Brier score,
- F1,
- predicted-Up rate,
- correction count,
- regression count,
- net correction.

## Result Upload Policy

For every N12 substep:

- Commit and push the updated checklist before running Kaggle if code changes.
- After Kaggle, save the downloaded local result bundle path in the matching
  checklist result file.
- Commit only compact tables, JSON summaries, result notes, and runnable
  notebooks.
- Do not commit large output bundles, checkpoints, or repeated full prediction
  CSVs unless they are intentionally small and needed for a table.
- If a result table already exists for the same experiment, update the existing
  note instead of creating a duplicate result document.

## N13. Sentiment/Event Feature Extension

Run only if the TF-IDF/SVD headline representation is too weak or too hard to
interpret.

Candidate features:

```text
headline sentiment score
positive / negative / neutral counts
crypto regulation / exchange / ETF / macro event tags
```

Leakage rule:

```text
Only headlines available by strict t-1 may be used.
Record encoder/model/version/date/cache hash.
```

## N14. Final Stage 4 Interpretability Report

Purpose:

```text
Turn the selected Stage 4 model into thesis-ready evidence.
```

Required sections:

- Stage 2 baseline vs selected context-FiLM metrics;
- correction/regression table;
- predicted-Up distribution;
- targeted Stage 2 vs Stage 4 Grad-CAM;
- gamma/beta/modulation-gate summaries;
- representative `Stage2 wrong -> Stage4 correct` and
  `Stage2 correct -> Stage4 wrong` samples.
