# 4-N13-7 Final FiLM Constraint And Scale Ablation Plan

## Purpose

This is the final modeling ablation after the Stage 4 context-source comparison.
It does not introduce a new context source. It tests whether the current
baseline-preserving FiLM correction was too conservative under the Stage 2
frozen protocol.

Main question:

```text
After freezing the strong Stage 2 chart CNN and classifier, can a larger or
less constrained FiLM modulation improve correction without class collapse?
```

## Fixed Setup

The experiment must keep these conditions fixed:

```text
image setting: I60/R20/ohlc_ma_vb
visual model: Stage 2 seed-matched checkpoint
visual backbone: frozen
classifier: frozen
context source: best stable source selected after 4-N13-5/6
seeds: 42, 43, 44, 45, 46
split: same Stage 2/Stage 4 test split
```

The likely first candidate is still `F&G-only`, because it is the current best
compact accuracy candidate under the frozen protocol. However, if N13-4 RORO or
N13-5 comparison finds a stronger stable source, that source should be used
instead.

## Why The Current Constraint Was Used

Current bounded FiLM:

```text
gamma = 1 + scale * tanh(raw_gamma)
beta  =     scale * tanh(raw_beta)
```

Reasoning:

```text
raw_gamma = 0 and raw_beta = 0 gives gamma = 1, beta = 0.
So the model starts from the Stage 2 baseline identity path.

tanh bounds the modulation, so scale gives an explicit maximum correction size.
For example, scale 0.05 means roughly +/-5% gamma adjustment.

This reduces the risk that a small context branch destroys strong chart
features or collapses predictions into one class.
```

This is a defensible baseline-preserving design, but it may be too conservative
after the visual backbone and classifier are frozen.

## Primary Grid

Use the existing bounded last-block FiLM implementation first:

```text
scale = 0.02
scale = 0.05
scale = 0.10
scale = 0.20
```

Interpretation:

```text
0.02: very conservative correction
0.05: conservative correction
0.10: moderate correction
0.20: strong correction
```

Do not go beyond `0.20` unless the grid is clearly stable. The purpose is not to
search random numbers, but to test whether larger allowed gamma/beta movement
helps once Stage 2 is fixed.

## Optional Relaxed-Constraint Grid

Run only if the bounded scale grid does not collapse:

```text
unbounded or weakly regularized last-block FiLM
zero-initialized gamma/beta heads
regularization on ||gamma - 1|| and ||beta||
```

This requires new model code. It should be reported as a FiLM constraint
ablation, not as a new context-source experiment.

## Required Metrics

Report more than accuracy:

```text
accuracy
ROC-AUC
Brier score
predicted_positive_rate
collapse warning
Stage2 wrong -> FiLM correct
Stage2 correct -> FiLM wrong
net correction
gamma/beta magnitude summary
```

## Decision Rule

Keep a larger scale or relaxed-constraint model only if:

```text
it improves accuracy or ROC/Brier,
does not collapse predicted class distribution,
does not create a large increase in Stage2-correct -> FiLM-wrong regressions,
and has interpretable gamma/beta movement.
```

If large scales do not help, keep the current bounded small-scale FiLM as the
more defensible thesis model.
