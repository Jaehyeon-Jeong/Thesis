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

## Ablation Axes

### A. Same Equation, Larger Scale

Use the existing bounded last-block FiLM implementation first. This requires no
new model code:

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

### B. Relax Gamma/Beta Constraint

Run only if the bounded scale grid does not collapse:

```text
unbounded or weakly regularized last-block FiLM
zero-initialized gamma/beta heads
regularization on ||gamma - 1|| and ||beta||
```

This requires new model code. It should be reported as a FiLM constraint
ablation, not as a new context-source experiment.

### C. Alternative Gamma/Beta Equation

Compare the current bounded residual rule against another
baseline-preserving FiLM parameterization:

```text
current:
gamma = 1 + scale * tanh(raw_gamma)
beta  =     scale * tanh(raw_beta)

candidate 1: positive-gamma sigmoid rule
gamma = 2 * sigmoid(scale * raw_gamma)
beta  = scale * tanh(raw_beta)

candidate 2: weakly regularized residual-linear rule
gamma = 1 + scale * raw_gamma
beta  = scale * raw_beta
loss += lambda * ||gamma - 1||^2 + lambda * ||beta||^2
```

Use only one alternative equation first. The goal is not to try every possible
FiLM formula, but to check whether the current `tanh` saturation is too
restrictive.

### D. Classifier-Unfreeze Variant

Keep the Stage 2 visual CNN frozen, but unfreeze the final classifier:

```text
visual CNN: frozen
classifier: trainable
context encoder: trainable
FiLM heads: trainable
```

This tests whether the context-FiLM branch can help if the final decision
boundary is allowed to adapt, while still preserving the Stage 2 visual feature
extractor. It should be reported as a freeze-policy ablation.

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

## If Overall Metrics Still Do Not Improve

Do not keep expanding random architecture variants. Use the same outputs for
conditional analysis:

```text
extreme F&G / FSI / RORO regime
high-volatility or high-stress regime
Stage 2 wrong -> FiLM correct samples
Stage 2 correct -> FiLM wrong samples
high-confidence Stage 2 cases vs low-confidence Stage 2 cases
```

This can support a narrower thesis claim:

```text
context-FiLM does not materially improve average BTC direction accuracy over a
strong chart baseline, but it can be inspected as a conditional correction path
for specific market-regime regimes.
```
