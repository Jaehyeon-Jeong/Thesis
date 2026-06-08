# 4-N15 I60/R20 Image-Spec Context Complement Plan

## Status

Planned.

## Purpose

Stage 4 has mostly tested context-FiLM on the strongest Stage 2 image baseline:

```text
I60/R20/ohlc_ma_vb
```

That image already contains OHLC, moving-average, and volume-bar information.
Therefore, technical context such as Bollinger, MFI, and realized volatility may
be redundant.

N15 changes the question:

```text
If an image spec is missing part of the chart information, can context-FiLM
recover part of that missing information?
```

This is a more thesis-useful comparison than simply adding more context to the
already strongest image.

## Fixed Protocol

Keep the model protocol fixed:

```text
image_window: 60
return_horizon: 20
image specs: ohlc, ohlc_ma, ohlc_vb, ohlc_ma_vb
Stage 2 checkpoint: seed-matched checkpoint for the same image spec
visual CNN: frozen
classifier: frozen
FiLM: bounded last-block full FiLM
FiLM scale: 0.02
seeds: 42, 43, 44, 45, 46
```

Important: each image spec must use its own Stage 2 checkpoint. Do not reuse the
`ohlc_ma_vb` checkpoint for `ohlc`, `ohlc_ma`, or `ohlc_vb`.

## Stage 2 Reference

Five-seed Stage 2 baselines for `I60/R20`:

| Image spec | Accuracy mean | ROC-AUC mean | Interpretation |
| --- | ---: | ---: | --- |
| `ohlc_ma_vb` | `0.579320` | `0.584862` | strongest, most information-rich image |
| `ohlc_vb` | `0.567384` | `0.561247` | volume bars help strongly |
| `ohlc` | `0.558085` | `0.560218` | plain chart baseline |
| `ohlc_ma` | `0.557529` | `0.564495` | MA alone does not match volume-bar variants |

## N15-A. Same-Image Stage 2 Reload

Reload and verify the frozen Stage 2 baseline for all four image specs:

```text
I60/R20/ohlc
I60/R20/ohlc_ma
I60/R20/ohlc_vb
I60/R20/ohlc_ma_vb
```

Purpose:

```text
Establish the exact same-image baseline for every later FiLM comparison.
```

Primary comparison:

```text
context-FiLM(image spec X) vs Stage2(image spec X)
```

Secondary comparison:

```text
gap to strongest Stage2 ohlc_ma_vb baseline
```

## N15-B. Image-Missing-Feature Complement FiLM

This comes before F&G-across-all-images. It directly tests whether context
features can replace information that was not drawn into the chart image.

Use only already implemented Stage 4 features first.

| Image spec | Missing or weak image information | Context feature set | Feature names |
| --- | --- | --- | --- |
| `ohlc` | MA/trend, band position, volume-aware pressure, volatility | `technical_all` | `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60` |
| `ohlc_ma` | volume-aware pressure and volatility/volume-bar proxy | `volume_volatility` | `mfi_60`, `rv_60` |
| `ohlc_vb` | moving-average/band trend context | `bb_trend` | `bb_percent_b_60`, `bb_bandwidth_60` |
| `ohlc_ma_vb` | little technical information is missing | `technical_all_control` | `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60` |

Interpretation:

```text
If ohlc_ma + volume_volatility moves toward ohlc_ma_vb,
then volume-derived context can partially substitute for visual volume bars.

If it does not, then volume bars are more useful when encoded directly in the
chart image than summarized as a small context vector.
```

The `ohlc_ma_vb + technical_all_control` row is expected to be weak or neutral.
It checks the redundancy hypothesis on the strongest image.

## N15-C. F&G-Only Across All Image Specs

After N15-B, attach the same compact external regime vector to every image spec:

```text
fg_value
fg_mean_60
fg_delta_60
fg_std_60
```

Purpose:

```text
Check whether image-external market regime information helps regardless of
which chart information is drawn into the image.
```

This is different from N15-B:

```text
N15-B: context replaces missing chart-derived information.
N15-C: context adds external market-regime information.
```

## N15-D. Selected Hybrid Only If B/C Show Signal

Do not run a large all-context vector by default.

Run selected hybrid rows only if N15-B or N15-C shows a useful same-image gain:

```text
ohlc + technical_all + F&G
ohlc_ma + volume_volatility + F&G
ohlc_vb + bb_trend + F&G
```

Skip N15-D if N15-B/C only preserve the baseline or increase regressions.

## Metrics

Report:

```text
accuracy
ROC-AUC
Brier score
predicted_positive_rate
same-image Stage2 delta
gap-to-ohlc_ma_vb-baseline delta
Stage2 wrong -> FiLM correct
Stage2 correct -> FiLM wrong
net correction
```

## Decision Rule

Keep an N15 result only if it provides one of these thesis-useful outcomes:

```text
1. Same-image improvement:
   context-FiLM improves its own image baseline.

2. Gap closing:
   weaker image + context moves closer to ohlc_ma_vb baseline.

3. Negative but interpretable result:
   direct visual encoding is stronger than context-summary replacement.
```

Do not claim improvement from a single seed. Use five-seed mean and
correction/regression behavior.

## Expected Thesis Value

N15 can support a stronger final discussion:

```text
Context-FiLM did not materially improve the strongest ohlc_ma_vb baseline, but
it was also tested as a controlled replacement for missing chart information in
weaker image specs. This separates "context is useless" from "context is
redundant with what the image already contains."
```
