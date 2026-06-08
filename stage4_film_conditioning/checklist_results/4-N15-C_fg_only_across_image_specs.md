# 4-N15-C F&G-Only Across Image Specs

## Status

Completed.

## Purpose

N15-C tests whether the external F&G market-regime vector helps across all
`I60/R20` chart image specs.

This is deliberately different from N15-B:

```text
N15-B: chart-derived technical context tries to replace missing visual chart information.
N15-C: F&G adds external market-regime information that is not directly drawn from OHLCV.
```

## Fixed Protocol

```text
image_window: 60
return_horizon: 20
image_specs: ohlc, ohlc_ma, ohlc_vb, ohlc_ma_vb
context_method: film_full_bounded_last_block
modulation_scale: 0.02
seeds: 42, 43, 44, 45, 46
visual CNN: frozen
classifier: frozen
trainable modules: context encoder + bounded last-block FiLM heads only
```

Each image spec uses its own Stage 2 checkpoint from N15-A:

```text
ohlc       -> stage2_i60_ohlc_r20
ohlc_ma    -> stage2_i60_ohlc_ma_r20
ohlc_vb    -> stage2_i60_ohlc_vb_r20
ohlc_ma_vb -> stage2_i60_ohlc_ma_vb_r20
```

## Context Features

The same F&G-only vector is used for every image spec:

```text
fg_value
fg_mean_60
fg_delta_60
fg_std_60
```

FiLM is bounded:

```text
gamma = 1 + 0.02 * tanh(raw_gamma)
beta  =     0.02 * tanh(raw_beta)
```

So the model learns sample-specific gamma/beta values from F&G, but the maximum
modulation remains conservative:

```text
gamma: 0.98 to 1.02
beta: -0.02 to 0.02
```

## Runner

Use:

[kaggle_stage4_n15c_fg_only_across_image_specs_one_cell.md](../notebooks/kaggle_stage4_n15c_fg_only_across_image_specs_one_cell.md)

The runner writes:

```text
reports/tables/stage4_n15c_fg_only_across_image_specs_seed_results.csv
reports/tables/stage4_n15c_fg_only_across_image_specs_mean_std_results.csv
reports/tables/stage4_n15c_fg_only_across_image_specs_correction_seed_summary.csv
reports/tables/stage4_n15c_fg_only_across_image_specs_correction_transition_summary.csv
```

and creates:

```text
/kaggle/working/stage4_n15c_fg_only_across_image_specs_result_bundle.zip
```

## Interpretation Rule

Primary comparison:

```text
N15-C(image spec X + F&G) vs Stage2(image spec X)
```

Secondary comparison:

```text
N15-C weaker image + F&G vs Stage2 ohlc_ma_vb
```

Useful outcomes:

```text
1. Same-image improvement:
   F&G improves an image spec over its own Stage 2 baseline.

2. Stronger effect on weaker images:
   F&G helps ohlc/ohlc_ma/ohlc_vb more than ohlc_ma_vb, suggesting external
   regime information can compensate for lower visual richness.

3. Negative but interpretable:
   F&G also preserves rather than improves, suggesting the current bounded
   frozen-FiLM setup is too conservative or the signal is not strong enough for
   hard decision changes.
```

Do not claim improvement from a single seed. Use five-seed mean,
correction/regression, and predicted-positive-rate stability.

## Result Summary

N15-C completed all four image specs over seeds `42,43,44,45,46`.

| Image spec | Stage 2 acc mean | N15-C acc mean | Delta vs same image | Stage 2 ROC-AUC | N15-C ROC-AUC | ROC-AUC delta |
|---|---:|---:|---:|---:|---:|---:|
| `ohlc` | 0.558085 | 0.557668 | -0.000416 | 0.560218 | 0.559828 | -0.000390 |
| `ohlc_ma` | 0.557529 | 0.556697 | -0.000833 | 0.564495 | 0.564479 | -0.000016 |
| `ohlc_vb` | 0.567384 | 0.567939 | +0.000555 | 0.561247 | 0.561033 | -0.000214 |
| `ohlc_ma_vb` | 0.579320 | 0.580291 | +0.000972 | 0.584862 | 0.584930 | +0.000068 |

The two rows with volume bars show small positive same-image accuracy deltas:

```text
ohlc_vb    +0.000555
ohlc_ma_vb +0.000972
```

This is directionally interesting because F&G is an external market-regime
signal and the positive deltas appear on image specs that include volume bars.
However, the size is very small. It should be reported as a weak/tentative
effect, not as a strong performance gain.

## Correction/Regression Check

Same-image decision changes were sparse.

| Image spec | Net correction pattern |
|---|---|
| `ohlc` | neutral to slightly negative |
| `ohlc_ma` | neutral to slightly negative |
| `ohlc_vb` | mildly positive in several seeds |
| `ohlc_ma_vb` | mildly positive, best seed-level net at seed `44` |

For `ohlc_ma_vb`, the seed-level correction summary shows:

```text
seed 42: +1 net correction
seed 43: -2 net correction
seed 44: +6 net corrections
seed 45:  0 net correction
seed 46: +2 net corrections
```

The mean improvement is therefore driven by small decision changes, not by a
large shift in the classifier.

## Comparison With Earlier F&G Runs

The N15-C `ohlc_ma_vb + F&G` row matches the previous N8-B frozen/bounded
F&G-only result:

```text
N8-B / N15-C ohlc_ma_vb:
accuracy mean = 0.580291
ROC-AUC mean  = 0.584930
```

The apparent mismatch with earlier memory comes from comparing against older
non-frozen or unstable F&G experiments:

```text
P7 film_full, not frozen/bounded: accuracy mean 0.552394
P8 bounded but not the final pretrained/frozen protocol: accuracy mean 0.542540
N8-B / N15-C frozen pretrained bounded: accuracy mean 0.580291
```

So N15-C is not contradicting N8-B. It reproduces the same strongest F&G-only
setting for `ohlc_ma_vb`, then extends the comparison to the other image specs.

## Interpretation

The result does not prove a large F&G effect, but it gives a more precise
finding:

```text
F&G as external regime context is safer and slightly more useful when the
visual baseline already includes volume-bar information, especially
ohlc_ma_vb. The improvement is too small to claim a strong model-level gain,
but it is more plausible than chart-derived technical context as a FiLM signal.
```

For thesis reporting, this supports a conservative conclusion:

```text
Frozen bounded FiLM can preserve the strong Stage 2 visual baseline and make
small context-driven corrections. The best evidence remains weak but
directionally consistent for volume-aware image specs.
```
