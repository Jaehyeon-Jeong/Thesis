# 4-N15 Integrated Result Summary

## Status

Completed and organized.

N15 was designed to answer a narrower question than the earlier Stage 4
context experiments:

```text
If Stage 2 visual CNN checkpoints are frozen, can context-FiLM add useful
sample-specific correction on top of each I60/R20 image representation?
```

The protocol uses:

```text
image_window: 60
return_horizon: 20
seeds: 42, 43, 44, 45, 46
visual CNN: Stage 2 pretrained and frozen
classifier: frozen
trainable modules: context encoder + bounded last-block FiLM only
FiLM bound: gamma = 1 + 0.02 * tanh(raw_gamma), beta = 0.02 * tanh(raw_beta)
```

## Result Tables

Key tables are stored in:

```text
reports/tables/stage2_n15a_i60_r20_four_image_specs_five_seed_mean_std_results.csv
reports/tables/stage2_n15a_i60_r20_four_image_specs_five_seed_seed_results.csv
reports/tables/stage4_n15b_image_missing_feature_complement_mean_std_results.csv
reports/tables/stage4_n15b_image_missing_feature_complement_correction_seed_summary.csv
reports/tables/stage4_n15b_image_missing_feature_complement_correction_transition_summary.csv
reports/tables/stage4_n15c_fg_only_across_image_specs_mean_std_results.csv
reports/tables/stage4_n15c_fg_only_across_image_specs_correction_seed_summary.csv
reports/tables/stage4_n15c_fg_only_across_image_specs_correction_transition_summary.csv
```

## N15-A: Stage 2 Four-Image Baselines

N15-A rebuilt and bundled the Stage 2 five-seed checkpoints for all four
`I60/R20` image specs.

| Image spec | Accuracy mean | Accuracy std | ROC-AUC mean | ROC-AUC std |
| --- | ---: | ---: | ---: | ---: |
| `ohlc` | `0.558085` | `0.015183` | `0.560218` | `0.015566` |
| `ohlc_ma` | `0.557529` | `0.023290` | `0.564495` | `0.016252` |
| `ohlc_vb` | `0.567384` | `0.017332` | `0.561247` | `0.018586` |
| `ohlc_ma_vb` | `0.579320` | `0.018218` | `0.584862` | `0.023250` |

Main finding:

```text
ohlc_ma_vb remains the strongest Stage 2 visual baseline.
Volume-aware images are stronger than ohlc/ohlc_ma alone.
```

## N15-B: Missing-Feature Technical Context

N15-B tested whether compact technical context can compensate for visual
information not drawn into the image.

| Image spec | Context set | Stage 2 acc | N15-B acc | Acc delta | Stage 2 ROC-AUC | N15-B ROC-AUC | ROC-AUC delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ohlc` | `technical_all` | `0.558085` | `0.558085` | `+0.000000` | `0.560218` | `0.561056` | `+0.000838` |
| `ohlc_ma` | `volume_volatility` | `0.557529` | `0.557668` | `+0.000139` | `0.564495` | `0.564543` | `+0.000048` |
| `ohlc_vb` | `bb_trend` | `0.567384` | `0.567523` | `+0.000139` | `0.561247` | `0.561120` | `-0.000127` |
| `ohlc_ma_vb` | `technical_all_control` | `0.579320` | `0.579736` | `+0.000416` | `0.584862` | `0.584778` | `-0.000084` |

### Correction/Regression

Counts are aggregated over five seeds:

```text
5 seeds x 1,441 test samples = 7,205 sample-decisions per image spec
```

| Image spec | Context set | Stage2 wrong -> FiLM correct | Stage2 correct -> FiLM wrong | Net correction | Changed decisions | Changed rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ohlc` | `technical_all` | `9` | `9` | `0` | `18` | `0.2498%` |
| `ohlc_ma` | `volume_volatility` | `8` | `7` | `+1` | `15` | `0.2082%` |
| `ohlc_vb` | `bb_trend` | `15` | `14` | `+1` | `29` | `0.4025%` |
| `ohlc_ma_vb` | `technical_all_control` | `11` | `8` | `+3` | `19` | `0.2637%` |

Interpretation:

```text
Technical context preserves the frozen Stage 2 baseline, but it barely changes
hard decisions. It does not recover the performance gap between weaker image
specs and ohlc_ma_vb.
```

This supports the visual-redundancy interpretation:

```text
BB/MFI/RV are derived from OHLCV, so the CNN may already extract much of their
useful information when the chart image is sufficiently rich.
```

## N15-C: F&G External Regime Context

N15-C tested the same frozen bounded FiLM protocol with F&G-only context across
all four image specs.

Context vector:

```text
fg_value
fg_mean_60
fg_delta_60
fg_std_60
```

| Image spec | Stage 2 acc | N15-C acc | Acc delta | Stage 2 ROC-AUC | N15-C ROC-AUC | ROC-AUC delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ohlc` | `0.558085` | `0.557668` | `-0.000416` | `0.560218` | `0.559828` | `-0.000390` |
| `ohlc_ma` | `0.557529` | `0.556697` | `-0.000833` | `0.564495` | `0.564479` | `-0.000016` |
| `ohlc_vb` | `0.567384` | `0.567939` | `+0.000555` | `0.561247` | `0.561033` | `-0.000214` |
| `ohlc_ma_vb` | `0.579320` | `0.580291` | `+0.000972` | `0.584862` | `0.584930` | `+0.000068` |

### Correction/Regression

Counts are aggregated over five seeds:

| Image spec | Context set | Stage2 wrong -> FiLM correct | Stage2 correct -> FiLM wrong | Net correction | Changed decisions | Changed rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ohlc` | `fg_only` | `8` | `11` | `-3` | `19` | `0.2637%` |
| `ohlc_ma` | `fg_only` | `12` | `18` | `-6` | `30` | `0.4164%` |
| `ohlc_vb` | `fg_only` | `23` | `19` | `+4` | `42` | `0.5829%` |
| `ohlc_ma_vb` | `fg_only` | `21` | `14` | `+7` | `35` | `0.4858%` |

Interpretation:

```text
F&G helps only the volume-aware image specs, especially ohlc_ma_vb.
The effect is directionally positive but very small.
```

The most useful result is:

```text
ohlc_ma_vb + F&G:
accuracy +0.000972
ROC-AUC +0.000068
net correction +7 over 7,205 decisions
```

This is the same conclusion as the earlier N8-B frozen/bounded F&G result:

```text
F&G is safer than chart-derived technical context because it is external
market-regime information, but the current bounded FiLM correction is too
small to claim a strong performance gain.
```

## Overall N15 Conclusion

N15 produced a clean and useful result, but not a large model improvement.

1. `ohlc_ma_vb` remains the strongest image baseline.
2. Technical OHLCV-derived context does not materially improve weaker images.
3. F&G gives the best context behavior when the image already includes volume.
4. The frozen bounded FiLM design is stable and preserves the baseline.
5. The decision-change rate is very low, usually below `0.6%`, so the model is
   mostly making conservative probability-level corrections rather than
   changing many labels.

For thesis reporting, the balanced claim is:

```text
Stage 4 context-FiLM does not substantially outperform the strongest Stage 2
visual baseline, but frozen bounded FiLM can preserve that baseline and make
small, interpretable corrections. Among tested context sources, external
F&G regime context is more plausible than chart-derived technical summaries,
especially with volume-aware chart images.
```

## Decision For Next Work

N15 does not justify running a larger technical-context grid.

The next useful direction is:

```text
N16: derivatives/leverage context
```

because funding, futures activity, and open interest are more distinct from the
chart image than BB/MFI/RV and are closer to crypto-specific market regime.
