# 4-N15-B Image-Missing-Feature Complement FiLM

## Status

Completed.

## Purpose

N15-B tests whether context-FiLM can compensate for information that was not
drawn into each chart image spec.

This is not another all-context run on the strongest image. It is a controlled
same-image comparison:

```text
Stage2(image spec X) vs Stage2(image spec X frozen) + matching context-FiLM
```

## Fixed Protocol

```text
image_window: 60
return_horizon: 20
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

## Feature Mapping

| Image spec | Context feature set | Features |
| --- | --- | --- |
| `ohlc` | `technical_all` | `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60` |
| `ohlc_ma` | `volume_volatility` | `mfi_60`, `rv_60` |
| `ohlc_vb` | `bb_trend` | `bb_percent_b_60`, `bb_bandwidth_60` |
| `ohlc_ma_vb` | `technical_all_control` | `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60` |

## Runner

Use:

[kaggle_stage4_n15b_image_missing_feature_complement_film_one_cell.md](../notebooks/kaggle_stage4_n15b_image_missing_feature_complement_film_one_cell.md)

The runner writes:

```text
reports/tables/stage4_n15b_image_missing_feature_complement_seed_results.csv
reports/tables/stage4_n15b_image_missing_feature_complement_mean_std_results.csv
reports/tables/stage4_n15b_image_missing_feature_complement_correction_seed_summary.csv
reports/tables/stage4_n15b_image_missing_feature_complement_correction_transition_summary.csv
```

and creates a resume/download bundle:

```text
/kaggle/working/stage4_n15b_image_missing_feature_complement_result_bundle.zip
```

## Interpretation Rule

Primary comparison:

```text
N15-B row vs the same-image Stage 2 baseline.
```

Secondary comparison:

```text
N15-B row vs the strongest Stage 2 ohlc_ma_vb baseline.
```

Useful outcomes:

```text
1. Same-image improvement: context-FiLM improves its own image baseline.
2. Gap closing: weaker image + context moves closer to ohlc_ma_vb.
3. Negative but interpretable: direct visual encoding is stronger than compact
   context-summary replacement.
```

Do not claim improvement from a single seed. Use five-seed mean and
correction/regression behavior.

## Result Summary

Local result folder:

```text
/Users/jaehyeonjeong/Desktop/논문/N15b_results
```

All 20 runs completed:

```text
4 image specs x 5 seeds = 20 ok runs
```

The intended frozen protocol was also confirmed:

```text
pretrained Stage 2 checkpoint loaded: True
freeze visual backbone: True
freeze classifier: True
trainable parameters: about 35k context/FiLM parameters
frozen parameters: about 2.95M Stage 2 parameters
```

### Mean/Std Results

| Image spec | Context set | Stage 2 acc | N15-B acc | Acc delta | Stage 2 ROC-AUC | N15-B ROC-AUC | ROC-AUC delta | Gap to `ohlc_ma_vb` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ohlc_ma_vb` | `technical_all_control` | `0.579320` | `0.579736` | `+0.000416` | `0.584862` | `0.584778` | `-0.000084` | `+0.000416` |
| `ohlc_vb` | `bb_trend` | `0.567384` | `0.567523` | `+0.000139` | `0.561247` | `0.561120` | `-0.000127` | `-0.011797` |
| `ohlc` | `technical_all` | `0.558085` | `0.558085` | `0.000000` | `0.560218` | `0.561056` | `+0.000838` | `-0.021235` |
| `ohlc_ma` | `volume_volatility` | `0.557529` | `0.557668` | `+0.000139` | `0.564495` | `0.564543` | `+0.000048` | `-0.021652` |

### Correction/Regression

| Image spec | Context set | Corrections | Regressions | Net correction | Changed decision rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `ohlc_ma_vb` | `technical_all_control` | `11` | `8` | `+3` | `0.2637%` |
| `ohlc_vb` | `bb_trend` | `15` | `14` | `+1` | `0.4025%` |
| `ohlc` | `technical_all` | `9` | `9` | `0` | `0.2498%` |
| `ohlc_ma` | `volume_volatility` | `8` | `7` | `+1` | `0.2082%` |

## Interpretation

N15-B produced valid outputs, but it did not produce a material improvement.

The reason is not an execution failure. The model changed very few hard
decisions. Across five seeds, only about `0.2%` to `0.4%` of test predictions
changed in each row. Therefore the final accuracy stays almost identical to the
frozen Stage 2 baseline.

This supports the redundancy/direct-encoding interpretation:

```text
BB/MFI/RV-style technical summaries do not recover the performance gap between
weaker images and ohlc_ma_vb.
```

In particular:

```text
ohlc + technical_all remains about 2.12 percentage points below ohlc_ma_vb.
ohlc_ma + volume_volatility remains about 2.17 percentage points below ohlc_ma_vb.
ohlc_vb + bb_trend remains about 1.18 percentage points below ohlc_ma_vb.
```

So the visual encoding itself appears stronger than adding these technical
signals later as a compact context vector.

## Decision

N15-B is useful as a negative/control result:

```text
Technical context summaries preserve the frozen Stage 2 baseline, but they do
not materially improve or replace the missing visual information.
```

Next, N15-C is still meaningful because it asks a different question:

```text
F&G is external market-regime information, not a direct OHLCV-derived technical
summary.
```

Run N15-C only as a compact external-regime comparison across the same four
image specs.
