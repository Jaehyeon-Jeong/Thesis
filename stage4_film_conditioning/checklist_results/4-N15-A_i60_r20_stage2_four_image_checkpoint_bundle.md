# 4-N15-A I60/R20 Four-Image Stage 2 Checkpoint Bundle

## Status

Completed and verified.

## Purpose

N15 tests whether context-FiLM can complement information missing from weaker
chart image specs. Before adding context, every image spec needs its own frozen
Stage 2 baseline checkpoint.

N15-A rebuilds and bundles the Stage 2 checkpoints for:

```text
I60/R20/ohlc
I60/R20/ohlc_ma
I60/R20/ohlc_vb
I60/R20/ohlc_ma_vb
```

with seeds:

```text
42, 43, 44, 45, 46
```

This is not a new Stage 4 model. It creates the exact Stage 2 visual baselines
that N15-B/C will freeze.

## Why This Is Required

Each image spec has a different visual input:

```text
ohlc       -> price bars only
ohlc_ma    -> price bars + moving average
ohlc_vb    -> price bars + volume bars
ohlc_ma_vb -> price bars + moving average + volume bars
```

Therefore each image spec must use a checkpoint trained on that same image
spec. Reusing the `ohlc_ma_vb` checkpoint for `ohlc` or `ohlc_ma` would be
invalid because the learned convolution filters are tied to the visual encoding.

## Kaggle Runner

Use:

[kaggle_stage4_n15a_rebuild_i60_r20_four_image_stage2_checkpoints_one_cell.md](../notebooks/kaggle_stage4_n15a_rebuild_i60_r20_four_image_stage2_checkpoints_one_cell.md)

The runner trains/evaluates:

```text
4 image specs x 5 seeds = 20 Stage 2 runs
```

and writes:

```text
reports/tables/stage2_n15a_i60_r20_four_image_specs_five_seed_seed_results.csv
reports/tables/stage2_n15a_i60_r20_four_image_specs_five_seed_mean_std_results.csv
reports/tables/stage2_n15a_i60_r20_four_image_specs_checkpoint_bundle_manifest.json
```

It also creates:

```text
/kaggle/working/stage2_i60_r20_four_image_specs_seed42_46_checkpoints_for_stage4_n15.zip
```

## Expected Reference

Existing Stage 2 five-seed results:

| Image spec | Accuracy mean | ROC-AUC mean |
| --- | ---: | ---: |
| `ohlc_ma_vb` | `0.579320` | `0.584862` |
| `ohlc_vb` | `0.567384` | `0.561247` |
| `ohlc` | `0.558085` | `0.560218` |
| `ohlc_ma` | `0.557529` | `0.564495` |

N15-A should reproduce these values within normal deterministic tolerance.

## Result Check

Local bundle checked:

```text
/Users/jaehyeonjeong/Desktop/논문/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15
```

The N15-A mean/std table exactly reproduces the existing Stage 2 I60/R20
four-image five-seed results:

| Image spec | N15-A accuracy mean | Existing Stage 2 accuracy mean | Delta |
| --- | ---: | ---: | ---: |
| `ohlc` | `0.558085` | `0.558085` | `0.0` |
| `ohlc_ma` | `0.557529` | `0.557529` | `~1e-16` |
| `ohlc_vb` | `0.567384` | `0.567384` | `0.0` |
| `ohlc_ma_vb` | `0.579320` | `0.579320` | `0.0` |

Checkpoint count:

```text
4 image specs x 5 seeds = 20 best.pt files
```

Conclusion: the bundle is sufficient for N15-B/C. If Kaggle resets, upload this
folder or zip as a dataset and set `STAGE2_CHECKPOINT_BUNDLE` to that dataset
path. Stage 2 N15-A does not need to be rerun.

## Next Step

After the N15-A bundle exists, run N15-B:

```text
image-missing-feature complement FiLM
```

using the N15-A bundle as the Stage 2 pretrained checkpoint source.
