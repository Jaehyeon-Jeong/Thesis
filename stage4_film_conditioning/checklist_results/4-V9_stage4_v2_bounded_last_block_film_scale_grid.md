# 4-V9 Stage 4 v2 Bounded Last-Block FiLM Scale Grid

Status: completed; move to news-context track

## Purpose

`4-V9` is the final structured numeric-context stabilization check before
moving to news context. It does not introduce a new architecture. It keeps the
`4-V7` bounded last-block FiLM design fixed and changes only the FiLM
modulation scale.

This is necessary because `4-V8` showed that:

- P7 `film_full` with F&G-only context had seeds `43`/`44` collapse mostly Up.
- P8 `film_full_bounded_last_block` improved ROC-AUC ranking signal but seeds
  `43`/`44` collapsed mostly Down.
- Therefore the next controlled question is whether the bounded modulation
  scale is still too strong or poorly calibrated.

## Fixed Experiment

```text
Image window: I60
Return horizon: R20
Image spec: ohlc_ma_vb
Context window: 60
Context features: fg_value, fg_mean_60, fg_delta_60, fg_std_60
Context method: film_full_bounded_last_block
Seeds: 42, 43, 44, 45, 46
Scales: 0.02, 0.05, 0.10
```

The output experiment suffix is scale-specific:

```text
fg_only_scale_0p02
fg_only_scale_0p05
fg_only_scale_0p10
```

This prevents the V9 runs from overwriting P8 or each other.

## What Changes

Only this parameter changes:

```text
stage4_model.film_full_bounded_last_block.modulation_scale
```

The model uses:

```text
gamma = 1 + scale * tanh(raw_gamma)
beta  = scale * tanh(raw_beta)
```

So smaller scale values force FiLM closer to identity:

- `scale=0.02`: very conservative modulation.
- `scale=0.05`: medium conservative modulation.
- `scale=0.10`: current P8 setting, rerun with scale-specific output naming.

## What Does Not Change

- The chart image remains `ohlc_ma_vb`.
- The context remains F&G-only.
- The Stage 2 visual data/image/split/evaluation path remains fixed.
- The checkpoint rule remains validation loss.
- Validation-threshold calibration is recorded separately in V8, but V9 does
  not use threshold calibration to select checkpoints.

This keeps V9 interpretable as a scale-only experiment. If the checkpoint rule
were changed at the same time, it would be unclear whether improvement came
from FiLM scale or from threshold/checkpoint selection.

## Outputs

Kaggle one-cell:

```text
notebooks/kaggle_stage4_v2_v9_bounded_last_block_film_scale_grid_one_cell.md
```

Expected result tables:

```text
reports/tables/stage4_v2_v9_bounded_last_block_film_scale_grid_seed_results.csv
reports/tables/stage4_v2_v9_bounded_last_block_film_scale_grid_mean_std_results.csv
reports/tables/stage4_v2_v9_bounded_last_block_film_scale_grid_collapse_summary.csv
reports/tables/stage4_v2_v9_bounded_last_block_film_scale_grid_run_summary.json
```

The notebook also creates:

```text
/kaggle/working/stage4_v2_v9_bounded_last_block_film_scale_grid_result_bundle.zip
```

## Result Summary

V9 completed the scale grid:

| Scale | Accuracy mean | ROC-AUC mean | Test collapse | Interpretation |
|:---|---:|---:|---:|:---|
| `0.02` | `0.5432` | `0.5704` | `1/5` | Most conservative; reduced some collapse but stayed below baseline |
| `0.05` | `0.5417` | `0.5733` | `1/5` | Similar to `0.02`; not enough recovery |
| `0.10` | `0.5425` | `0.5763` | `2/5` | Best ROC-AUC, but more collapse |

Reference Stage 2 visual baseline:

```text
I60/R20/ohlc_ma_vb
accuracy mean = 0.5793
ROC-AUC mean  = 0.5849
```

Seed-level conclusion:

- seeds `42`, `45`, and `46` were usable across scales;
- seed `43` improved slightly when scale was reduced but remained weak;
- seed `44` collapsed mostly Down for every scale.

Therefore, the V9 result is not a reason to keep searching gamma/beta scales.
The controlled conclusion is:

> F&G-only structured numeric FiLM gives some ranking signal, but it does not
> robustly improve the strong Stage 2 visual baseline. Stage 4 should move to
> richer external context, especially news.

## Evaluation Criteria Used

Promote a V9 scale to the final structured numeric-context FiLM candidate only
if it satisfies both conditions:

1. It is competitive with or better than the Stage 2 visual baseline:
   - Stage 2 `I60/R20/ohlc_ma_vb` accuracy mean: `0.5793`
   - Stage 2 `I60/R20/ohlc_ma_vb` ROC-AUC mean: `0.5849`
2. It materially reduces seed collapse:
   - fewer seeds with `predicted_positive_rate <= 0.15`
   - fewer seeds with `predicted_positive_rate >= 0.85`

No scale satisfied these criteria. The decision is:

- close F&G-only numeric FiLM as negative/unstable for the main claim;
- move to headline-only, strict `t-1`, non-LLM news context;
- keep bounded last-block FiLM as the preferred protected-visual-path
  architecture for the first news FiLM test.

## News Handoff

Next track:

```text
news headlines -> strict t-1 daily aggregation
               -> train-only TF-IDF/SVD vector
               -> CNN + news concat
               -> CNN + news bounded last-block FiLM
```

First news vector:

```text
news_svd_7d + news_svd_20d + news_svd_60d
news_count_7d + news_count_20d + news_count_60d
```

News + F&G should be tested only after news-only shows useful signal.
