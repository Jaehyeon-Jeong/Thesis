# 4-N13-7A F&G Bounded FiLM Scale Grid

## Status

Completed and reviewed.

## Purpose

N8-B showed that the baseline-preserving structure works:

```text
Stage 2 I60/R20/ohlc_ma_vb CNN/classifier frozen
F&G context -> MLP -> bounded last-block FiLM
```

However, N13-6 showed that the actual gamma/beta movement was very small. N13-7A
therefore tests whether the bounded FiLM branch was too conservative.

## Fixed Setup

```text
image: I60/R20/ohlc_ma_vb
context: F&G-only
context features: fg_value, fg_mean_60, fg_delta_60, fg_std_60
model: film_full_bounded_last_block
visual CNN: frozen Stage 2 checkpoint
classifier: frozen Stage 2 checkpoint
trainable modules: F&G context encoder + final-block FiLM heads
seeds: 42, 43, 44, 45, 46
```

## Tested Axis

The equation stays the same:

```text
gamma = 1 + scale * tanh(raw_gamma)
beta  =     scale * tanh(raw_beta)
```

N8-B already tested `scale=0.02` and `scale=0.05`. N13-7A tests:

```text
scale = 0.10
scale = 0.20
```

This is not a random search. It directly tests whether allowing larger
gamma/beta movement creates useful correction or causes prediction collapse.

## Kaggle Runner

Use:

[kaggle_stage4_n13_7a_fg_bounded_scale_grid_one_cell.md](../notebooks/kaggle_stage4_n13_7a_fg_bounded_scale_grid_one_cell.md)

The runner writes:

```text
reports/tables/stage4_n13_7a_fg_bounded_scale_grid_seed_results.csv
reports/tables/stage4_n13_7a_fg_bounded_scale_grid_mean_std_results.csv
reports/tables/stage4_n13_7a_fg_bounded_scale_grid_correction_summary.csv
reports/tables/stage4_n13_7a_fg_bounded_scale_grid_s0p1_stage2_vs_film_correction_analysis_*.csv
reports/tables/stage4_n13_7a_fg_bounded_scale_grid_s0p2_stage2_vs_film_correction_analysis_*.csv
```

It also creates:

```text
/kaggle/working/stage4_n13_7a_fg_bounded_scale_grid_result_bundle.zip
```

## Readout

Keep a larger scale only if it improves at least one main metric without
damaging stability:

```text
accuracy / ROC-AUC / Brier
predicted_positive_rate
Stage2 wrong -> FiLM correct
Stage2 correct -> FiLM wrong
net correction
gamma/beta magnitude in Grad-CAM modulation summaries
```

If `0.10` or `0.20` increases correction but also increases regression or class
collapse, then the thesis conclusion is that bounded small-scale FiLM is the
more defensible model.

## Result

N13-7A tested larger bounded scales after N8-B had already tested `0.02` and
`0.05`.

| Scale | Accuracy mean | ROC-AUC mean | Brier mean | Predicted positive rate | Net correction vs Stage 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.02` N8-B reference | `0.580291` | `0.584930` | `0.274004` | `0.660652` | `+7` |
| `0.05` N8-B reference | `0.579320` | `0.584921` | `0.273838` | `0.656627` | not rerun in N13-7A |
| `0.10` | `0.579042` | `0.584811` | `0.273867` | `0.656627` | `-2` |
| `0.20` | `0.578487` | `0.584539` | `0.273940` | `0.654129` | `-6` |

Seed-level correction summary:

```text
scale 0.10:
corrections 37, regressions 39, net -2

scale 0.20:
corrections 51, regressions 57, net -6
```

No severe class-collapse was observed. The predicted-positive rate stayed around
`0.65`, close to the N8-B scale `0.05` setting. However, increasing the scale
did not improve the mean accuracy, ROC-AUC, Brier score, trading metrics, or net
correction.

The Grad-CAM modulation summaries show that the larger scales increased
gamma/beta movement only modestly. The average sampled final-block
`delta_gamma` standard deviation rose from roughly `0.00075` at scale `0.10` to
roughly `0.00102` at scale `0.20`, but this extra movement produced more
regressions than corrections.

## Decision

Do not keep `scale=0.10` or `scale=0.20` as the final F&G FiLM model.

The best F&G setting remains N8-B `scale=0.02`: it is the only tested F&G
bounded-FiLM setting with positive net correction over Stage 2 and the highest
accuracy/ROC-AUC among the F&G scale grid.

Next step should not be another larger-scale run. Move to the N13-7B/C/D
constraint/freeze-policy ablations only if a final robustness check is needed:

```text
N13-7B: relaxed or weakly regularized gamma/beta constraint
N13-7C: alternative positive-gamma parameterization
N13-7D: classifier-unfreeze while keeping visual CNN frozen
```
