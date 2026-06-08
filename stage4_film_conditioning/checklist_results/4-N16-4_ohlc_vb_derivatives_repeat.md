# 4-N16-4 OHLC-VB Derivatives Repeat

## Status

Completed. Five-seed Kaggle run finished and report tables were archived.

Runner:

```text
notebooks/kaggle_stage4_n16_4_ohlc_vb_derivatives_repeat_one_cell.md
```

## Purpose

N16-3 on `ohlc_ma_vb` did not materially beat the same-image Stage 2 baseline,
but it also did not collapse. N16-4 is a narrow completeness check: test whether
derivatives/leverage context is more useful on `ohlc_vb`, where the image has
volume but no moving-average channel.

This is not a new broad grid. It is a selected volume-aware repeat.

## Fixed Setup

```text
image_window: 60
return_horizon: 20
image_spec: ohlc_vb
context_method: film_full_bounded_last_block
modulation_scale: 0.02
run_seeds: 42, 43, 44, 45, 46
Stage 2: seed-matched I60/R20/ohlc_vb checkpoint loaded
freeze_visual_backbone: true
freeze_classifier: true
trainable modules: context encoder + bounded final-block FiLM heads
```

## Selected Feature Sets

| Feature set | Reason |
| --- | --- |
| `funding_plus_cftc_oi` | Best N16-3 accuracy row; tied same-image Stage 2 baseline on `ohlc_ma_vb`. |
| `funding_only` | Cleanest derivatives signal from N16-2; smallest context family. |

The output uses `n16d_` experiment/context suffixes so it cannot overwrite the
N16-3 `ohlc_ma_vb` results.

## Expected Outputs

```text
reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_seed_results.csv
reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_mean_std_results.csv
reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_correction_seed_summary.csv
reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_correction_transition_summary.csv
/kaggle/working/stage4_n16_4_ohlc_vb_derivatives_repeat_result_bundle.zip
```

## Result Summary

Same-image Stage 2 baseline:

```text
stage2_i60_ohlc_vb_r20
accuracy_mean: 0.567384
roc_auc_mean: 0.561247
f1_mean: 0.658574
brier_score_mean: 0.258306
predicted_positive_rate_mean: 0.741013
```

N16-4 results:

| feature set | accuracy mean | delta vs Stage 2 `ohlc_vb` | ROC-AUC mean | Brier mean | net corrections |
| --- | ---: | ---: | ---: | ---: | ---: |
| `funding_plus_cftc_oi` | 0.569466 | +0.002082 | 0.561820 | 0.257451 | +15 |
| `funding_only` | 0.567661 | +0.000278 | 0.561255 | 0.258085 | +2 |

`funding_plus_cftc_oi` improved over the same-image `ohlc_vb` Stage 2 baseline
in all five seeds. The effect is small, but it is more consistent than the
`ohlc_ma_vb` N16-3 result and stronger than prior `ohlc_vb` context checks:

```text
N15-B ohlc_vb + bb_trend:       accuracy delta +0.000139
N15-C ohlc_vb + F&G-only:       accuracy delta +0.000555
N16-4 ohlc_vb + funding+CFTC:   accuracy delta +0.002082
```

The model still does not beat the strongest visual baseline
`stage2_i60_ohlc_ma_vb_r20` (`0.579320`), so the result should be presented as
a same-image improvement on a volume-aware but MA-free chart, not as a new
overall best model.

## Decision

Treat N16-4 as useful only if it improves the same-image `ohlc_vb` Stage 2
baseline, not merely if it is close to the stronger `ohlc_ma_vb` baseline.

Continue to interpretation only if at least one row shows:

```text
accuracy_mean > same-image Stage 2 ohlc_vb baseline
or net corrections > 0 without predicted-positive-rate collapse
or ROC-AUC/Brier improves while accuracy is essentially tied
```

If this also fails, derivatives/leverage context should be reported as tested
but not robustly improving the frozen Stage 2 visual baseline.

N16-4 satisfies the same-image improvement criterion for
`funding_plus_cftc_oi`. The effect is modest, so the next step should be either:

1. interpret `ohlc_vb + funding_plus_cftc_oi` with correction/regression samples,
   or
2. stop expansion and report it as the clearest small positive Stage 4 context
   result.

## Linked Tables

```text
reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_seed_results.csv
reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_mean_std_results.csv
reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_correction_seed_summary.csv
reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_correction_transition_summary.csv
```
