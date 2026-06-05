# 4-N10 News Interpretability Report

Status: initial report completed; targeted N10 Grad-CAM/gamma-beta export
completed in 4-N10-B

## Purpose

N10 checks whether Stage 4 news/F&G FiLM adds interpretable correction to the
strong Stage 2 visual baseline.

The main question is not only:

```text
Did accuracy or ROC-AUC improve?
```

but also:

```text
Did context-FiLM avoid seed collapse?
Did it reduce class bias?
Did gamma/beta actually modulate visual features?
Can we identify samples where context corrected a Stage 2 visual-baseline error?
```

## Inputs Used

Metric-level inputs:

- Stage 2 selected baseline:
  `I60/R20/ohlc_ma_vb`, five seeds.
- N8-B:
  F&G-only, Stage 2 CNN/classifier frozen, bounded last-block FiLM,
  scale `0.02`.
- N9-A:
  news SVD8-only, Stage 2 CNN/classifier frozen, bounded last-block FiLM,
  scale `0.02`.
- N9 grid:
  news SVD8/16/32 and scale `0.02/0.05`, excluding the already completed
  SVD8/scale0.02 anchor.

Written comparison tables:

```text
reports/tables/stage4_n10_metric_comparison.csv
reports/tables/stage4_n10_seed_delta_comparison.csv
reports/tables/stage4_n10_seed42_gradcam_sample_comparison.csv
```

Available Grad-CAM/context-modulation artifacts:

- Stage 2 seed-42 selected Grad-CAM samples.
- N8-B seed-42 selected Grad-CAM samples.
- N9-A seed-42 selected Grad-CAM samples.

Important limitation:

```text
The N9 SVD/scale grid bundle is metric-only. It does not include prediction CSV,
checkpoint, Grad-CAM, or modulation exports for SVD32/scale0.02. Therefore the
grid-best model cannot yet be used for matched-sample Grad-CAM interpretation.
```

Update after 4-N10-B:

```text
The selected N10 grid-best candidate, SVD32/scale0.02, was rerun/exported with
targeted Stage 2 vs N10 Grad-CAM and gamma/beta modulation metadata.
```

Targeted result:

```text
Stage2 wrong -> N10 correct: 27
Stage2 correct -> N10 wrong: 24
Net improvement: +3 / 7205
```

This means the targeted export is useful for interpretation, but the
performance gain remains too small to claim a strong accuracy improvement.

## Metric-Level Interpretation

| Model | Accuracy mean | ROC-AUC mean | Brier mean | Predicted Up mean |
|---|---:|---:|---:|---:|
| Stage 2 baseline | 0.579320 | 0.584862 | 0.274337 | 0.664400 |
| N8-B F&G-only scale0.02 | 0.580291 | 0.584930 | 0.274004 | 0.660652 |
| N9-A news SVD8 scale0.02 | 0.579459 | 0.585670 | 0.273367 | 0.651770 |
| N9 grid SVD32 scale0.02 | 0.579736 | 0.585353 | 0.273765 | 0.657321 |
| N9 grid SVD8 scale0.05 | 0.578626 | 0.586619 | 0.272327 | 0.639001 |

### Main Read

N8-B and N9 do not clearly beat the Stage 2 baseline in accuracy. The
improvements are too small to claim a strong performance gain.

However, the news/F&G FiLM path is not useless:

- N9-A improves ROC-AUC slightly versus Stage 2: `+0.000808`.
- N9-A improves Brier score slightly: `-0.000970`.
- N9-A reduces predicted-Up bias from `0.664400` to `0.651770`.
- N9 grid SVD8/scale0.05 has the best ROC-AUC and Brier score, but its
  accuracy is lower than Stage 2.
- N9 grid SVD32/scale0.02 gives the best grid accuracy, but the gain is only
  `+0.000416` versus Stage 2.

This supports a cautious conclusion:

```text
News/F&G bounded FiLM preserves the Stage 2 baseline and slightly improves
ranking/calibration signals, but it does not yet deliver a defensible accuracy
gain.
```

## Collapse and Bias Review

No N9 grid configuration has severe class collapse.

Collapse rule:

```text
predicted_positive_rate < 0.1 or predicted_positive_rate > 0.9
```

Result:

```text
collapse count = 0
```

But there is still an Up-heavy tendency:

```text
test true positive rate ~= 0.541
Stage 2 predicted Up mean = 0.664
N9-A predicted Up mean    = 0.652
SVD32/0.02 predicted Up   = 0.657
SVD8/0.05 predicted Up    = 0.639
```

So the model is not collapsing, but it remains biased toward Up predictions.
The N9 variants reduce this bias slightly, especially `SVD8/scale0.05`.

Seed `44` remains the main warning seed. It frequently shows predicted-Up rates
above `0.70`, even though this is not a full collapse.

## Seed-Level Interpretation

N9-A versus Stage 2:

- ROC-AUC improves in all five seeds.
- Brier score improves in all five seeds.
- Predicted-Up rate decreases in all five seeds.
- Accuracy improves in seeds `42`, `44`, `45`, but decreases in seeds `43`,
  `46`.

N9 grid SVD32/scale0.02 versus Stage 2:

- ROC-AUC improves in all five seeds.
- Brier score improves in all five seeds.
- Predicted-Up rate decreases in all five seeds.
- Accuracy improves in seeds `42`, `43`, `44`, `45`, but decreases in seed
  `46`.

This suggests that the bounded news-FiLM correction is stable and mildly useful
for ranking/calibration, but the decision threshold and class bias still limit
accuracy gains.

## Grad-CAM and Gamma/Beta Interpretation

The currently available matched Grad-CAM samples are seed-42 selected examples.
They are all already correct under Stage 2:

| Date | True label | Stage 2 prob_up | N8-B prob_up | N9-A prob_up |
|---|---:|---:|---:|---:|
| 2023-12-26 | Up | 0.988826 | 0.988657 | 0.988504 |
| 2021-01-19 | Up | 0.988402 | 0.988240 | 0.988141 |
| 2024-06-21 | Down | 0.011502 | 0.011390 | 0.011293 |
| 2024-06-23 | Down | 0.018830 | 0.018684 | 0.018486 |

Interpretation:

```text
For high-confidence correct samples, N8-B and N9-A barely change the Stage 2
prediction. This is the desired behavior for a baseline-preserving FiLM model.
```

The modulation magnitudes are also very small:

```text
N8-B seed42 selected samples:
block4_delta_gamma_mean ~= 0.000067 to 0.000090
block4_beta_mean        ~= 0.000023 to 0.000029

N9-A seed42 selected samples:
block4_delta_gamma_mean ~= 0.000059 to 0.000093
block4_beta_mean        ~= 0.000020 to 0.000032
```

This means:

```text
The current bounded FiLM is acting as a small residual correction, not as a
large feature override.
```

That is consistent with the N8/N9 design goal: preserve the strong Stage 2 CNN
and allow only small context-conditioned adjustments.

## What Is Still Missing for Full N10

The most important interpretability group is:

```text
Stage 2 wrong -> N9 correct
```

The available Grad-CAM samples do not cover this group. They are all
high-confidence correct examples, so they only show preservation behavior.

To finish the full interpretability claim, generate targeted samples for:

```text
Stage 2 wrong -> N9 correct
Stage 2 correct -> N9 wrong
both correct
both wrong
```

For each selected sample, export:

- Stage 2 prediction and Grad-CAM;
- N9 prediction and Grad-CAM;
- news context values;
- gamma/beta summary;
- probability delta;
- correctness transition.

## Added Correction-Analysis Code

Prepared the N10-A analysis path for the missing targeted comparison:

```text
Stage 2 prediction CSV
N10 prediction CSV
-> merge by sample_index
-> classify transition type
-> export Stage2 wrong -> N10 correct samples
-> export Stage2 correct -> N10 wrong samples
-> export selected sample-index lists for targeted Grad-CAM
```

Files:

```text
scripts/analyze_stage4_stage2_context_corrections.py
notebooks/kaggle_stage4_n10_stage2_vs_n10_correction_analysis_one_cell.md
```

Important execution note:

```text
The compact N10 result bundle contains Grad-CAM figures/metrics but not full
prediction CSVs/checkpoints. Run the N10-A Kaggle cell immediately after the N10
selected interpretability run, while full outputs are still in /kaggle/working,
or regenerate predictions from the available checkpoints first.
```

Recommended N9 model for this targeted export:

```text
Primary:   N9 grid SVD32 / scale0.02
Secondary: N9-A SVD8 / scale0.02 or N9 grid SVD8 / scale0.05
```

## Current Thesis-Safe Claim

The current evidence supports this limited claim:

```text
Pretrained/frozen bounded FiLM avoids the scratch-FiLM collapse mode and
preserves the strong Stage 2 visual baseline. News/F&G context produces small,
stable corrections that slightly improve ROC-AUC/calibration and reduce
Up-class bias, but it does not yet produce a strong accuracy improvement.
```

Do not claim yet:

```text
News-FiLM clearly outperforms Stage 2.
```

A stronger claim requires targeted matched-sample analysis showing that FiLM
corrects some Stage 2 visual-baseline errors in interpretable market/news
conditions.
