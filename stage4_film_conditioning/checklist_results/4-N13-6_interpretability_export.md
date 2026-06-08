# 4-N13-6 Interpretability Export

## Goal

Export thesis-ready interpretation artifacts for the strongest compact
Stage 2 frozen context-FiLM candidates. This is not a new training experiment.
It is an evidence package for checking whether context-FiLM changes the visual
baseline in interpretable cases.

## Candidate Models

| Candidate | Context | Why included |
| --- | --- | --- |
| Stage 2 baseline | visual-only `I60/R20/ohlc_ma_vb` | Main frozen visual reference. |
| N8-B | F&G-only, bounded last-block FiLM, scale `0.02` | Best compact accuracy candidate. |
| N10 | news TF-IDF/SVD32, bounded last-block FiLM, scale `0.02` | Best news interpretability/ROC-Brier candidate. |

The selected-combo N13-5B model is not included as a primary interpretation
candidate because it tied the Stage 2 hard decisions and had net correction
`0.0`.

## Export Design

The Kaggle runner writes candidate-specific correction analysis tables:

- Stage 2 wrong -> context-FiLM correct.
- Stage 2 correct -> context-FiLM wrong.
- Seed-level transition summaries.
- Augmented selected-sample CSVs for targeted Grad-CAM.

It then exports matched Grad-CAM and modulation artifacts for the same sample
indices:

- Stage 2 targeted Grad-CAM.
- Stage 4 context-FiLM targeted Grad-CAM.
- Context values.
- FiLM gamma/beta summary CSV.
- Full modulation JSON.
- A compact downloadable bundle.

## Targeted Sample Panels

| Candidate | Extra context panels |
| --- | --- |
| F&G-only | `fg_extreme_fear`, `fg_extreme_greed` from `fg_mean_60_normalized`. |
| News SVD32 | `news_high_count_60d`, `news_svd60_09_high`, `news_svd60_09_low`. |

These panels allow interpretation even when the overall hard-decision change is
small. The claim is not only "accuracy improved"; it is also whether FiLM uses
context in plausible extreme regimes.

## Artifacts

- [N13-6 Kaggle one-cell runner](../notebooks/kaggle_stage4_n13_6_interpretability_export_one_cell.md)
- [Stage2-vs-context correction analyzer](../scripts/analyze_stage4_stage2_context_corrections.py)
- [Stage4 targeted Grad-CAM exporter](../scripts/generate_stage4_gradcam_context.py)
- Stage2 targeted Grad-CAM exporter: `stage2_btc_extension/scripts/generate_stage2_gradcam.py`

Expected Kaggle bundle:

```text
/kaggle/working/stage4_n13_6_interpretability_export_bundle.zip
```

## Status

Completed and reviewed.

Main readout:

```text
F&G N8-B scale 0.02:
Stage 2 accuracy mean 0.579320
Stage 4 accuracy mean 0.580291
delta +0.000972
corrections 21, regressions 14, net +7

News N10 SVD32 scale 0.02:
Stage 4 accuracy mean 0.579736
delta +0.000416
corrections 27, regressions 24, net +3
```

The targeted Grad-CAM/gamma-beta review showed that the current bounded FiLM
path mostly preserves Stage 2 decisions. Its gamma/beta movement is very small,
so N13-7A tests whether a larger bounded scale can create more useful
correction without class collapse.
