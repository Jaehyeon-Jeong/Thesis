# 4-N8 Stage 2 Pretrained Baseline-Preserving FiLM

Status: completed for checkpoint reload and F&G-only frozen-FiLM test; news
pretrained/frozen extension remains next

## Why This Replaces the Previous N8

The previous N8 candidate was `News + F&G combined-context ablation`. That is
now deferred. The reason is that N7 showed a more basic issue:

```text
Stage 4 reused the Stage 2 CNN architecture,
but it did not reuse the Stage 2 learned weights.
```

So previous Stage 4/N-series runs answered this question:

```text
Can a scratch-trained Stage2-style CNN plus context/FiLM beat the baseline?
```

The thesis question is stronger and cleaner:

```text
Can market/news context improve an already strong Stage 2 chart CNN by bounded
feature-wise correction?
```

N8 changes the experiment to answer the second question.

## Target Baseline

Primary baseline:

```text
Stage 2 I60/R20/ohlc_ma_vb
five-seed accuracy mean = 0.5793
five-seed ROC-AUC mean  = 0.5849
```

The next code path must load the selected Stage 2 checkpoint weights, not only
instantiate the same CNN architecture.

## Required Artifact Check

N8 needs Stage 2 `I60/R20/ohlc_ma_vb` checkpoints, ideally for seeds
`42, 43, 44, 45, 46`.

Local check before implementation:

```text
Expected checkpoint pattern:
outputs/stage2/checkpoints/stage2_i60_ohlc_ma_vb_r20/seed_<seed>/best.pt
```

Current local note: the local `stage2_btc_extension/outputs` folder currently
contains an I20 checkpoint, while the I60/R20/ohlc_ma_vb result archive stores
metrics/Grad-CAM metadata but not the full I60 checkpoint. For Kaggle N8, the
Stage 2 I60/R20/ohlc_ma_vb checkpoint must be available through one of:

- rerun/export Stage 2 selected baseline checkpoints,
- restore the checkpoint from a Kaggle output bundle if it exists,
- or run N8 in the same Kaggle session immediately after rebuilding the Stage 2
  checkpoint.

Do not proceed to frozen FiLM until checkpoint reload sanity passes.

Prepared rebuild cell:

- [kaggle_stage4_n8a0_rebuild_stage2_selected_checkpoints_one_cell.md](../notebooks/kaggle_stage4_n8a0_rebuild_stage2_selected_checkpoints_one_cell.md)

This cell trains/evaluates only:

```text
I60/R20/ohlc_ma_vb x seeds 42, 43, 44, 45, 46
```

It intentionally skips the full Stage 2 grid and Grad-CAM. At the end, download
this bundle:

```text
/kaggle/working/stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8.zip
```

The bundle includes `best.pt`, train metadata/history, classification/trading
metrics, predictions, and summary tables. `last.pt` is verified on Kaggle but
not included in the bundle because N8 only needs the best checkpoint.

Local bundle now available:

```text
/Users/jaehyeonjeong/Desktop/논문/stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8
```

## N8 Substeps

### 4-N8-A. Checkpoint Reload Sanity

Goal:

```text
Load Stage 2 best checkpoint inside the Stage 4 code path.
Run context-free prediction.
Verify metrics reproduce the Stage 2 baseline for the same seed/sample split.
```

Pass condition:

```text
Stage4 reload prediction ~= Stage2 checkpoint prediction
```

This separates a code/checkpoint-loading problem from a context-model problem.

Implementation:

- [check_stage4_n8_stage2_checkpoint_reload.py](../scripts/check_stage4_n8_stage2_checkpoint_reload.py)

Local result:

```text
status: passed
run seeds: 42, 43, 44, 45, 46
comparison tolerance: 1e-5
accuracy mean: 0.579320
ROC-AUC mean: 0.584863
F1 mean: 0.651071
predicted-positive-rate mean: 0.664400
```

Stage4-side reload reproduced the Stage 2 checkpoint bundle. Accuracy, F1, and
predicted class rates matched exactly. ROC-AUC and average precision differed
only by small floating-point-level deltas, with maximum absolute delta below
`1e-5`.

Generated local tables:

- `reports/tables/stage4_n8_stage2_checkpoint_reload_seed_results.csv`
- `reports/tables/stage4_n8_stage2_checkpoint_reload_metric_comparison.csv`
- `reports/tables/stage4_n8_stage2_checkpoint_reload_report.json`

Conclusion: the Stage 2 pretrained visual baseline can now be safely used by
the Stage 4/N8 code path.

### 4-N8-B. Frozen Backbone + Bounded Last-Block FiLM

Goal:

```text
Load Stage 2 CNN weights.
Freeze visual CNN blocks.
Initialize FiLM as identity: gamma = 1, beta = 0.
Train only context encoder + bounded final-block FiLM heads.
```

Initial formula:

```text
gamma = 1 + scale * tanh(raw_gamma)
beta  =     scale * tanh(raw_beta)
```

Initial scale grid should stay small:

```text
scale = 0.02 or 0.05
```

Context order:

```text
1. F&G-only
2. news SVD8-only
3. F&G + news SVD8, only if 1 or 2 is promising
```

Implementation status:

```text
implemented and smoke-tested
```

Implemented behavior:

- Load Stage 2 checkpoint keys into the Stage 4 bounded last-block FiLM model.
- Freeze the visual CNN blocks.
- Freeze the Stage 2 classifier.
- Keep frozen CNN BatchNorm and frozen classifier dropout in eval mode during
  Stage 4 training.
- Train only:
  - `context_encoder`
  - `gamma_head`
  - `beta_head`

Local smoke result:

```text
context: F&G-only
seed: 42
scale: 0.05
max rows: train 64, validation 64, test 64
loaded Stage 2 keys: 30
frozen parameters: 2,952,962
trainable parameters: 35,008
trainable names:
  context_encoder.net.0.weight
  context_encoder.net.0.bias
  context_encoder.net.3.weight
  context_encoder.net.3.bias
  gamma_head.weight
  gamma_head.bias
  beta_head.weight
  beta_head.bias
```

Prepared Kaggle runner:

- [kaggle_stage4_n8b_fg_only_pretrained_frozen_bounded_film_one_cell.md](../notebooks/kaggle_stage4_n8b_fg_only_pretrained_frozen_bounded_film_one_cell.md)

Kaggle five-seed F&G-only run:

```text
context: F&G-only
scales: 0.02, 0.05
seeds: 42, 43, 44, 45, 46
baseline: Stage 2 I60/R20/ohlc_ma_vb, accuracy mean 0.579320
```

Result:

| model | accuracy mean | ROC-AUC mean | F1 mean | predicted Up rate mean |
|---|---:|---:|---:|---:|
| Stage 2 baseline reload | 0.579320 | 0.584863 | 0.651071 | 0.664400 |
| N8-B F&G-only, scale 0.02 | 0.580291 | 0.584930 | 0.650814 | 0.660652 |
| N8-B F&G-only, scale 0.05 | 0.579320 | 0.584921 | 0.648829 | 0.656627 |

Seed-level observation for scale `0.02`:

```text
seed 42: +0.000694 accuracy vs Stage 2
seed 43: -0.001388
seed 44: +0.004164
seed 45:  0.000000
seed 46: +0.001388
```

Interpretation:

```text
N8-B does not produce a large improvement over the Stage 2 visual baseline.
However, unlike scratch-trained Stage 4 FiLM runs, it preserves the Stage 2
baseline and avoids the severe seed-level prediction collapse seen in P7/P8/V9.
Scale 0.02 is the best F&G-only setting, with a very small accuracy/ROC-AUC
increase over the Stage 2 reload baseline.
```

Generated tables:

- `reports/tables/stage4_n8b_fg_only_pretrained_frozen_bounded_film_seed_results.csv`
- `reports/tables/stage4_n8b_fg_only_pretrained_frozen_bounded_film_mean_std_results.csv`
- `reports/tables/stage4_n8b_fg_only_pretrained_frozen_bounded_film_run_summary.json`
- `reports/tables/stage4_n8b_context_source_audit.json`

Conclusion:

```text
The baseline-preserving FiLM structure is now validated as a stable attachment
method. The next question is no longer whether Stage 4 can preserve the strong
visual baseline, but whether richer external context, especially news, provides
incremental signal when attached through the same pretrained/frozen FiLM path.
```

### 4-N8-C. Frozen Backbone + Trainable Classifier

If 4-N8-B is too constrained, keep CNN blocks frozen but allow the final
classifier to train with the FiLM/context branch.

Question:

```text
Can the classifier learn to read FiLM-adjusted visual features without changing
the chart feature extractor?
```

### 4-N8-D. Final Block Partial-Unfreeze

Only after A-C:

```text
Freeze early CNN blocks.
Train final CNN block + bounded FiLM + classifier.
```

This is more flexible but less baseline-preserving, so it should not be the
first N8 experiment.

Current decision:

```text
Do not run N8-C/D immediately. N8-B already preserves the baseline, so the next
priority is to test news-only and news+F&G context with the same frozen
baseline-preserving structure. N8-C/D are kept as optional follow-ups if the
frozen classifier proves too restrictive for news context.
```

## Decision Rule

Promote a pretrained FiLM method only if:

1. checkpoint reload sanity reproduces the Stage 2 baseline,
2. five-seed results are at least competitive with Stage 2
   `I60/R20/ohlc_ma_vb`,
3. seed-level predicted-positive-rate collapse is reduced or absent,
4. exported gamma/beta values show bounded, interpretable correction rather
   than uncontrolled class bias.

## Thesis Interpretation if Successful

N8-B result:

```text
F&G-only market context is stable when applied as a bounded correction to a
strong pretrained visual chart model. It only very slightly improves the Stage 2
baseline, so it is not enough as a final performance claim, but it fixes the
main scratch-FiLM failure mode and gives a clean structure for the final
news-context tests.
```

If news-only or news+F&G also fails under the same structure:

```text
The selected chart representation already captures most usable signal for this
BTC setup, and headline/F&G context did not provide robust incremental value
under leakage-safe conditioning.
```
