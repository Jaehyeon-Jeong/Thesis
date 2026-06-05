# 4-N12-A Uncertainty-Gated News FiLM

Status: completed on Kaggle, reviewed locally.

## Purpose

N9/N10 showed that pretrained/frozen bounded FiLM can preserve the Stage 2
visual baseline, but the net correction is small. N12-A tests a more targeted
claim: news context should intervene mainly when the frozen Stage 2 chart model
is uncertain.

## Model

Fixed visual path:

```text
chart image -> Stage 2 I60/R20/ohlc_ma_vb CNN + classifier, frozen
```

Context path:

```text
headline TF-IDF/SVD context -> MLP -> raw_gamma/raw_beta
```

Uncertainty-gated bounded FiLM:

```text
stage2_prob_up = softmax(stage2_logits)[up]
uncertainty    = 4 * stage2_prob_up * (1 - stage2_prob_up)
gamma          = 1 + uncertainty * scale * tanh(raw_gamma)
beta           =     uncertainty * scale * tanh(raw_beta)
```

When Stage 2 is near `0.5`, `uncertainty` is near `1.0`, so the context-FiLM
correction is allowed to act. When Stage 2 is confident, `uncertainty` becomes
smaller and FiLM is constrained.

## Implementation

- Added `film_full_uncertainty_gated_last_block`.
- Added `UncertaintyGatedLastBlockFilmContextStockCNN`.
- Reuses the same Stage 2 pretrained/frozen loading path as N8/N9/N10.
- Exports `modulation_gate` and `stage2_prob_up` with gamma/beta metadata for
  later interpretation.

## Kaggle Runner

Runner:

```text
notebooks/kaggle_stage4_n12a_uncertainty_gated_news_film_one_cell.md
```

Default grid:

```text
image/spec/horizon: I60 / R20 / ohlc_ma_vb
context: news TF-IDF/SVD32 over 7/20/60-day headline windows
method: film_full_uncertainty_gated_last_block
scale: 0.02, 0.05
seeds: 42, 43, 44, 45, 46
Stage 2 CNN/classifier: loaded and frozen
```

## Local Check

Command:

```bash
python scripts/check_stage4_model_shapes.py \
  --config configs/env_local.yaml \
  --model film_full_uncertainty_gated_last_block \
  --image-window 60 \
  --image-spec ohlc_ma_vb \
  --return-horizon 20 \
  --batch-size 2
```

Result:

- Status: `ok`.
- Parameter count: `2,988,098`, matching expected.
- Delta versus Stage 2 I60 baseline: `+35,136`.
- Initial modulation identity passed: `gamma=1`, `beta=0`.
- Exported tensor shapes include `modulation_gate4` and `stage2_prob_up4`.

## Decision Criteria

N12-A is useful only if it improves one of these without reintroducing seed
collapse:

- accuracy mean versus Stage 2 `0.5793`;
- ROC-AUC mean versus Stage 2 `0.5849`;
- predicted-positive-rate stability across seeds;
- correction/regression balance in Stage 2 vs N12-A comparison;
- interpretable cases where `Stage2 wrong -> N12-A correct` occurs with high
  uncertainty gate.

## Kaggle Result

Local result bundle:

```text
/Users/jaehyeonjeong/Desktop/논문/N12_A_results
```

Summary:

| scale | accuracy mean | accuracy std | ROC-AUC mean | F1 mean | predicted-Up rate mean |
|---:|---:|---:|---:|---:|---:|
| `0.02` | `0.5790` | `0.0177` | `0.5851` | `0.6487` | `0.6572` |
| `0.05` | `0.5786` | `0.0176` | `0.5855` | `0.6462` | `0.6498` |

Interpretation:

- Accuracy is essentially tied with, and slightly below, the selected Stage 2
  baseline (`0.5793`).
- ROC-AUC improves only marginally (`+0.0002` to `+0.0006`), which is too small
  to claim a meaningful performance gain.
- The structure avoids severe collapse, but the prediction distribution remains
  Up-biased (`predicted_positive_rate` around `0.65` versus true positive rate
  around `0.54`).
- N12-A is therefore useful as a diagnostic/stability result, not as a final
  thesis performance-improvement claim.

Next controlled follow-up: N12-B confidence-gated news FiLM, which tests the
opposite interpretation: context may be more useful when reinforcing
high-confidence Stage 2 chart evidence rather than correcting uncertain cases.
