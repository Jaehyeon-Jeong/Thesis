# 4-N12-B Confidence-Gated News FiLM

Status: prepared and locally shape-checked.

## Purpose

N12-A made FiLM stronger when the frozen Stage 2 visual classifier was
uncertain. Its five-seed result was stable but did not meaningfully beat the
Stage 2 baseline. N12-B tests the opposite hypothesis:

```text
news context may be useful as reinforcement when Stage 2 chart evidence is
already confident.
```

This is not the preferred final model by default. It is a controlled diagnostic
for deciding whether context should act near uncertainty or near confidence.

## Model

Fixed visual path:

```text
chart image -> Stage 2 I60/R20/ohlc_ma_vb CNN + classifier, frozen
```

Context path:

```text
headline TF-IDF/SVD context -> MLP -> raw_gamma/raw_beta
```

Confidence-gated bounded FiLM:

```text
stage2_prob_up = softmax(stage2_logits)[up]
confidence     = abs(2 * stage2_prob_up - 1)
gamma          = 1 + confidence * scale * tanh(raw_gamma)
beta           =     confidence * scale * tanh(raw_beta)
```

When Stage 2 is near `0.5`, `confidence` is near `0.0`, so FiLM is constrained.
When Stage 2 is confident toward Up or Down, `confidence` approaches `1.0`, so
the news branch can reinforce the final-block chart features.

## Risk

Confidence gating can also reinforce confidently wrong Stage 2 predictions.
Therefore this run must be judged by:

- accuracy and ROC-AUC;
- predicted-Up rate;
- correction/regression counts against Stage 2;
- examples where `Stage2 correct -> N12-B wrong`;
- gamma/beta modulation on confident wrong samples.

## Implementation

- Added `film_full_confidence_gated_last_block`.
- Added `ConfidenceGatedLastBlockFilmContextStockCNN`.
- Reuses the N8/N9/N10/N12-A Stage 2 pretrained/frozen loading path.
- Exports `modulation_gate` and `stage2_prob_up` with gamma/beta metadata.

## Kaggle Runner

Runner:

```text
notebooks/kaggle_stage4_n12b_confidence_gated_news_film_one_cell.md
```

Default grid:

```text
image/spec/horizon: I60 / R20 / ohlc_ma_vb
context: news TF-IDF/SVD32 over 7/20/60-day headline windows
method: film_full_confidence_gated_last_block
scale: 0.02, 0.05
seeds: 42, 43, 44, 45, 46
Stage 2 CNN/classifier: loaded and frozen
```

## Local Check

Command:

```bash
python scripts/check_stage4_model_shapes.py \
  --config configs/env_local.yaml \
  --model film_full_confidence_gated_last_block \
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
