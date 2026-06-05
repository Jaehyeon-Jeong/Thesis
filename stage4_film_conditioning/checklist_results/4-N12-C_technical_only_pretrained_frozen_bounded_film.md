# 4-N12-C Technical-Only Pretrained Frozen Bounded FiLM

Status: runner prepared; Kaggle result pending.

## Goal

N12-C tests whether OHLCV-derived technical context can improve the strong
Stage 2 `I60/R20/ohlc_ma_vb` visual baseline when the Stage 2 CNN and classifier
are preserved.

This is not another from-scratch Stage 4 run. The protocol is:

```text
chart image -> pretrained Stage 2 CNN/classifier frozen
technical context -> MLP -> bounded final-block FiLM
trainable part -> context encoder + FiLM gamma/beta heads only
```

## Context Features

The technical-only vector uses four features computed from BTC OHLCV history:

```text
bb_percent_b_60
bb_bandwidth_60
mfi_60
rv_60
```

These are intentionally separated from F&G/news context because they are mostly
derived from the same price/volume source already visible in `ohlc_ma_vb`.

## Kaggle Runner

Notebook:

```text
notebooks/kaggle_stage4_n12c_technical_only_pretrained_frozen_bounded_film_one_cell.md
```

Default grid:

```text
image: I60/R20/ohlc_ma_vb
method: film_full_bounded_last_block
context: technical_only
scales: 0.02, 0.05
seeds: 42, 43, 44, 45, 46
```

The runner writes:

```text
reports/tables/stage4_n12c_technical_only_pretrained_frozen_bounded_film_seed_results.csv
reports/tables/stage4_n12c_technical_only_pretrained_frozen_bounded_film_mean_std_results.csv
/kaggle/working/stage4_n12c_technical_only_pretrained_frozen_bounded_film_result_bundle.zip
```

## Local Check

Local shape check passed with:

```text
context_dim: 4
context_embedding: (2, 32)
input_shape: (1, 96, 180)
logits: (2, 2)
```

## Interpretation Plan

After Kaggle results, compare N12-C against:

- Stage 2 baseline;
- N8-B F&G-only;
- N9/N10 news-only;
- N12-A/B gated news.

If technical-only context does not improve the frozen Stage 2 baseline, the
thesis interpretation is straightforward: BB/MFI/RV-like signals are already
mostly available to the chart CNN through `ohlc_ma_vb`, so they are weak as a
separate FiLM condition source.
