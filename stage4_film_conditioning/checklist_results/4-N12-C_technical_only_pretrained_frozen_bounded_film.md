# 4-N12-C Technical-Only Pretrained Frozen Bounded FiLM

Status: completed.

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

## Result

Result bundle:

```text
/Users/jaehyeonjeong/Desktop/논문/N12_C_results
```

Committed compact tables:

```text
reports/tables/stage4_n12c_technical_only_pretrained_frozen_bounded_film_seed_results.csv
reports/tables/stage4_n12c_technical_only_pretrained_frozen_bounded_film_mean_std_results.csv
reports/tables/stage4_n12c_context_source_audit.json
```

Five-seed mean/std:

| context | scale | accuracy mean | accuracy std | ROC-AUC mean | F1 mean | Brier mean | predicted-Up mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| technical-only bounded FiLM | `0.02` | `0.579736` | `0.018047` | `0.584778` | `0.650834` | `0.274218` | `0.662318` |
| technical-only bounded FiLM | `0.05` | `0.579320` | `0.018039` | `0.584684` | `0.649920` | `0.274068` | `0.660236` |

Reference Stage 2 frozen reload baseline:

| model | accuracy mean | ROC-AUC mean | F1 mean | Brier mean | predicted-Up mean |
|---|---:|---:|---:|---:|---:|
| Stage 2 `I60/R20/ohlc_ma_vb` | `0.579320` | `0.584863` | `0.651071` | `0.274337` | `0.664400` |

Best N12-C setting is scale `0.02` by accuracy, but the improvement over Stage
2 is only `+0.000416` accuracy mean. ROC-AUC is slightly lower
(`-0.000085`), and F1 is also slightly lower (`-0.000237`). This is effectively
a tie, not a meaningful improvement.

Seed-level behavior versus Stage 2:

| seed | scale `0.02` accuracy delta | scale `0.02` ROC-AUC delta | scale `0.05` accuracy delta | scale `0.05` ROC-AUC delta |
|---:|---:|---:|---:|---:|
| `42` | `+0.000000` | `+0.000004` | `+0.000694` | `+0.000058` |
| `43` | `+0.000000` | `-0.000014` | `+0.000000` | `-0.000048` |
| `44` | `+0.001388` | `-0.000644` | `+0.000000` | `-0.001311` |
| `45` | `+0.000000` | `-0.000002` | `+0.000694` | `-0.000012` |
| `46` | `+0.000694` | `+0.000229` | `-0.001388` | `+0.000419` |

Context audit:

```text
primary_features: bb_percent_b_60, bb_bandwidth_60, mfi_60, rv_60
sample_count: 2399
test_count: 1441
date range: 2018-04-29 to 2024-12-11
technical feature missing rate: 0.0 for train/validation/test
```

## Interpretation

N12-C does not provide meaningful evidence that OHLCV-derived technical context
improves the frozen Stage 2 chart model. The result supports the working
interpretation that BB/MFI/RV-like information is already mostly available in
the `ohlc_ma_vb` chart image, so adding it again as a separate FiLM condition
has little value.

This is still useful for the thesis because it separates context-source types:

- `technical-only`: mostly redundant with the visual chart;
- `F&G-only`: compact external regime context, slightly more promising in N8-B;
- `news-only`: external text-derived context, useful for interpretability but
  still weak as a performance-improving signal so far.

Next comparison:

- Stage 2 baseline;
- N8-B F&G-only;
- N9/N10 news-only;
- N12-A/B gated news.

N12-D should consolidate these context sources under the same frozen Stage 2
protocol instead of adding another technical-only variant.
