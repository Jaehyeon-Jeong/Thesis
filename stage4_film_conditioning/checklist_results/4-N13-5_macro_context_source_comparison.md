# 4-N13-5 Macro Context-Source Comparison

## Purpose

Compare all completed Stage 2 frozen context-FiLM sources under one baseline
before building a selected combined context vector.

## Protocol

- Visual baseline: Stage 2 `I60/R20/ohlc_ma_vb`
- Freeze policy: Stage 2 CNN and classifier frozen
- Context protocol: bounded last-block FiLM unless noted by the source result
- Seeds: `42, 43, 44, 45, 46`
- Comparison sources:
  - F&G-only
  - headline news TF-IDF/SVD
  - technical-only BB/MFI/RV
  - OFR FSI
  - public-data RORO proxy

## Compact Result

| Source | Best Row | Accuracy | dAcc | ROC-AUC | dROC | Brier | dBrier | F1 | Collapse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Visual baseline | Stage2 frozen baseline | 0.579320 | 0.000000 | 0.584862 | 0.000000 | 0.274337 | 0.000000 | 0.651071 | 0 |
| F&G | N8-B F&G-only s0.02 | 0.580291 | +0.000972 | 0.584930 | +0.000068 | 0.274004 | -0.000333 | 0.650814 | 0 |
| FSI | N13-2 `fsi_all` | 0.579875 | +0.000555 | 0.584859 | -0.000003 | 0.274190 | -0.000147 | 0.651319 | 0 |
| News | N9 news SVD32 s0.02 | 0.579736 | +0.000416 | 0.585353 | +0.000491 | 0.273765 | -0.000572 | 0.649307 | 0 |
| Technical | N12-C technical s0.02 | 0.579736 | +0.000416 | 0.584778 | -0.000084 | 0.274218 | -0.000119 | 0.650834 | 0 |
| RORO | N13-4 `roro_3` | 0.579320 | 0.000000 | 0.584748 | -0.000113 | 0.274278 | -0.000059 | 0.650924 | 0 |

## Interpretation

No completed context source materially beats the frozen Stage 2 visual baseline.
The best accuracy row remains F&G-only scale `0.02`, but the improvement is
only `+0.000972`. That is too small to claim a strong performance gain.

The sources separate into three roles:

- F&G is the best compact accuracy candidate.
- News is the best interpretability/calibration candidate because it improves
  ROC-AUC/Brier more clearly and has a correction/regression table.
- FSI/RORO are stable macro-risk context sources, but neither is strong enough
  to be selected alone.

Technical-only context is mostly redundant with the `ohlc_ma_vb` chart image.
That supports the current thesis logic: chart-derived indicators are already
partly encoded visually, while external regime sources are conceptually cleaner
but still weak in hard-decision improvement.

## Decision

Do not stack every context feature directly. Move to `4-N13-5A` first:
cross-context train-only feature audit. The next selected-combo FiLM should be
based on redundancy and Stage 2 error-correlation diagnostics, not on manual
feature accumulation.

## Artifacts

- [full comparison](../reports/tables/stage4_n13_5_context_source_comparison.csv)
- [compact comparison](../reports/tables/stage4_n13_5_context_source_comparison_compact.csv)
- [recommendation table](../reports/tables/stage4_n13_5_context_source_comparison_recommendation.csv)
- [builder script](../scripts/build_stage4_n13_5_context_source_comparison.py)
