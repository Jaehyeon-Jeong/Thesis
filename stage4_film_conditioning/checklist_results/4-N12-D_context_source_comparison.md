# 4-N12-D Context-Source Comparison

Status: completed for existing Stage 2 frozen context sources; `news + F&G` is recorded as a planned but not-yet-run complementarity check.

## Protocol

- Fixed visual backbone: Stage 2 `I60/R20/ohlc_ma_vb` checkpoints.
- Freeze policy: CNN and classifier are frozen; only context encoder plus final-block FiLM/gate heads are trained.
- Evaluation: five seeds `42-46`, test split, same `1441` BTC samples per seed.
- Main question: which context source is defensible for Stage 4 before moving to macro/RORO?

## Main Comparison

| Model | Context | Status | Acc | dAcc | ROC-AUC | dROC | F1 | dF1 | Brier | dBrier | Pred-Up | Correction/Regression |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage2 frozen baseline | visual-only ohlc_ma_vb | completed | 0.579320 | 0.000000 | 0.584862 | 0.000000 | 0.651071 | 0.000000 | 0.274337 | 0.000000 | 0.664400 | baseline |
| N8-B F&G-only bounded FiLM s0.02 | F&G-only external crypto regime | completed | 0.580291 | 0.000972 | 0.584930 | 0.000068 | 0.650814 | -0.000257 | 0.274004 | -0.000333 | 0.660652 | prediction bundle needed |
| N9 grid news SVD32 s0.02 | headline news TF-IDF/SVD | completed | 0.579736 | 0.000416 | 0.585353 | 0.000491 | 0.649307 | -0.001764 | 0.273765 | -0.000572 | 0.657321 | 27/24, net 3 |
| N9 grid news SVD8 s0.05 | headline news TF-IDF/SVD | completed | 0.578626 | -0.000694 | 0.586619 | 0.001757 | 0.642980 | -0.008092 | 0.272327 | -0.002010 | 0.639001 | prediction bundle needed |
| N12-C technical-only bounded FiLM s0.02 | BB60/MFI60/RV60 chart-derived technical context | completed | 0.579736 | 0.000416 | 0.584778 | -0.000084 | 0.650834 | -0.000237 | 0.274218 | -0.000119 | 0.662318 | prediction bundle needed |
| News + F&G combined bounded FiLM | headline news TF-IDF/SVD + F&G-only | not_run |  |  |  |  |  |  |  |  |  | not run |

## Interpretation

1. F&G-only is the best compact accuracy candidate, but the improvement over Stage 2 is only about `+0.001` accuracy. This is useful as a baseline-preserving context-FiLM result, not as a strong SOTA claim.
2. News-only has the clearest ROC-AUC/Brier signal. The best ROC row is `SVD8/scale0.05`, but it lowers accuracy/F1 and shifts predicted-Up rate downward. This means news changes probability ranking more than hard decisions.
3. Technical-only is effectively tied with Stage 2. That supports the current thesis interpretation: `ohlc_ma_vb` images already contain much of the BB/MFI/RV-like information, so adding those features as context is mostly redundant.
4. N12-A/B gated news variants do not produce a meaningful classifier gain. They are diagnostics for modulation behavior, not final models.
5. The only available prediction-level correction table is for the N10 news SVD32 model: `27` corrections vs `24` regressions, net `+3` over `7205` seed-samples. This is too small for a performance claim, but still useful for targeted Grad-CAM/gamma-beta examples.

## Decision

- Keep Stage 2 `I60/R20/ohlc_ma_vb` as the main baseline.
- Keep N8-B F&G-only scale `0.02` as the best compact context-FiLM candidate.
- Keep N9/N10 news SVD32 scale `0.02` for interpretability samples, not as a strong accuracy claim.
- Do not continue BB/MFI/RV-only tuning unless a reviewer asks for a technical-context ablation.
- Move to N13 macro/RORO context because it is image-external and better aligned with the market-regime motivation.

## Outputs

- Main table: `reports/tables/stage4_n12d_context_source_comparison.csv`
- Compact table: `reports/tables/stage4_n12d_context_source_comparison_compact.csv`
- Recommendation table: `reports/tables/stage4_n12d_context_source_recommendation.csv`
