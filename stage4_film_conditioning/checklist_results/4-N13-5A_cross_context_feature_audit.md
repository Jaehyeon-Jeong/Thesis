# 4-N13-5A Cross-Context Feature Audit

## Goal

Audit all available Stage 4 context sources on the same `I60/R20/ohlc_ma_vb`
sample universe before trying a selected-combo FiLM model.

This step does **not** train a model. It only checks which context features have
usable train-split signal and which features are redundant.

## Inputs

- Base/F&G context: P7/P8 full artifact, seed 42, 2,399 rows.
- News context: headline TF-IDF/SVD `7d/20d/60d`, 102 normalized features.
- Technical context: recomputed from BTC OHLCV with the existing Stage 4
  formulas: `BB%_60`, `BB bandwidth_60`, `MFI_60`, `RV_60`.
- FSI context: OFR FSI `value/mean/delta/std` features.
- RORO context: public-data proxy from VIX, S&P500, DXY, US 10Y.
- Stage 2 baseline predictions: seed 42-46 `I60/R20/ohlc_ma_vb`.

## Leakage Rule

Feature selection diagnostics use the **train split only**:

`score = max(abs(corr(label)), abs(corr(future_return)), abs(spearman(future_return)))`

Stage 2 error-rate correlation is reported only as a **test diagnostic** and is
not used to choose selected-combo features.

## Output Tables

- [feature audit](../reports/tables/stage4_n13_5a_cross_context_feature_audit.csv)
- [family summary](../reports/tables/stage4_n13_5a_cross_context_family_summary.csv)
- [selected candidates](../reports/tables/stage4_n13_5a_cross_context_selected_feature_candidates.csv)
- [redundancy pairs](../reports/tables/stage4_n13_5a_cross_context_redundancy_pairs.csv)
- [Stage 2 error diagnostics](../reports/tables/stage4_n13_5a_cross_context_stage2_error_diagnostics.csv)
- [manifest](../reports/tables/stage4_n13_5a_cross_context_manifest.json)

## Family Summary

| Family | Features | Best train-only score | Best feature | Max abs corr with Stage2 error |
| --- | ---: | ---: | --- | ---: |
| News | 102 | 0.3500 | `news_svd_60d_09` | 0.1146 |
| Fear & Greed | 4 | 0.1808 | `fg_mean_60` | 0.1241 |
| FSI | 6 | 0.1692 | `ofr_fsi_std_60` | 0.0892 |
| Technical | 4 | 0.1510 | `bb_bandwidth_60` | 0.1001 |
| RORO | 10 | 0.1405 | `riskoff_dollar_return_20` | 0.1133 |

## Selected Feature Candidates

The selected list is capped by feature family and by redundancy.

| Rank | Family | Feature | Train-only score | Test corr with Stage2 error |
| ---: | --- | --- | ---: | ---: |
| 1 | News | `news_svd_60d_09` | 0.3500 | -0.0111 |
| 2 | News | `news_svd_20d_09` | 0.3406 | -0.0579 |
| 3 | News | `news_svd_20d_18` | 0.3121 | 0.0746 |
| 4 | F&G | `fg_mean_60` | 0.1808 | -0.0581 |
| 5 | FSI | `ofr_fsi_std_60` | 0.1692 | 0.0892 |
| 6 | F&G | `fg_delta_60` | 0.1596 | -0.1003 |
| 7 | Technical | `bb_bandwidth_60` | 0.1510 | 0.0246 |
| 8 | FSI | `ofr_fsi_mean_60` | 0.1469 | -0.0112 |
| 9 | RORO | `riskoff_dollar_return_20` | 0.1405 | -0.1133 |
| 10 | Technical | `bb_percent_b_60` | 0.1287 | -0.0402 |
| 11 | Technical | `rv_60` | 0.1241 | 0.0241 |
| 12 | RORO | `roro_proxy_std_60` | 0.1211 | 0.0764 |

## Interpretation

News has the strongest train-only relationship with the label/future return, but
many news SVD dimensions are highly redundant across `7d/20d/60d` windows.

F&G remains the cleanest compact regime source: it has fewer features, moderate
train-only signal, and prior five-seed results already showed the best small
accuracy lift.

FSI/RORO are stable risk-regime sources, but their train-only signal is weaker
than the top news dimensions and their prior model results only tied Stage 2.

Technical features are mostly redundant with the chart image. `bb_bandwidth_60`
and `bb_percent_b_60` have some train-only signal, but prior technical-only FiLM
did not materially beat Stage 2.

The Stage 2 error diagnostic is weak across all families. This means there is no
clear single feature that explains where Stage 2 fails. Therefore, the next
selected-combo experiment should be small and conservative, not a large
all-context vector.

## Suggested 4-N13-5B Candidate

Use a compact feature set:

- `news_svd_60d_09`
- `news_svd_20d_18`
- `fg_mean_60`
- `fg_delta_60`
- `ofr_fsi_std_60`
- `riskoff_dollar_return_20`

Optional technical add-on only if a 6-feature combo is too narrow:

- `bb_bandwidth_60`

Protocol:

- Stage 2 CNN/classifier frozen.
- Bounded last-block FiLM.
- Five seeds: `42,43,44,45,46`.
- Start with scale `0.02`; run `0.05` only if stable and too conservative.
