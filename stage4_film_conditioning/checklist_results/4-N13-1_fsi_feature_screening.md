# 4-N13-1 OFR FSI Feature Screening

## Purpose

Screen the six OFR FSI candidate features before running the frozen Stage 2
FiLM model. The goal is to avoid treating all six FSI features as automatically
correct. Feature selection must use train/validation evidence only; test values
are reported as a diagnostic but not used for selection.

## Inputs

- Context artifact: `stage4_fsi_context_i60_ohlc_ma_vb_r20_ofr_fsi_lag1_w20_60`
- Split: Stage 4 `I60/R20/ohlc_ma_vb`
- Features:
  - `ofr_fsi_value`
  - `ofr_fsi_mean_20`
  - `ofr_fsi_mean_60`
  - `ofr_fsi_delta_20`
  - `ofr_fsi_delta_60`
  - `ofr_fsi_std_60`

## Diagnostics

- Univariate train/validation AUC against the Up/Down label.
- Pearson correlation with the label.
- Spearman correlation with future return.
- Train feature-feature correlation matrix to identify redundant features.

## Result

The most useful compact candidate is:

```text
FSI-2 = ofr_fsi_mean_60 + ofr_fsi_delta_60
```

Reason:

- `ofr_fsi_mean_60` has the strongest stable train/validation univariate signal
  among the six FSI features.
- `ofr_fsi_delta_60` has useful validation signal and is much less redundant
  with `ofr_fsi_mean_60` than `value`, `mean_20`, or `std_60`.
- `ofr_fsi_value`, `ofr_fsi_mean_20`, and `ofr_fsi_std_60` are highly
  correlated with each other, so using all six features may inject redundant
  regime context into FiLM.

The next N13-2 Kaggle runner therefore tests:

```text
FSI-2   = mean_60 + delta_60
FSI-3   = mean_60 + delta_60 + std_60
FSI-all = all six FSI features
```

## Artifacts

- [selection view](../reports/tables/stage4_n13_1_fsi_feature_selection_view.csv)
- [univariate diagnostics](../reports/tables/stage4_n13_1_fsi_feature_univariate_diagnostics.csv)
- [train correlation matrix](../reports/tables/stage4_n13_1_fsi_feature_train_correlation_matrix.csv)

