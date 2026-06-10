# 5-14. FinBERT + F&G Gamma/Beta Relaxation Ablation

## Purpose

This check tests whether the conservative bounded FiLM setup was too restrictive
for the final FinBERT + F&G context model.

Baseline protocol:

- Visual backbone: Stage 2 `I60/R20/ohlc_ma_vb`.
- Context: FinBERT headline sentiment + Fear and Greed features.
- Training protocol: Stage2 CNN and classifier frozen.
- Standard bounded FiLM scale: `gamma_scale=0.02`, `beta_scale=0.02`.

Relaxation variants:

| Variant | Gamma scale | Beta scale |
| --- | ---: | ---: |
| `gamma_relaxed_g0p10_b0p02` | `0.10` | `0.02` |
| `beta_relaxed_g0p02_b0p10` | `0.02` | `0.10` |

## Result

Both relaxation variants completed for seeds `42,43,44,45,46`.

Mean/std result:

| Variant | Accuracy | ROC-AUC | AP | F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gamma relaxed | `0.569466` | `0.588825` | `0.613885` | `0.613803` | `0.270900` |
| Beta relaxed | `0.569466` | `0.588825` | `0.613885` | `0.613803` | `0.270900` |

The two exports have different experiment names and checkpoint paths, but their
seed-level classification metrics are identical. This means they should not be
presented in the thesis as two independent empirical improvements.

## Interpretation

Relaxing one FiLM component to `0.10` did not improve accuracy relative to the
standard FinBERT + F&G bounded FiLM result.

Important comparison:

- Standard FinBERT + F&G bounded FiLM accuracy: `0.580569`.
- Gamma/beta relaxed accuracy: `0.569466`.

The relaxation therefore supports the thesis design choice: the conservative
`0.02` bounded FiLM protocol is not arbitrary; stronger one-sided modulation
can reduce hard-decision accuracy even if ranking metrics remain acceptable.

## Thesis Usage

Use this result only as a diagnostic ablation:

- It supports the restriction of FiLM modulation scale in the methodology.
- It should not appear as a separate positive result row in the main result
  table.
- It can be referenced briefly when explaining why the final model uses
  bounded modulation rather than relaxed gamma/beta correction.

## Artifacts

- `reports/tables/stage5_14_finbert_fg_gamma_beta_relaxation_mean_std_results.csv`
- `reports/tables/stage5_14_finbert_fg_gamma_beta_relaxation_seed_results.csv`
- `reports/tables/stage5_14_finbert_fg_gamma_beta_relaxation_manifest.json`
- `reports/tables/stage5_14_finbert_fg_gamma_beta_relax_gamma_relaxed_g0p10_b0p02_mean_std_results.csv`
- `reports/tables/stage5_14_finbert_fg_gamma_beta_relax_beta_relaxed_g0p02_b0p10_mean_std_results.csv`
