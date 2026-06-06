# 4-N13-2 FSI-Only Frozen Bounded FiLM

## Purpose

Test whether official macro financial-stress context improves the strongest
Stage 2 BTC visual baseline when the Stage 2 CNN and classifier are preserved.

## Protocol

- Visual baseline: Stage 2 `I60/R20/ohlc_ma_vb`
- Context source: OFR Financial Stress Index
- Model: Stage 2 CNN/classifier loaded and frozen
- Trainable module: FSI context encoder + bounded final-block FiLM head
- FiLM method: `film_full_bounded_last_block`
- Scale: `0.02`
- Seeds: `42, 43, 44, 45, 46`
- Feature sets:
  - `fsi_2`: `ofr_fsi_mean_60`, `ofr_fsi_delta_60`
  - `fsi_3`: `ofr_fsi_mean_60`, `ofr_fsi_delta_60`, `ofr_fsi_std_60`
  - `fsi_all`: all six FSI features

## Result

| Context | Accuracy | ROC-AUC | Brier | F1 | Net Correction | Collapse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 2 frozen baseline | 0.579320 | 0.584862 | 0.274337 | 0.651071 | - | - |
| `fsi_2` | 0.579598 | 0.584863 | 0.274256 | 0.651058 | +2 total | 0 |
| `fsi_3` | 0.579459 | 0.584807 | 0.274250 | 0.650844 | +1 total | 0 |
| `fsi_all` | 0.579875 | 0.584859 | 0.274190 | 0.651319 | +4 total | 0 |

## Interpretation

FSI context can be attached safely under the frozen Stage 2 FiLM protocol: no
seed collapse was observed for any feature set. However, the gains are extremely
small. The best FSI row is `fsi_all`, with accuracy +0.000555 versus Stage 2 and
net correction +4 across five seeds. This is not strong enough to claim a
meaningful performance improvement.

Compared with prior context sources, FSI is roughly tied with technical-only
context and weaker than the best F&G-only row:

- Best compact row remains N8-B F&G-only scale `0.02`:
  accuracy `0.580291`.
- Best FSI row is `fsi_all`:
  accuracy `0.579875`.
- FSI has no collapse, which is useful, but it does not materially outperform
  the frozen visual baseline.

## Decision

Do not select FSI-only as the final Stage 4 model. Keep it as macro/risk-off
context evidence and proceed to `4-N13-3/4`: build a public-data RORO proxy and
test whether a more directly risk-on/risk-off aligned vector is stronger than
OFR FSI.

## Artifacts

- [seed results](../reports/tables/stage4_n13_2_fsi_only_pretrained_frozen_bounded_film_seed_results.csv)
- [mean/std results](../reports/tables/stage4_n13_2_fsi_only_pretrained_frozen_bounded_film_mean_std_results.csv)
- [comparison with prior context sources](../reports/tables/stage4_n13_2_with_prior_context_comparison_compact.csv)

