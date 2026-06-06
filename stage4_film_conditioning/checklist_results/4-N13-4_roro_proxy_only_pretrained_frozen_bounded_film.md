# 4-N13-4 RORO-Proxy-Only Frozen Bounded FiLM

## Purpose

Test whether a KC Fed-inspired public-data risk-on/risk-off proxy improves the
strongest Stage 2 BTC visual baseline when the Stage 2 CNN and classifier are
preserved.

## Protocol

- Visual baseline: Stage 2 `I60/R20/ohlc_ma_vb`
- Context source: public-data RORO proxy from VIX, S&P 500, DXY, and US 10Y
- Model: Stage 2 CNN/classifier loaded and frozen
- Trainable module: RORO context encoder + bounded final-block FiLM head
- FiLM method: `film_full_bounded_last_block`
- Scale: `0.02`
- Seeds: `42, 43, 44, 45, 46`
- Feature sets:
  - `roro_2`: `roro_proxy_mean_60`, `roro_proxy_delta_60`
  - `roro_3`: `roro_proxy_mean_60`, `roro_proxy_delta_60`, `roro_proxy_std_60`
  - `roro_proxy_all`: six RORO proxy features

## Result

| Context | Accuracy | ROC-AUC | Brier | F1 | Net Correction | Collapse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 2 frozen baseline | 0.579320 | 0.584862 | 0.274337 | 0.651071 | - | - |
| `roro_2` | 0.579181 | 0.584799 | 0.274254 | 0.650681 | -0.2 mean | 0 |
| `roro_3` | 0.579320 | 0.584748 | 0.274278 | 0.650924 | 0.0 mean | 0 |
| `roro_proxy_all` | 0.579181 | 0.584872 | 0.274121 | 0.650382 | -0.2 mean | 0 |

## Interpretation

The RORO proxy attaches safely under the frozen Stage 2 bounded-FiLM protocol:
no seed produced a class-collapse warning. However, the context barely changes
the Stage 2 decisions. `roro_3` ties Stage 2 accuracy but slightly lowers
ROC-AUC and F1; `roro_proxy_all` gives the best Brier score among the three RORO
sets, but the gain is too small to use as a main performance claim.

The high seed-42 accuracy is not evidence that RORO is strong by itself. The
Stage 2 seed-42 checkpoint already reaches about `0.6031` accuracy; RORO changes
only one or two decisions per seed. Therefore the right conclusion is
baseline-preservation, not a meaningful macro-context improvement.

Compared with prior context sources, RORO-only is weaker than the best compact
F&G-only row and roughly tied with FSI/technical-only:

- Best compact row remains N8-B F&G-only scale `0.02`:
  accuracy `0.580291`.
- Best FSI row is `fsi_all`:
  accuracy `0.579875`.
- Best RORO row is `roro_3`:
  accuracy `0.579320`.

## Decision

Do not select RORO-only as the final Stage 4 model. Keep it as macro/risk-off
context evidence and move to `4-N13-5/5A`: compare all context sources under the
same frozen protocol, then audit cross-context redundancy/correlation before
trying a small selected-combo context.

## Artifacts

- [seed results](../reports/tables/stage4_n13_4_roro_only_pretrained_frozen_bounded_film_seed_results.csv)
- [mean/std results](../reports/tables/stage4_n13_4_roro_only_pretrained_frozen_bounded_film_mean_std_results.csv)
- [run summary](../reports/tables/stage4_n13_4_roro_only_pretrained_frozen_bounded_film_run_summary.json)
