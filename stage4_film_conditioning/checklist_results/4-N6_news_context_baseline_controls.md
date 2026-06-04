# 4-N6 News-Context Baseline Controls

Status: completed; five-seed Kaggle result available

## Goal

Test whether the headline-only news vector is useful as side information before
claiming that FiLM-news modulation improves the visual baseline.

## Fixed Setup

- Visual input: `I60/R20/ohlc_ma_vb`
- Reference baseline: Stage 2 selected five-seed visual CNN
- News windows: `7d`, `20d`, `60d`
- News vector: train-only TF-IDF/SVD from headline windows
- Context dimension: `102`
  - `96` SVD features = `32` components x `3` windows
  - `6` log-count features = news count and unique-source count x `3` windows
- First control model: `CNN + news concat`
- Seeds: `42, 43, 44, 45, 46`

## Implementation Prepared

Added prebuilt context support to the Stage 4 runner:

- `context.source = prebuilt_news`
- `context.prebuilt_context_name =
  stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60`
- `context_scaler.json` supplies the normalized feature order.
- The model runner consumes the existing N5 `context_features.csv` rather than
  rebuilding structured F&G/technical context.

Prepared Kaggle one-cell runner:

- [kaggle_stage4_news_context_n6_baseline_controls_one_cell.md](../notebooks/kaggle_stage4_news_context_n6_baseline_controls_one_cell.md)

The Kaggle cell:

1. Copies Stage 4 and Stage 2 code snapshots.
2. Rebuilds N3/N4 news vectors if missing.
3. Builds N5 prebuilt news context artifacts for five seeds.
4. Patches `env_kaggle.yaml` to use `prebuilt_news`.
5. Runs `CNN + news concat` over five seeds.
6. Exports seed-level and mean/std CSV tables.
7. Writes one compact download bundle:
   `/kaggle/working/stage4_news_context_n6_result_bundle.zip`.

## Local Smoke Verification

Local small-sample smoke passed:

- training: passed
- prediction evaluation: passed
- trading evaluation: passed
- Grad-CAM/context export: passed
- output checker: passed

Smoke experiment:

```text
stage4_concat_i60_ohlc_ma_vb_r20_c60_news_tfidf_svd_w7_20_60_smoke
```

The smoke confirmed that the Stage 4 runner can load the `102`-dimensional
prebuilt news context table and align it to BTC samples through `sample_index`.

## Five-Seed Result

Kaggle five-seed result files:

- `stage4_news_context_n6_concat_five_seed_seed_results.csv`
- `stage4_news_context_n6_concat_five_seed_mean_std_results.csv`

Summary:

| Metric | Mean | Std |
| --- | ---: | ---: |
| Accuracy | `0.5478` | `0.0527` |
| ROC-AUC | `0.5644` | `0.0418` |
| F1 | `0.5283` | `0.2964` |
| Predicted positive rate | `0.5885` | `0.3627` |
| Long/flat Sharpe net | `2.7438` | `1.5455` |

Seed behavior:

| Seed | Accuracy | ROC-AUC | Predicted positive rate | Note |
| ---: | ---: | ---: | ---: | --- |
| `42` | `0.5822` | `0.5971` | `0.6565` | useful signal |
| `43` | `0.5420` | `0.5145` | `0.9993` | near all-Up collapse |
| `44` | `0.5697` | `0.5835` | `0.6607` | useful signal |
| `45` | `0.4587` | `0.5244` | `0.0000` | all-Down collapse |
| `46` | `0.5864` | `0.6027` | `0.6260` | useful signal |

Interpretation:

- Headline-only news context has useful signal in some seeds.
- The `102`-dimensional context is too unstable to move directly into FiLM as
  the main model.
- Before 4-N7, test smaller SVD dimensions (`16`, `8`) with the same concat
  control. This separates “news vector instability” from “FiLM instability.”

## Decision Rule

- If `CNN + news concat` improves or stays close to the Stage 2 visual baseline,
  proceed to `4-N7` bounded last-block FiLM with news context.
- If `CNN + news concat` is clearly worse, inspect seed collapse and news
  coverage before adding FiLM.

Decision:

- Proceed to 4-N6.1 SVD-dimension stability grid first.
- Then run 4-N7 bounded last-block FiLM using the most stable news vector size.
