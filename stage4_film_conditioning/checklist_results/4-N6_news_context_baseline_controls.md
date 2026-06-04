# 4-N6 News-Context Baseline Controls

Status: prepared for Kaggle execution; five-seed result pending

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

## Result Pending

N6 should be marked complete only after the Kaggle five-seed CSVs are available:

- `stage4_news_context_n6_concat_five_seed_seed_results.csv`
- `stage4_news_context_n6_concat_five_seed_mean_std_results.csv`

## Decision Rule

- If `CNN + news concat` improves or stays close to the Stage 2 visual baseline,
  proceed to `4-N7` bounded last-block FiLM with news context.
- If `CNN + news concat` is clearly worse, inspect seed collapse and news
  coverage before adding FiLM.
