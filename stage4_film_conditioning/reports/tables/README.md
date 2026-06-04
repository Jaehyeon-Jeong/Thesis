# Stage 4 Tables

## English

Current planning tables:
- `stage4_implementation_task_map.csv`: implementation checklist item to file
  map fixed by `4-I0`.

Current implementation audit tables:
- `stage4_context_source_audit.json`: BTC/F&G source coverage audit for the
  primary Stage 4 context build.
- `stage4_context_i60_ohlc_ma_vb_r20_c60_seed42_context_feature_audit.json`:
  split counts, feature missing rates, and warnings.
- `stage4_context_i60_ohlc_ma_vb_r20_c60_seed42_context_feature_summary.csv`:
  split-level raw and normalized context feature summary.

Current result tables:
- `stage4_four_ablation_seed42_run_summary.csv`: Kaggle seed-42 result for
  `concat`, `gating`, `film_gamma`, and `film_full` on
  `I60/R20/ohlc_ma_vb`.
- `stage4_v2_v8_p7_p8_seed_collapse_*.csv`: P7/P8 seed-collapse diagnostic
  and validation-threshold calibration tables.
- `stage4_v2_v9_bounded_last_block_film_scale_grid_*.csv`: bounded
  last-block FiLM scale-grid result tables.

Current news-context tables:
- `stage4_news_source_audit.json`: selected BTC news source audit.
- `stage4_news_alignment_*.csv/json`: strict `t-1` publication-time alignment
  and 7/20/60-day coverage audit.
- `stage4_news_headline_windows_*.csv/json`: 7/20/60-day headline-window
  aggregation reports.
- `stage4_news_tfidf_svd_*.csv/json`: train-only TF-IDF/SVD vectorizer
  manifest, split summary, component summaries, top terms, and examples.
- `stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60_seed42_*`:
  final 4-N5 news context audit, feature summary, and manifest for the
  `102`-dimensional normalized news context vector.

Future result tables:
- `stage4_four_ablation_five_seed_seed_results.csv`
- `stage4_four_ablation_five_seed_mean_std_results.csv`

## 한국어

현재 planning table:
- `stage4_implementation_task_map.csv`: `4-I0`에서 고정한 구현 checklist와 파일
  mapping입니다.

현재 구현 audit table:
- `stage4_context_source_audit.json`: primary Stage 4 context build용 BTC/F&G
  source coverage audit입니다.
- `stage4_context_i60_ohlc_ma_vb_r20_c60_seed42_context_feature_audit.json`:
  split count, feature missing rate, warning을 기록합니다.
- `stage4_context_i60_ohlc_ma_vb_r20_c60_seed42_context_feature_summary.csv`:
  split별 raw/normalized context feature summary입니다.

현재 result table:
- `stage4_four_ablation_seed42_run_summary.csv`: `I60/R20/ohlc_ma_vb`에서
  `concat`, `gating`, `film_gamma`, `film_full`을 비교한 Kaggle seed-42 결과입니다.
- `stage4_v2_v8_p7_p8_seed_collapse_*.csv`: P7/P8 seed-collapse 진단과
  validation-threshold calibration table입니다.
- `stage4_v2_v9_bounded_last_block_film_scale_grid_*.csv`: bounded last-block
  FiLM scale-grid 결과 table입니다.

현재 news-context table:
- `stage4_news_source_audit.json`: 선택한 BTC news source audit입니다.
- `stage4_news_alignment_*.csv/json`: strict `t-1` publication-time alignment와
  7/20/60-day coverage audit입니다.
- `stage4_news_headline_windows_*.csv/json`: 7/20/60-day headline-window
  aggregation report입니다.
- `stage4_news_tfidf_svd_*.csv/json`: train-only TF-IDF/SVD vectorizer
  manifest, split summary, component summary, top term, example table입니다.
- `stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60_seed42_*`:
  `102`차원 normalized news context vector의 최종 4-N5 audit, feature
  summary, manifest입니다.

추후 result table:
- `stage4_four_ablation_five_seed_seed_results.csv`
- `stage4_four_ablation_five_seed_mean_std_results.csv`
