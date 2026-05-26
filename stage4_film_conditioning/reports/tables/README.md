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

추후 result table:
- `stage4_four_ablation_five_seed_seed_results.csv`
- `stage4_four_ablation_five_seed_mean_std_results.csv`
