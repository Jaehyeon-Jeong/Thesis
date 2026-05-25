# 4-I2 Structured Context Feature Builder

## English

Status: complete

Purpose:
- Build the first Stage 4 structured numeric context table for the fixed
  Stage 2 baseline configuration.
- Keep all context values leakage-safe: every feature is available at or before
  the image end date `t`.
- Fit context preprocessing statistics on the train split only.

Implemented files:
- `src/stage4_film/context/sources.py`
- `src/stage4_film/context/features.py`
- `src/stage4_film/context/normalization.py`
- `scripts/audit_stage4_context_sources.py`
- `scripts/build_stage4_context_features.py`

Primary context setting:
- Image/model target: `I60/R20/ohlc_ma_vb`.
- Context window: `60`.
- Features:
  - `fg_value`
  - `fg_mean_60`
  - `fg_delta_60`
  - `fg_std_60`
  - `bb_percent_b_60`
  - `bb_bandwidth_60`
  - `mfi_60`
  - `rv_60`

Source audit result:
- BTC source: local `btc_1d_data_2018_to_2025.csv`.
- F&G source: local `F&G_data/fear_greed_index.csv`.
- F&G cleaned rows: `2,644`.
- F&G date range: `2018-02-01` to `2025-05-02`.
- F&G missing calendar days inside range: `4`
  - `2018-04-14`
  - `2018-04-15`
  - `2018-04-16`
  - `2024-10-26`
- For the primary I60/R20 samples, F&G as-of coverage is complete:
  - Train missing F&G dates: `0`
  - Validation missing F&G dates: `0`
  - Test missing F&G dates: `0`
  - Max F&G as-of age: `1` day

Context table result:
- Total rows: `2,399`.
- Split counts:
  - Train: `671`
  - Validation: `287`
  - Test: `1,441`
- Date range: `2018-04-29` to `2024-12-11`.
- Missing-rate warnings: none.
- Primary feature missing rate is `0.0` for every split and every feature.

Preprocessing policy:
- F&G and MFI 0-100 values: divide by `100`.
- Bollinger bandwidth and realized volatility: `log1p`.
- Bollinger `%B`: identity.
- Train-only median imputation.
- Train-only 1/99% clipping.
- Train-only z-score normalization.

Generated local artifacts:
- `outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_features.csv`
- `outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_scaler.json`
- `outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_feature_audit.json`
- `outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_feature_summary.csv`

Tracked report copies:
- `reports/tables/stage4_context_source_audit.json`
- `reports/tables/stage4_context_i60_ohlc_ma_vb_r20_c60_seed42_context_feature_audit.json`
- `reports/tables/stage4_context_i60_ohlc_ma_vb_r20_c60_seed42_context_feature_summary.csv`

Validation:
- `python -m py_compile` passed for context modules and the two scripts.
- `python scripts/audit_stage4_context_sources.py --config configs/env_local.yaml`
  passed.
- `python scripts/build_stage4_context_features.py --config configs/env_local.yaml --write-report-copy`
  passed.

Notable implementation note:
- `historical_data.csv` in the local F&G dataset folder is not used in this
  numeric context run. It appears to be transaction/order-style data rather
  than the Fear & Greed index table.
- The raw F&G CSV files are not tracked in GitHub.

Next:
- `4-I3`: implement the small shared context MLP encoder.

## 한국어

상태: 완료

목적:
- Stage 2 고정 baseline 설정에 맞춰 Stage 4 첫 structured numeric context table을
  실제로 생성했습니다.
- 모든 context 값은 image end date `t` 또는 그 이전 정보만 사용합니다.
- context preprocessing 통계는 train split에서만 fit합니다.

구현 파일:
- `src/stage4_film/context/sources.py`
- `src/stage4_film/context/features.py`
- `src/stage4_film/context/normalization.py`
- `scripts/audit_stage4_context_sources.py`
- `scripts/build_stage4_context_features.py`

Primary context 설정:
- Image/model target: `I60/R20/ohlc_ma_vb`.
- Context window: `60`.
- Feature:
  - `fg_value`
  - `fg_mean_60`
  - `fg_delta_60`
  - `fg_std_60`
  - `bb_percent_b_60`
  - `bb_bandwidth_60`
  - `mfi_60`
  - `rv_60`

Source audit 결과:
- BTC source: local `btc_1d_data_2018_to_2025.csv`.
- F&G source: local `F&G_data/fear_greed_index.csv`.
- F&G cleaned rows: `2,644`.
- F&G date range: `2018-02-01` to `2025-05-02`.
- F&G 내부 calendar missing day: `4`
  - `2018-04-14`
  - `2018-04-15`
  - `2018-04-16`
  - `2024-10-26`
- Primary I60/R20 sample 기준 F&G as-of coverage는 완전합니다:
  - Train missing F&G dates: `0`
  - Validation missing F&G dates: `0`
  - Test missing F&G dates: `0`
  - Max F&G as-of age: `1` day

Context table 결과:
- Total rows: `2,399`.
- Split counts:
  - Train: `671`
  - Validation: `287`
  - Test: `1,441`
- Date range: `2018-04-29` to `2024-12-11`.
- Missing-rate warning: 없음.
- 8개 primary feature 모두 train/validation/test missing rate `0.0`.

Preprocessing 정책:
- F&G/MFI처럼 0-100 scale인 값: `100`으로 나눔.
- Bollinger bandwidth와 realized volatility: `log1p`.
- Bollinger `%B`: identity.
- Train-only median imputation.
- Train-only 1/99% clipping.
- Train-only z-score normalization.

생성된 local artifact:
- `outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_features.csv`
- `outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_scaler.json`
- `outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_feature_audit.json`
- `outputs/stage4/context/stage4_context_i60_ohlc_ma_vb_r20_c60/seed_42/context_feature_summary.csv`

GitHub에 올리는 report copy:
- `reports/tables/stage4_context_source_audit.json`
- `reports/tables/stage4_context_i60_ohlc_ma_vb_r20_c60_seed42_context_feature_audit.json`
- `reports/tables/stage4_context_i60_ohlc_ma_vb_r20_c60_seed42_context_feature_summary.csv`

검증:
- context module과 script 2개의 `python -m py_compile` 통과.
- `python scripts/audit_stage4_context_sources.py --config configs/env_local.yaml`
  통과.
- `python scripts/build_stage4_context_features.py --config configs/env_local.yaml --write-report-copy`
  통과.

특이사항:
- local F&G dataset folder의 `historical_data.csv`는 이번 numeric context run에
  사용하지 않습니다. Fear & Greed index table이라기보다 transaction/order 성격의
  데이터로 보입니다.
- Raw F&G CSV 파일은 GitHub에 track하지 않습니다.

다음:
- `4-I3`: 작은 shared context MLP encoder 구현.
