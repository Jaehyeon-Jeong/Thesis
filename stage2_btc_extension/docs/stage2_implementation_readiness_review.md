# Stage 2 Implementation Readiness Review

## English

Status: checklist `2-I0` complete.

Verdict:
- Stage 2 is ready to move from planning to implementation.
- The next checklist item should be `2-I1`: shared Stage 2 config/code
  scaffold.
- Stage 1 full paper-style outputs are still needed for final comparison
  tables, but they do not block Stage 2 implementation.

## Checked Inputs

Planning documents completed before this review:
- `docs/stage2_image_generation_plan.md`
- `docs/stage2_label_split_normalization_plan.md`
- `docs/stage2_baseline_cnn_adaptation_plan.md`
- `docs/stage2_evaluation_trading_metric_plan.md`
- `docs/stage2_gradcam_plan.md`
- `docs/stage2_kaggle_runner_output_plan.md`
- `docs/source_map.md`

Small planning artifacts completed:
- `reports/data_audit/btc_ohlcv_audit.md`
- `reports/tables/stage2_label_split_plan_counts.csv`
- `reports/tables/stage2_baseline_cnn_architecture_plan.csv`
- `reports/tables/stage2_metric_schema.csv`
- `reports/tables/stage2_gradcam_output_schema.csv`
- `reports/tables/stage2_kaggle_run_matrix.csv`

Primary local data:
- `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV/btc_1d_data_2018_to_2025.csv`

## Ready Decisions

Data:
- Use daily BTC OHLCV file.
- Canonical columns are `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
- No daily resampling is needed for the audited baseline file.
- Report baseline period is capped at `2024-12-31`; 2025-2026 is optional later
  holdout.

Images:
- Generate `I5`, `I20`, and `I60`.
- Compare four specs:
  `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`.
- Compute SMA from close prices:
  5-day, 20-day, and 60-day SMA for the corresponding image window.
- All images remain 1-channel grayscale.

Labels and split:
- Future return:
  `Close_{t+R} / Close_t - 1`.
- Label:
  `1` if future return is positive, otherwise `0`.
- Train/validation pool:
  signal dates `2018-01-01` to `2020-12-31`.
- Train/validation split:
  70/30 random split with seed `42`.
- Test:
  signal dates `2021-01-01` to `2024-12-31`.
- Period-end purge:
  require `label_end_date <= split_signal_end`.
- Pixel normalization:
  fit on training images only, per experiment tuple.

Models:
- Select model by image window, not by return horizon.
- `I5`: `stock_cnn_i5`.
- `I20`: `stock_cnn_i20`.
- `I60`: `stock_cnn_i60`.
- I20 can reuse the Stage 1/GitHub-style core.
- I5 and I60 require separate variants and must not silently reuse I20 reshape
  or classifier dimensions.

Evaluation:
- Use classification metrics plus BTC single-asset trading metrics.
- Do not use stock cross-sectional H-L decile portfolios.
- Store predictions with date, label, future return, logits, probabilities,
  predicted class, and correctness.
- Use `long_flat` and `long_short` strategies.

Grad-CAM:
- Required for BTC baseline.
- Use pre-softmax class logits as target scores.
- Save Figure-13-style grids.
- Interpret heatmaps as class-discriminative Grad-CAM, not raw feature maps.

Kaggle:
- Use one-cell wrapper pattern.
- Actual implementation remains in `src/` and `scripts/`.
- Run one experiment tuple at a time.
- Default Stage 2 batch size is `128`.

## Remaining Constraints

These are not blockers for `2-I1`, but must remain visible:
- Stage 1 full outputs for `I20/R5`, `I20/R20`, and `I20/R60` are still needed
  before final Stage 1 vs Stage 2 comparison.
- Stage 2 BTC train/validation random split follows the paper style, but BTC is
  a single rolling time series, so chronological validation should remain a
  later robustness check.
- Transaction cost defaults are implementation assumptions and must be reported
  as configurable.
- The Kaggle Stage 2 baseline one-cell file is currently an interface draft
  until implementation scripts exist.

## Implementation Order

Proceed in this order:
1. `2-I1`: create shared config/code scaffold.
2. `2-I2`: implement BTC OHLCV loader.
3. `2-I3`: implement BTC image generator.
4. `2-I4`: implement label/split/normalization code.
5. `2-I5`: implement BTC baseline runner and model variants.
6. `2-I6`: implement prediction and classification metric export.
7. `2-I7`: implement trading metric/backtest export.
8. `2-I8`: implement BTC Grad-CAM export.
9. `2-I9`: run local or small Kaggle smoke test.
10. `2-I10`: run full BTC baseline Kaggle experiments.
11. `2-I11`: assemble Stage 2 report outputs.

Source/task mapping:
- See `reports/tables/stage2_implementation_task_map.csv`.

## 한국어

상태: checklist `2-I0` 완료.

판정:
- Stage 2는 planning에서 implementation으로 넘어갈 준비가 됐습니다.
- 다음 체크리스트는 `2-I1`: Stage 2 공통 config/code scaffold입니다.
- Stage 1 full paper-style output은 최종 비교표에는 필요하지만, Stage 2 구현을
  시작하는 blocker는 아닙니다.

## 확인한 입력

이번 review 전에 완료된 planning 문서:
- `docs/stage2_image_generation_plan.md`
- `docs/stage2_label_split_normalization_plan.md`
- `docs/stage2_baseline_cnn_adaptation_plan.md`
- `docs/stage2_evaluation_trading_metric_plan.md`
- `docs/stage2_gradcam_plan.md`
- `docs/stage2_kaggle_runner_output_plan.md`
- `docs/source_map.md`

완료된 작은 planning artifact:
- `reports/data_audit/btc_ohlcv_audit.md`
- `reports/tables/stage2_label_split_plan_counts.csv`
- `reports/tables/stage2_baseline_cnn_architecture_plan.csv`
- `reports/tables/stage2_metric_schema.csv`
- `reports/tables/stage2_gradcam_output_schema.csv`
- `reports/tables/stage2_kaggle_run_matrix.csv`

주요 로컬 데이터:
- `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV/btc_1d_data_2018_to_2025.csv`

## 구현에 사용할 확정 결정

Data:
- daily BTC OHLCV file을 사용합니다.
- canonical column은 `Date`, `Open`, `High`, `Low`, `Close`, `Volume`입니다.
- audit된 baseline file에서는 daily resampling이 필요 없습니다.
- 기본 보고 기간은 `2024-12-31`까지이고, 2025-2026은 optional later holdout입니다.

Images:
- `I5`, `I20`, `I60`을 생성합니다.
- 네 가지 spec을 비교합니다:
  `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`.
- close price에서 SMA를 계산합니다:
  해당 image window에 맞춰 5-day, 20-day, 60-day SMA를 사용합니다.
- 모든 image는 1-channel grayscale입니다.

Labels and split:
- Future return:
  `Close_{t+R} / Close_t - 1`.
- Label:
  future return이 양수이면 `1`, 아니면 `0`.
- Train/validation pool:
  signal date `2018-01-01`부터 `2020-12-31`.
- Train/validation split:
  seed `42`로 70/30 random split.
- Test:
  signal date `2021-01-01`부터 `2024-12-31`.
- Period-end purge:
  `label_end_date <= split_signal_end`를 요구합니다.
- Pixel normalization:
  experiment tuple별 train image에서만 fit합니다.

Models:
- model은 return horizon이 아니라 image window로 선택합니다.
- `I5`: `stock_cnn_i5`.
- `I20`: `stock_cnn_i20`.
- `I60`: `stock_cnn_i60`.
- I20은 Stage 1/GitHub식 core를 재사용할 수 있습니다.
- I5와 I60은 별도 variant가 필요하며 I20 reshape/classifier dimension을 조용히
  재사용하면 안 됩니다.

Evaluation:
- classification metric과 BTC 단일 자산 trading metric을 함께 사용합니다.
- stock cross-sectional H-L decile portfolio는 사용하지 않습니다.
- prediction에는 date, label, future return, logits, probabilities, predicted
  class, correctness를 저장합니다.
- `long_flat`, `long_short` strategy를 사용합니다.

Grad-CAM:
- BTC baseline에서도 필수입니다.
- target score는 softmax 이전 class logit입니다.
- Figure-13-style grid를 저장합니다.
- heatmap은 raw feature map이 아니라 class-discriminative Grad-CAM으로 해석합니다.

Kaggle:
- one-cell wrapper pattern을 사용합니다.
- 실제 구현은 `src/`와 `scripts/`에 둡니다.
- experiment tuple 하나씩 실행합니다.
- Stage 2 기본 batch size는 `128`입니다.

## 남은 제한사항

아래 항목은 `2-I1` blocker는 아니지만 계속 명시해야 합니다.
- Stage 1 vs Stage 2 최종 비교 전에는 Stage 1 `I20/R5`, `I20/R20`, `I20/R60`
  full output이 필요합니다.
- Stage 2 BTC train/validation random split은 논문 방식에 맞춘 것이지만, BTC는
  단일 rolling time series이므로 chronological validation은 나중 robustness check로
  남깁니다.
- transaction cost 기본값은 구현상 가정이므로 configurable assumption으로 보고합니다.
- Kaggle Stage 2 baseline one-cell file은 implementation script가 생기기 전까지
  interface draft입니다.

## 구현 순서

이 순서대로 진행합니다.
1. `2-I1`: shared config/code scaffold 생성.
2. `2-I2`: BTC OHLCV loader 구현.
3. `2-I3`: BTC image generator 구현.
4. `2-I4`: label/split/normalization code 구현.
5. `2-I5`: BTC baseline runner와 model variants 구현.
6. `2-I6`: prediction과 classification metric export 구현.
7. `2-I7`: trading metric/backtest export 구현.
8. `2-I8`: BTC Grad-CAM export 구현.
9. `2-I9`: local 또는 작은 Kaggle smoke test.
10. `2-I10`: full BTC baseline Kaggle experiment.
11. `2-I11`: Stage 2 report output 정리.

Source/task mapping:
- `reports/tables/stage2_implementation_task_map.csv`를 참고합니다.
