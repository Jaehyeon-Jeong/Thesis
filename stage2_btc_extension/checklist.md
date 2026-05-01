# Stage 2 Checklist

## English

Proceed one item at a time. Stage 2 may begin before Stage 1 full Kaggle results
finish, but Stage 2 final comparisons must wait for Stage 1 outputs.

Planning phase:
- [x] 2-0. Stage 2 folder and planning documents
- [x] 2-1. Source, Stage 1 dependency, and constraint re-check
  - Result: [2-1 source/dependency/constraint re-check](checklist_results/2-1_source_dependency_constraint_recheck.md)
- [x] 2-2. BTC OHLCV data audit
  - Result: [2-2 BTC OHLCV data audit](checklist_results/2-2_btc_ohlcv_data_audit.md)
- [x] 2-3. BTC image-generation detail plan
  - Result: [2-3 BTC image-generation detail plan](checklist_results/2-3_btc_image_generation_detail_plan.md)
- [x] 2-4. BTC label, split, and normalization detail plan
  - Result: [2-4 BTC label/split/normalization detail plan](checklist_results/2-4_btc_label_split_normalization_plan.md)
- [x] 2-5. BTC baseline CNN adaptation plan
  - Result: [2-5 BTC baseline CNN adaptation plan](checklist_results/2-5_btc_baseline_cnn_adaptation_plan.md)
- [x] 2-6. BTC evaluation and trading-metric plan
  - Result: [2-6 BTC evaluation/trading-metric plan](checklist_results/2-6_btc_evaluation_trading_metric_plan.md)
- [x] 2-7. BTC Grad-CAM plan
  - Result: [2-7 BTC Grad-CAM plan](checklist_results/2-7_btc_gradcam_plan.md)
- [x] 2-8. Kaggle runner and output plan
  - Result: [2-8 Kaggle runner/output plan](checklist_results/2-8_kaggle_runner_output_plan.md)

Implementation phase:
- [x] 2-I0. Implementation readiness review
  - Result: [2-I0 implementation readiness review](checklist_results/2-I0_implementation_readiness_review.md)
- [x] 2-I1. Shared Stage 2 config/code scaffold
  - Result: [2-I1 shared config/code scaffold](checklist_results/2-I1_shared_config_code_scaffold.md)
- [x] 2-I2. BTC OHLCV loader
  - Result: [2-I2 BTC OHLCV loader](checklist_results/2-I2_btc_ohlcv_loader.md)
- [x] 2-I3. BTC image generator
  - Result: [2-I3 BTC image generator](checklist_results/2-I3_btc_image_generator.md)
- [x] 2-I4. BTC label/split/normalization code
  - Result: [2-I4 BTC label/split/normalization code](checklist_results/2-I4_label_split_normalization_code.md)
- [x] 2-I5. BTC baseline runner using Stage 1 CNN core
  - Result: [2-I5 BTC baseline runner](checklist_results/2-I5_btc_baseline_runner.md)
- [x] 2-I6. BTC prediction and metric export
  - Result: [2-I6 prediction/metric export](checklist_results/2-I6_prediction_metric_export.md)
- [x] 2-I7. BTC trading metric/backtest export
  - Result: [2-I7 trading metric export](checklist_results/2-I7_trading_metric_export.md)
- [x] 2-I8. BTC Grad-CAM export
  - Result: [2-I8 BTC Grad-CAM export](checklist_results/2-I8_btc_gradcam_export.md)
- [x] 2-I9. Local or small Kaggle smoke test
  - Result: [2-I9 local smoke test](checklist_results/2-I9_local_smoke_test.md)
- [x] 2-I10. Kaggle full BTC baseline runs
  - Result: [2-I10 Kaggle runner ready](checklist_results/2-I10_kaggle_runner_ready.md)
- [x] 2-I11. Stage 2 grid runners and result viewer
  - Result: [2-I11 Stage 2 grid runners and result viewer](checklist_results/2-I11_stage2_grid_runners_and_viewer.md)

Important:
- Do not change the Stage 1 CNN core when moving to BTC unless a checklist item
  explicitly records why.
- Use batch size `128` by default for Stage 2 because BTC data is smaller.
- BTC is a single asset, so do not use stock cross-sectional high-minus-low
  decile portfolios.

## 한국어

한 항목씩 진행합니다. Stage 1 full Kaggle 결과가 아직 없어도 Stage 2 준비는 시작할 수
있지만, Stage 2 최종 비교는 Stage 1 output이 나온 뒤 확정합니다.

계획 단계:
- [x] 2-0. Stage 2 폴더와 planning 문서
- [x] 2-1. source, Stage 1 dependency, constraint 재확인
  - 결과: [2-1 source/dependency/constraint 재확인](checklist_results/2-1_source_dependency_constraint_recheck.md)
- [x] 2-2. BTC OHLCV 데이터 audit
  - 결과: [2-2 BTC OHLCV 데이터 audit](checklist_results/2-2_btc_ohlcv_data_audit.md)
- [x] 2-3. BTC image generation 세부계획
  - 결과: [2-3 BTC image-generation 세부계획](checklist_results/2-3_btc_image_generation_detail_plan.md)
- [x] 2-4. BTC label, split, normalization 세부계획
  - 결과: [2-4 BTC label/split/normalization 세부계획](checklist_results/2-4_btc_label_split_normalization_plan.md)
- [x] 2-5. BTC baseline CNN adaptation 계획
  - 결과: [2-5 BTC baseline CNN adaptation 계획](checklist_results/2-5_btc_baseline_cnn_adaptation_plan.md)
- [x] 2-6. BTC evaluation과 trading metric 계획
  - 결과: [2-6 BTC evaluation/trading-metric 계획](checklist_results/2-6_btc_evaluation_trading_metric_plan.md)
- [x] 2-7. BTC Grad-CAM 계획
  - 결과: [2-7 BTC Grad-CAM 계획](checklist_results/2-7_btc_gradcam_plan.md)
- [x] 2-8. Kaggle runner와 output 계획
  - 결과: [2-8 Kaggle runner/output 계획](checklist_results/2-8_kaggle_runner_output_plan.md)

구현 단계:
- [x] 2-I0. 구현 readiness review
  - 결과: [2-I0 구현 readiness review](checklist_results/2-I0_implementation_readiness_review.md)
- [x] 2-I1. Stage 2 공통 config/code scaffold
  - 결과: [2-I1 공통 config/code scaffold](checklist_results/2-I1_shared_config_code_scaffold.md)
- [x] 2-I2. BTC OHLCV loader
  - 결과: [2-I2 BTC OHLCV loader](checklist_results/2-I2_btc_ohlcv_loader.md)
- [x] 2-I3. BTC image generator
  - 결과: [2-I3 BTC image generator](checklist_results/2-I3_btc_image_generator.md)
- [x] 2-I4. BTC label/split/normalization code
  - 결과: [2-I4 BTC label/split/normalization code](checklist_results/2-I4_label_split_normalization_code.md)
- [x] 2-I5. Stage 1 CNN core를 사용하는 BTC baseline runner
  - 결과: [2-I5 BTC baseline runner](checklist_results/2-I5_btc_baseline_runner.md)
- [x] 2-I6. BTC prediction과 metric export
  - 결과: [2-I6 prediction/metric export](checklist_results/2-I6_prediction_metric_export.md)
- [x] 2-I7. BTC trading metric/backtest export
  - 결과: [2-I7 trading metric export](checklist_results/2-I7_trading_metric_export.md)
- [x] 2-I8. BTC Grad-CAM export
  - 결과: [2-I8 BTC Grad-CAM export](checklist_results/2-I8_btc_gradcam_export.md)
- [x] 2-I9. local 또는 작은 Kaggle smoke test
  - 결과: [2-I9 local smoke test](checklist_results/2-I9_local_smoke_test.md)
- [x] 2-I10. Kaggle full BTC baseline run
  - 결과: [2-I10 Kaggle runner ready](checklist_results/2-I10_kaggle_runner_ready.md)
- [x] 2-I11. Stage 2 grid runner와 result viewer
  - 결과: [2-I11 Stage 2 grid runner와 result viewer](checklist_results/2-I11_stage2_grid_runners_and_viewer.md)

중요:
- Stage 2로 넘어가도 Stage 1 CNN core는 임의로 바꾸지 않습니다. 바꿀 필요가 있으면
  checklist 항목에서 이유를 먼저 기록합니다.
- BTC 데이터는 작으므로 Stage 2 기본 batch size는 논문값 `128`을 사용합니다.
- BTC는 단일 자산이므로 stock cross-sectional high-minus-low decile portfolio를
  그대로 사용하지 않습니다.
