# Stage 2 Checklist

## English

Proceed one item at a time. Stage 2 may begin before Stage 1 full Kaggle results
finish, but Stage 2 final comparisons must wait for Stage 1 outputs.

Planning phase:
- [x] 2-0. Stage 2 folder and planning documents
- [x] 2-1. Source, Stage 1 dependency, and constraint re-check
  - Result: [2-1 source/dependency/constraint re-check](checklist_results/2-1_source_dependency_constraint_recheck.md)
- [ ] 2-2. BTC OHLCV data audit
- [ ] 2-3. BTC image-generation detail plan
- [ ] 2-4. BTC label, split, and normalization detail plan
- [ ] 2-5. BTC baseline CNN adaptation plan
- [ ] 2-6. BTC evaluation and trading-metric plan
- [ ] 2-7. BTC Grad-CAM plan
- [ ] 2-8. Kaggle runner and output plan

Implementation phase:
- [ ] 2-I0. Implementation readiness review
- [ ] 2-I1. Shared Stage 2 config/code scaffold
- [ ] 2-I2. BTC OHLCV loader
- [ ] 2-I3. BTC image generator
- [ ] 2-I4. BTC label/split/normalization code
- [ ] 2-I5. BTC baseline runner using Stage 1 CNN core
- [ ] 2-I6. BTC prediction and metric export
- [ ] 2-I7. BTC trading metric/backtest export
- [ ] 2-I8. BTC Grad-CAM export
- [ ] 2-I9. Local or small Kaggle smoke test
- [ ] 2-I10. Kaggle full BTC baseline runs
- [ ] 2-I11. Stage 2 report outputs

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
- [ ] 2-2. BTC OHLCV 데이터 audit
- [ ] 2-3. BTC image generation 세부계획
- [ ] 2-4. BTC label, split, normalization 세부계획
- [ ] 2-5. BTC baseline CNN adaptation 계획
- [ ] 2-6. BTC evaluation과 trading metric 계획
- [ ] 2-7. BTC Grad-CAM 계획
- [ ] 2-8. Kaggle runner와 output 계획

구현 단계:
- [ ] 2-I0. 구현 readiness review
- [ ] 2-I1. Stage 2 공통 config/code scaffold
- [ ] 2-I2. BTC OHLCV loader
- [ ] 2-I3. BTC image generator
- [ ] 2-I4. BTC label/split/normalization code
- [ ] 2-I5. Stage 1 CNN core를 사용하는 BTC baseline runner
- [ ] 2-I6. BTC prediction과 metric export
- [ ] 2-I7. BTC trading metric/backtest export
- [ ] 2-I8. BTC Grad-CAM export
- [ ] 2-I9. local 또는 작은 Kaggle smoke test
- [ ] 2-I10. Kaggle full BTC baseline run
- [ ] 2-I11. Stage 2 report output

중요:
- Stage 2로 넘어가도 Stage 1 CNN core는 임의로 바꾸지 않습니다. 바꿀 필요가 있으면
  checklist 항목에서 이유를 먼저 기록합니다.
- BTC 데이터는 작으므로 Stage 2 기본 batch size는 논문값 `128`을 사용합니다.
- BTC는 단일 자산이므로 stock cross-sectional high-minus-low decile portfolio를
  그대로 사용하지 않습니다.
