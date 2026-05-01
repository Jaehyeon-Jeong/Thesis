# 2-5 BTC Baseline CNN Adaptation Plan

## English

Status: complete.

This checklist item fixes how the Stage 1 Stock_CNN-style baseline will be
adapted to BTC images.

Key decisions:
- Do not force all BTC images through the I20-only Stage 1 model.
- Use one model variant per image window:
  - `I5` -> `stock_cnn_i5`
  - `I20` -> `stock_cnn_i20`
  - `I60` -> `stock_cnn_i60`
- Select the CNN by image window, not by return horizon.
- Keep all four image specs as one-channel grayscale inputs; MA and volume do
  not create extra channels.
- Keep Stage 1 training defaults for the baseline:
  cross-entropy, Adam, learning rate `1e-5`, batch size `128`, dropout `0.5`,
  Xavier initialization, validation-loss early stopping with patience `2`.
- Train the default BTC baseline from scratch. Do not transfer a stock
  checkpoint unless a later checklist item explicitly adds that experiment.

Source result:
- The checked `lich99/Stock_CNN/models/baseline.py` at commit
  `415e2acf2a5013afca67e383acd3edc61fced840` is I20-specific.
- It uses input reshape `(batch, 1, 64, 60)` and `Linear(46080, 2)`.
- Therefore `I5` and `I60` require separate paper-targeted variants.

Detailed plan:
- [Stage 2 BTC baseline CNN adaptation plan](../docs/stage2_baseline_cnn_adaptation_plan.md)

Architecture table:
- `stage2_btc_extension/reports/tables/stage2_baseline_cnn_architecture_plan.csv`

## 한국어

상태: 완료.

이번 체크리스트에서는 Stage 1 Stock_CNN식 baseline을 BTC image에 어떻게 이식할지
고정했습니다.

핵심 결정:
- 모든 BTC image를 I20 전용 Stage 1 model에 억지로 넣지 않습니다.
- image window별 model variant를 둡니다:
  - `I5` -> `stock_cnn_i5`
  - `I20` -> `stock_cnn_i20`
  - `I60` -> `stock_cnn_i60`
- CNN은 return horizon이 아니라 image window로 선택합니다.
- 네 가지 image spec은 모두 1-channel grayscale input입니다. MA와 volume은 extra
  channel을 만들지 않습니다.
- baseline training default는 Stage 1을 유지합니다:
  cross-entropy, Adam, learning rate `1e-5`, batch size `128`, dropout `0.5`,
  Xavier initialization, validation-loss early stopping patience `2`.
- 기본 BTC baseline은 from scratch로 학습합니다. stock checkpoint transfer는 이후
  checklist에서 별도 실험으로 추가하지 않는 한 사용하지 않습니다.

Source 결과:
- 확인한 `lich99/Stock_CNN/models/baseline.py` commit
  `415e2acf2a5013afca67e383acd3edc61fced840`는 I20 전용입니다.
- input을 `(batch, 1, 64, 60)`으로 reshape하고 `Linear(46080, 2)`를 사용합니다.
- 따라서 `I5`와 `I60`은 논문 target에 맞춘 별도 variant가 필요합니다.

상세 계획:
- [Stage 2 BTC baseline CNN adaptation plan](../docs/stage2_baseline_cnn_adaptation_plan.md)

Architecture table:
- `stage2_btc_extension/reports/tables/stage2_baseline_cnn_architecture_plan.csv`
