# 2-6 BTC Evaluation and Trading-Metric Plan

## English

Status: complete.

This checklist item fixes how Stage 2 BTC prediction outputs, classification
metrics, calibration diagnostics, and single-asset trading metrics will be
reported.

Key decisions:
- BTC does not use stock cross-sectional H-L decile portfolios.
- BTC reports classification metrics and time-series trading strategies.
- Prediction CSV keeps date, image/window/spec metadata, future return, label,
  logits, probabilities, predicted class, and correctness.
- Classification metrics include accuracy, precision, recall, F1, ROC AUC,
  average precision, Brier score, log loss, confusion matrix, majority-class
  baseline, and probability/return correlations.
- Calibration is required as a 10-bin `prob_up` table.
- Trading strategies:
  - long/flat: `prob_up >= 0.5` -> long, otherwise flat.
  - long/short: `prob_up >= 0.5` -> long, otherwise short.
- Trading metrics use overlapping-horizon daily backtests so R-day signals are
  not treated as independent non-overlapping returns.
- Annualization uses `365` daily periods because BTC trades every calendar day.
- Report both gross metrics and configurable transaction-cost-adjusted metrics.
  Initial planning value: `10 bps` per unit turnover.

Required report sentence:

```text
The original paper is a cross-sectional stock prediction study, so it can form
high-minus-low decile portfolios. BTC is a single asset, so the same H-L decile
structure cannot be directly applied. Therefore, BTC experiments report both
classification metrics and time-series trading strategies.
```

Detailed plan:
- [Stage 2 BTC evaluation/trading metric plan](../docs/stage2_evaluation_trading_metric_plan.md)

Metric schema:
- `stage2_btc_extension/reports/tables/stage2_metric_schema.csv`

## 한국어

상태: 완료.

이번 체크리스트에서는 Stage 2 BTC prediction output, classification metric,
calibration diagnostic, single-asset trading metric을 어떻게 보고할지 고정했습니다.

핵심 결정:
- BTC에서는 stock cross-sectional H-L decile portfolio를 사용하지 않습니다.
- BTC는 classification metric과 time-series trading strategy를 함께 보고합니다.
- Prediction CSV에는 date, image/window/spec metadata, future return, label,
  logits, probability, predicted class, correctness를 저장합니다.
- Classification metric은 accuracy, precision, recall, F1, ROC AUC, average
  precision, Brier score, log loss, confusion matrix, majority-class baseline,
  probability/return correlation을 포함합니다.
- Calibration은 `prob_up` 10-bin table을 필수로 만듭니다.
- Trading strategy:
  - long/flat: `prob_up >= 0.5`이면 long, 아니면 flat.
  - long/short: `prob_up >= 0.5`이면 long, 아니면 short.
- Trading metric은 overlapping-horizon daily backtest로 계산합니다. 그래서 R-day
  signal을 서로 독립적인 비중복 return처럼 잘못 취급하지 않습니다.
- BTC는 매일 거래되므로 annualization은 `365` daily periods를 사용합니다.
- gross metric과 configurable transaction-cost-adjusted metric을 모두 보고합니다.
  초기 계획값은 unit turnover당 `10 bps`입니다.

보고서 필수 문장:

```text
원 논문은 cross-sectional stock prediction이므로 H-L decile spread를 구성할 수 있지만,
BTC는 단일 자산이므로 동일한 H-L 구조를 직접 적용하기 어렵다. 따라서 BTC 실험에서는
classification metric과 time-series trading strategy를 함께 사용한다.
```

상세 계획:
- [Stage 2 BTC evaluation/trading metric plan](../docs/stage2_evaluation_trading_metric_plan.md)

Metric schema:
- `stage2_btc_extension/reports/tables/stage2_metric_schema.csv`
