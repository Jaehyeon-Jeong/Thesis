# Stage 2 BTC Evaluation and Trading Metric Plan

## English

Status: planning complete for checklist 2-6. Implementation happens later in
`2-I6` and `2-I7`.

Purpose:
- Define how BTC baseline predictions are evaluated.
- Keep classification evaluation close to Stage 1 where possible.
- Replace the stock paper's cross-sectional H-L decile portfolio with
  single-asset BTC trading strategies.

Source basis:
- Root plan `PLAN.md`, Stage 2 evaluation section.
- Re-image summary: `자료조사/Re-image 요약.md`
  - lines 5, 7, 73-77: classification accuracy and H-L portfolio evaluation.
  - line 77: short-horizon turnover can be high and should not be over-read as
    a live trading claim.
- Stage 1 evaluation implementation:
  - `stage1_reimage_reproduction/src/stage1_reimage/evaluation/prediction.py`
- Stage 2 source-map constraint:
  - BTC is a single asset, so stock cross-sectional H-L decile portfolios are
    not directly applicable.

Required reporting sentence:

```text
The original paper is a cross-sectional stock prediction study, so it can form
high-minus-low decile portfolios. BTC is a single asset, so the same H-L decile
structure cannot be directly applied. Therefore, BTC experiments report both
classification metrics and time-series trading strategies.
```

## Prediction Output Schema

Each Stage 2 prediction row should preserve:
- `sample_id`
- `image_window`
- `image_spec`
- `return_horizon`
- `split`
- `image_start_date`
- `image_end_date`
- `label_end_date`
- `entry_close`
- `exit_close`
- `future_return`
- `label`
- `run_seed`
- `checkpoint_path`
- `logit_down`
- `logit_up`
- `prob_down`
- `prob_up`
- `pred_class`
- `correct`

Class convention:
- class `0`: Down or non-positive future return.
- class `1`: Up / positive future return.
- `prob_up` is computed with `softmax(logits, dim=1)[:, 1]`.
- `pred_class = 1` if `prob_up >= 0.5`, otherwise `0`.
- The exact tie rule is an implementation convention, not a separately reported
  paper detail. Exact ties should be rare.

Multiple seeds:
- Save per-seed prediction CSVs.
- Save averaged prediction CSVs for full paper-style runs.
- Averaging rule: average `prob_up` across seeds, then recompute
  `pred_class`, `correct`, and metrics from the averaged probability.
- Do not average hard class labels.

## Classification Metrics

Core metrics:
- accuracy
- majority-class accuracy
- accuracy minus majority-class accuracy
- precision
- recall
- F1
- ROC AUC
- average precision
- Brier score
- binary log loss
- confusion matrix counts: TP, TN, FP, FN
- positive rate
- predicted-positive rate

Probability/return diagnostics:
- Pearson correlation between `prob_up` and `future_return`.
- Spearman correlation between `prob_up` and `future_return`.
- Mean future return by probability quantile.

Calibration outputs:
- 10-bin calibration table by `prob_up`.
- Each bin should include number of samples, mean predicted probability, observed
  positive rate, and mean future return.
- Calibration curve figure can be generated later, but the table is required.

Output files:
- `outputs/stage2/{experiment}/predictions_seed_{seed}.csv`
- `outputs/stage2/{experiment}/predictions_averaged.csv`
- `outputs/stage2/{experiment}/classification_metrics.json`
- `reports/tables/stage2_classification_metrics.csv`
- `reports/tables/stage2_calibration_bins.csv`

## BTC Trading Strategy Metrics

The trading evaluation uses BTC time-series strategies, not stock decile
portfolios.

Signal timing:
- Image end date is `t`.
- The image uses OHLCV information through date `t`.
- Label-aligned future return is `Close_{t+R} / Close_t - 1`.
- The primary Stage 2 trading report uses this label-aligned timing for
  consistency with the classification target.
- A stricter next-open tradability check can be added later as a robustness
  experiment, but it is not the default Stage 2 baseline.

Position rules:

| Strategy | Rule |
| --- | --- |
| long_flat | `position_t = 1` if `prob_up >= 0.5`, else `0` |
| long_short | `position_t = 1` if `prob_up >= 0.5`, else `-1` |

Optional robustness rule:
- `confidence_band`: long if `prob_up >= 0.55`, short if `prob_up <= 0.45`,
  flat otherwise.
- This is not the default baseline. It should be reported separately if used.

Overlapping-horizon daily backtest:
- A new signal is generated at every available signal date.
- A signal with horizon `R` is held for `R` daily close-to-close returns.
- On a daily return date `d`, the strategy position is the average of all active
  signals whose holding windows include `d`.
- This avoids treating overlapping R-day trades as independent non-overlapping
  returns.

Daily return definition:

```text
btc_return_d = Close_d / Close_{d-1} - 1
strategy_return_d = active_position_d * btc_return_d
```

Transaction cost adjustment:

```text
turnover_d = abs(active_position_d - active_position_{d-1})
cost_d = turnover_d * transaction_cost_bps / 10000
net_strategy_return_d = strategy_return_d - cost_d
```

Default transaction-cost reporting:
- Report gross metrics with `transaction_cost_bps = 0`.
- Report cost-adjusted metrics with configurable bps.
- Initial planning value: `10 bps` per unit turnover.
- This cost value is an implementation assumption, not a value reported by the
  Re-image paper.

Annualization:
- BTC trades every calendar day in the audited daily file.
- Use `365` periods per year for daily strategy returns.
- Annualized return: geometric from the daily equity curve.
- Annualized volatility: daily return standard deviation times `sqrt(365)`.
- Sharpe ratio: annualized mean excess return divided by annualized volatility,
  with risk-free rate set to `0` in the baseline report.

Trading metrics:
- cumulative return
- annualized return
- annualized volatility
- Sharpe ratio
- max drawdown
- average daily turnover
- total turnover
- average exposure
- percent time long
- percent time short
- percent time flat
- number of signal dates
- number of active daily return observations

Output files:
- `outputs/stage2/{experiment}/trading_daily_returns.csv`
- `outputs/stage2/{experiment}/trading_metrics.json`
- `reports/tables/stage2_trading_metrics.csv`

## Result Table Keys

Minimum result table columns:

```text
image_window
image_spec
return_horizon
model_name
run_seed_or_averaged
split
accuracy
roc_auc
f1
brier_score
pearson_prob_return
spearman_prob_return
strategy_name
transaction_cost_bps
annualized_return
annualized_volatility
sharpe_ratio
max_drawdown
average_turnover
average_exposure
```

CSV artifact:
- `stage2_btc_extension/reports/tables/stage2_metric_schema.csv`

## 한국어

상태: checklist 2-6 계획 완료. 실제 구현은 이후 `2-I6`, `2-I7`에서 합니다.

목적:
- BTC baseline prediction을 어떻게 평가할지 정의합니다.
- classification 평가는 가능한 한 Stage 1 방식을 유지합니다.
- 원논문의 stock cross-sectional H-L decile portfolio는 BTC 단일 자산에는 직접
  적용할 수 없으므로 BTC time-series trading strategy로 바꿉니다.

근거:
- Root plan `PLAN.md`, Stage 2 evaluation section.
- Re-image 요약: `자료조사/Re-image 요약.md`
  - lines 5, 7, 73-77: classification accuracy와 H-L portfolio evaluation.
  - line 77: short-horizon turnover가 높으므로 live trading claim으로 과도하게
    해석하면 안 된다는 주의.
- Stage 1 evaluation implementation:
  - `stage1_reimage_reproduction/src/stage1_reimage/evaluation/prediction.py`
- Stage 2 source-map constraint:
  - BTC는 단일 자산이므로 stock cross-sectional H-L decile portfolio를 직접 사용할
    수 없습니다.

보고서에 반드시 넣을 문장:

```text
원 논문은 cross-sectional stock prediction이므로 H-L decile spread를 구성할 수 있지만,
BTC는 단일 자산이므로 동일한 H-L 구조를 직접 적용하기 어렵다. 따라서 BTC 실험에서는
classification metric과 time-series trading strategy를 함께 사용한다.
```

## Prediction Output Schema

Stage 2 prediction row마다 다음을 보존합니다:
- `sample_id`
- `image_window`
- `image_spec`
- `return_horizon`
- `split`
- `image_start_date`
- `image_end_date`
- `label_end_date`
- `entry_close`
- `exit_close`
- `future_return`
- `label`
- `run_seed`
- `checkpoint_path`
- `logit_down`
- `logit_up`
- `prob_down`
- `prob_up`
- `pred_class`
- `correct`

Class convention:
- class `0`: Down 또는 non-positive future return.
- class `1`: Up / positive future return.
- `prob_up`은 `softmax(logits, dim=1)[:, 1]`로 계산합니다.
- `prob_up >= 0.5`이면 `pred_class = 1`, 아니면 `0`입니다.
- 정확한 tie rule은 논문에 별도로 보고된 detail이 아니라 implementation
  convention입니다. 실제 exact tie는 드뭅니다.

Multiple seeds:
- seed별 prediction CSV를 저장합니다.
- full paper-style run에서는 averaged prediction CSV를 저장합니다.
- averaging rule: seed별 `prob_up`을 평균낸 뒤 `pred_class`, `correct`, metric을 다시
  계산합니다.
- hard class label 자체를 평균내지 않습니다.

## Classification Metrics

Core metrics:
- accuracy
- majority-class accuracy
- accuracy minus majority-class accuracy
- precision
- recall
- F1
- ROC AUC
- average precision
- Brier score
- binary log loss
- confusion matrix counts: TP, TN, FP, FN
- positive rate
- predicted-positive rate

Probability/return diagnostics:
- `prob_up`과 `future_return`의 Pearson correlation.
- `prob_up`과 `future_return`의 Spearman correlation.
- probability quantile별 mean future return.

Calibration outputs:
- `prob_up` 기준 10-bin calibration table.
- 각 bin에는 sample 수, mean predicted probability, observed positive rate,
  mean future return을 포함합니다.
- calibration curve figure는 나중에 만들 수 있지만, table은 필수입니다.

Output files:
- `outputs/stage2/{experiment}/predictions_seed_{seed}.csv`
- `outputs/stage2/{experiment}/predictions_averaged.csv`
- `outputs/stage2/{experiment}/classification_metrics.json`
- `reports/tables/stage2_classification_metrics.csv`
- `reports/tables/stage2_calibration_bins.csv`

## BTC Trading Strategy Metrics

Trading 평가는 stock decile portfolio가 아니라 BTC time-series strategy입니다.

Signal timing:
- image 종료일을 `t`라고 둡니다.
- image는 `t`일까지의 OHLCV 정보를 사용합니다.
- label-aligned future return은 `Close_{t+R} / Close_t - 1`입니다.
- 기본 Stage 2 trading report는 classification target과 맞추기 위해 이
  label-aligned timing을 사용합니다.
- 더 엄격한 next-open tradability check는 나중 robustness experiment로 추가할 수
  있지만 기본 Stage 2 baseline은 아닙니다.

Position rules:

| Strategy | Rule |
| --- | --- |
| long_flat | `prob_up >= 0.5`이면 `position_t = 1`, 아니면 `0` |
| long_short | `prob_up >= 0.5`이면 `position_t = 1`, 아니면 `-1` |

Optional robustness rule:
- `confidence_band`: `prob_up >= 0.55`이면 long, `prob_up <= 0.45`이면 short,
  그 외에는 flat.
- 기본 baseline이 아닙니다. 사용하면 별도 보고합니다.

Overlapping-horizon daily backtest:
- 가능한 모든 signal date에서 새 signal을 만듭니다.
- horizon `R` signal은 `R`개의 daily close-to-close return 동안 유지합니다.
- daily return date `d`의 strategy position은 그 날짜에 활성화되어 있는 모든 signal
  position의 평균입니다.
- 이렇게 해야 겹치는 R-day trade를 서로 독립적인 비중복 return처럼 잘못 취급하지
  않습니다.

Daily return definition:

```text
btc_return_d = Close_d / Close_{d-1} - 1
strategy_return_d = active_position_d * btc_return_d
```

Transaction cost adjustment:

```text
turnover_d = abs(active_position_d - active_position_{d-1})
cost_d = turnover_d * transaction_cost_bps / 10000
net_strategy_return_d = strategy_return_d - cost_d
```

Default transaction-cost reporting:
- `transaction_cost_bps = 0`인 gross metric을 보고합니다.
- configurable bps의 cost-adjusted metric도 보고합니다.
- 초기 계획값은 unit turnover당 `10 bps`입니다.
- 이 cost 값은 Re-image 논문 보고값이 아니라 implementation assumption입니다.

Annualization:
- audit한 daily file에서 BTC는 매일 거래됩니다.
- daily strategy return에는 연 `365`일을 사용합니다.
- annualized return은 daily equity curve에서 geometric 방식으로 계산합니다.
- annualized volatility는 daily return standard deviation에 `sqrt(365)`를 곱합니다.
- Sharpe ratio는 baseline report에서 risk-free rate `0`으로 계산합니다.

Trading metrics:
- cumulative return
- annualized return
- annualized volatility
- Sharpe ratio
- max drawdown
- average daily turnover
- total turnover
- average exposure
- percent time long
- percent time short
- percent time flat
- number of signal dates
- number of active daily return observations

Output files:
- `outputs/stage2/{experiment}/trading_daily_returns.csv`
- `outputs/stage2/{experiment}/trading_metrics.json`
- `reports/tables/stage2_trading_metrics.csv`

## Result Table Keys

최소 result table columns:

```text
image_window
image_spec
return_horizon
model_name
run_seed_or_averaged
split
accuracy
roc_auc
f1
brier_score
pearson_prob_return
spearman_prob_return
strategy_name
transaction_cost_bps
annualized_return
annualized_volatility
sharpe_ratio
max_drawdown
average_turnover
average_exposure
```

CSV artifact:
- `stage2_btc_extension/reports/tables/stage2_metric_schema.csv`
