# Stage 1 Evaluation and Prediction-output Detail Plan

## English

Status:
- Stage 1-7 completed as a detail plan.
- No evaluation or prediction-output code has been implemented yet.

## Purpose

Define how Stage 1 model outputs are converted into probabilities,
predictions, metrics, saved CSV files, and paper-style stock ranking outputs.

## Sources Checked

Process source:
- `../PLAN.md`
- `docs/stage1_checklist.md`
- `docs/stage1_pipeline.md`
- `docs/source_map.md`
- `docs/label_construction_plan.md`
- `docs/training_loop_plan.md`
- `docs/kaggle_runner_plan.md`

Paper/source references:
- `../자료조사/Re-image 요약.md`
- `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
- `../stage0_data_check/docs/source_reference_check.md`

## Scope

Stage 1 evaluation targets:
- `stage1_i20_r5`
- `stage1_i20_r20`
- `stage1_i20_r60`

Each experiment evaluates:
- individual seed checkpoints
- full paper-style 5-run averaged predictions when available

Evaluation splits:
- primary final evaluation: test years `2001-2019`
- train/validation metrics may be saved for diagnostics, but final reported
  reproduction metrics must use the test set.

## Probability and Prediction Rule

Paper source:
- Re-image summary maps this to the CNN/training section around pp.20-22:
  the model output is interpreted as a softmax Up probability and the decision
  threshold is 50%.
- The exact tie behavior at probability `0.5` is not reported by the paper.

Model forward output:
- logits with shape `(batch_size, 2)`
- class `0`: non-positive future return
- class `1`: positive future return / Up

Probability:

```text
prob_down, prob_up = softmax(logits, dim=1)
```

Single-seed prediction:

```text
pred_class = 1 if prob_up >= 0.5 else 0
```

Tie rule:
- Implementation choice, not a paper-reported rule.
- Exact `0.5` is assigned to class `1` only to keep the implementation
  deterministic.
- Exact ties should be practically rare.

5-run averaged prediction:

```text
mean_prob_up = mean(prob_up_seed_42, ..., prob_up_seed_46)
pred_class = 1 if mean_prob_up >= 0.5 else 0
```

Important:
- Do not average logits for the final paper-style prediction.
- Average probabilities after softmax.

## Prediction Output Files

Per-seed prediction CSV:

```text
outputs/predictions/{experiment_name}/seed_{run_seed}/test_predictions.csv
```

Example:

```text
outputs/predictions/stage1_i20_r20/seed_42/test_predictions.csv
```

5-run averaged prediction CSV:

```text
outputs/predictions/{experiment_name}/averaged/test_predictions.csv
```

Example:

```text
outputs/predictions/stage1_i20_r20/averaged/test_predictions.csv
```

## Required Prediction Columns

Per-seed prediction CSV columns:

| Column | Meaning |
| --- | --- |
| `Date` | sample date |
| `StockID` | stock identifier |
| `year` | source shard year |
| `local_row` | row index inside the year shard |
| `split` | `test`, or diagnostic `train`/`val` if exported |
| `experiment_name` | e.g. `stage1_i20_r20` |
| `image_window` | fixed `20` for current Stage 1 |
| `target_horizon` | `5`, `20`, or `60` |
| `target_return_name` | `Ret_5d`, `Ret_20d`, or `Ret_60d` |
| `target_return` | realized future holding-period return |
| `label` | `1` if target return `> 0`, otherwise `0` |
| `MarketCap` | market capitalization metadata |
| `EWMA_vol` | volatility metadata, if present |
| `Ret_5d` | preserved metadata |
| `Ret_20d` | preserved metadata |
| `Ret_60d` | preserved metadata |
| `Ret_month` | preserved metadata |
| `run_seed` | seed for this prediction file |
| `checkpoint_path` | checkpoint used for prediction |
| `logit_down` | model logit for class `0` |
| `logit_up` | model logit for class `1` |
| `prob_down` | softmax probability for class `0` |
| `prob_up` | softmax probability for class `1` |
| `pred_class` | predicted class from `prob_up >= 0.5` |
| `correct` | `1` if `pred_class == label`, otherwise `0` |

5-run averaged prediction CSV columns:
- all non-logit metadata columns above
- `prob_up_seed_42`
- `prob_up_seed_43`
- `prob_up_seed_44`
- `prob_up_seed_45`
- `prob_up_seed_46`
- `mean_prob_up`
- `std_prob_up`
- `pred_class`
- `correct`

Logits are optional in the averaged file because averaged prediction is based
on probabilities, not averaged logits.

## Classification Metrics

Per-seed metrics file:

```text
outputs/metrics/{experiment_name}/seed_{run_seed}/test_metrics.json
```

5-run averaged metrics file:

```text
outputs/metrics/{experiment_name}/averaged/test_metrics.json
```

Core metrics:
- `num_samples`
- `positive_rate`
- `negative_rate`
- `predicted_positive_rate`
- `accuracy`
- `majority_class_accuracy`
- `accuracy_minus_majority_class_accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `average_precision`
- `brier_score`
- `log_loss`
- confusion matrix counts:
  - `tn`
  - `fp`
  - `fn`
  - `tp`

Rules:
- If a metric is undefined because only one class is present, store `null` and
  record the reason.
- `majority_class_accuracy` is required because positive rates are not exactly
  50%.
- Report `accuracy_minus_majority_class_accuracy` to avoid confusing class
  imbalance with predictive improvement.

## Return Correlation Diagnostics

The Re-image paper reports classification and stock-ranking style performance.
For Stage 1 diagnostics, save:
- Pearson correlation between `prob_up` and `target_return`
- Spearman correlation between `prob_up` and `target_return`
- date-wise cross-sectional Pearson correlation, averaged across dates
- date-wise cross-sectional Spearman correlation, averaged across dates

Files:

```text
outputs/metrics/{experiment_name}/seed_{run_seed}/correlation_metrics.json
outputs/metrics/{experiment_name}/averaged/correlation_metrics.json
```

Date-wise correlations should skip dates with too few valid stocks.

## Paper-style Decile / H-L Outputs

Stage 1 is stock cross-section reproduction, so paper-style ranking outputs are
part of the evaluation plan.

Ranking signal:
- single seed: `prob_up`
- averaged run: `mean_prob_up`

For each `Date`:
- rank stocks by the ranking signal
- assign deciles
- bottom decile: predicted weakest stocks
- top decile: predicted strongest stocks

Output files:

```text
outputs/predictions/{experiment_name}/seed_{run_seed}/test_decile_assignments.csv
outputs/predictions/{experiment_name}/averaged/test_decile_assignments.csv
outputs/metrics/{experiment_name}/seed_{run_seed}/portfolio_metrics.json
outputs/metrics/{experiment_name}/averaged/portfolio_metrics.json
```

Minimum decile assignment columns:
- `Date`
- `StockID`
- `target_return`
- `MarketCap`
- ranking signal
- `decile`
- `is_bottom_decile`
- `is_top_decile`

Portfolio metrics to plan:
- equal-weight top decile return
- equal-weight bottom decile return
- equal-weight high-minus-low return
- value-weight top decile return using `MarketCap`
- value-weight bottom decile return using `MarketCap`
- value-weight high-minus-low return
- annualized return
- annualized volatility
- Sharpe ratio
- turnover, if implementable from consecutive date-level holdings

Important:
- This is stock cross-sectional evaluation.
- BTC evaluation later cannot copy this H-L decile setup directly because BTC is
  a single asset.
- Exact portfolio annualization conventions should be rechecked before
  implementation and report writing.

## Summary Tables

Planned report tables:

```text
reports/tables/stage1_classification_metrics.csv
reports/tables/stage1_majority_baseline_comparison.csv
reports/tables/stage1_correlation_metrics.csv
reports/tables/stage1_portfolio_metrics.csv
```

Minimum classification table columns:
- `experiment_name`
- `image_window`
- `target_horizon`
- `run_type`: `seed` or `averaged`
- `run_seed`, nullable for averaged output
- `num_samples`
- `positive_rate`
- `predicted_positive_rate`
- `accuracy`
- `majority_class_accuracy`
- `accuracy_minus_majority_class_accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `brier_score`
- `log_loss`

## Config Fields

```yaml
evaluation:
  threshold: 0.5
  tie_break_class: 1
  probability_source: softmax_logits
  average_probabilities_across_seeds: true
  metrics:
    classification:
      - accuracy
      - majority_class_accuracy
      - precision
      - recall
      - f1
      - roc_auc
      - average_precision
      - brier_score
      - log_loss
      - confusion_matrix
    correlation:
      - pearson_prob_return
      - spearman_prob_return
      - datewise_pearson_prob_return
      - datewise_spearman_prob_return
    portfolio:
      enabled: true
      ranking_signal_single_seed: prob_up
      ranking_signal_averaged: mean_prob_up
      num_deciles: 10
      weights:
        - equal_weight
        - value_weight_market_cap
```

## Deferred Items

Deferred to implementation gate `1-I7`:
- actual metric functions
- prediction CSV writer
- averaged prediction merger
- decile assignment and portfolio metric code

Deferred to `1-8`:
- Grad-CAM sample selection from prediction files.
- Whether Grad-CAM uses the averaged prediction decision or a representative
  seed checkpoint.

## 한국어

상태:
- 1-7을 evaluation과 prediction-output 세부계획으로 완료했습니다.
- evaluation/prediction-output code는 아직 구현하지 않았습니다.

## 목적

1단계 모델 출력 logits를 probability, prediction, metric, CSV 파일,
paper-style stock ranking output으로 어떻게 바꿀지 정의합니다.

## 확인한 근거

진행 기준:
- `../PLAN.md`
- `docs/stage1_checklist.md`
- `docs/stage1_pipeline.md`
- `docs/source_map.md`
- `docs/label_construction_plan.md`
- `docs/training_loop_plan.md`
- `docs/kaggle_runner_plan.md`

논문/근거:
- `../자료조사/Re-image 요약.md`
- `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
- `../stage0_data_check/docs/source_reference_check.md`

## 범위

1단계 evaluation 대상:
- `stage1_i20_r5`
- `stage1_i20_r20`
- `stage1_i20_r60`

각 experiment는:
- individual seed checkpoint
- 가능하면 full paper-style 5-run averaged prediction

평가 split:
- 최종 보고용 primary evaluation은 test years `2001-2019`.
- train/validation metrics는 diagnostic으로 저장할 수 있지만, 재현 결과로 보고할
  값은 test set 기준입니다.

## Probability와 Prediction Rule

논문 근거:
- Re-image 요약 기준 CNN/training section pp.20-22에 해당합니다.
  model output은 softmax Up probability로 해석하고 decision threshold는 50%입니다.
- probability가 정확히 `0.5`일 때 어떻게 처리하는지는 논문에 보고되어 있지 않습니다.

모델 `forward` output:
- logits shape `(batch_size, 2)`
- class `0`: non-positive future return
- class `1`: positive future return / Up

Probability:

```text
prob_down, prob_up = softmax(logits, dim=1)
```

Single-seed prediction:

```text
pred_class = 1 if prob_up >= 0.5 else 0
```

Tie rule:
- 논문에 보고된 규칙이 아니라 implementation choice입니다.
- 정확히 `0.5`면 class `1`로 두는 것은 구현을 deterministic하게 만들기 위한 선택입니다.
- 정확한 tie는 실제로 거의 없을 가능성이 큽니다.

5-run averaged prediction:

```text
mean_prob_up = mean(prob_up_seed_42, ..., prob_up_seed_46)
pred_class = 1 if mean_prob_up >= 0.5 else 0
```

중요:
- 최종 paper-style prediction에서는 logits를 평균하지 않습니다.
- softmax 이후 probability를 평균합니다.

## Prediction Output Files

Seed별 prediction CSV:

```text
outputs/predictions/{experiment_name}/seed_{run_seed}/test_predictions.csv
```

예시:

```text
outputs/predictions/stage1_i20_r20/seed_42/test_predictions.csv
```

5-run averaged prediction CSV:

```text
outputs/predictions/{experiment_name}/averaged/test_predictions.csv
```

예시:

```text
outputs/predictions/stage1_i20_r20/averaged/test_predictions.csv
```

## 필수 Prediction Columns

Seed별 prediction CSV columns:

| Column | 의미 |
| --- | --- |
| `Date` | sample date |
| `StockID` | stock identifier |
| `year` | source shard year |
| `local_row` | year shard 내부 row index |
| `split` | `test`, 또는 diagnostic export일 경우 `train`/`val` |
| `experiment_name` | 예: `stage1_i20_r20` |
| `image_window` | 현재 1단계에서는 fixed `20` |
| `target_horizon` | `5`, `20`, 또는 `60` |
| `target_return_name` | `Ret_5d`, `Ret_20d`, 또는 `Ret_60d` |
| `target_return` | realized future holding-period return |
| `label` | target return `> 0`이면 `1`, 아니면 `0` |
| `MarketCap` | market capitalization metadata |
| `EWMA_vol` | volatility metadata, 있으면 저장 |
| `Ret_5d` | 보존 metadata |
| `Ret_20d` | 보존 metadata |
| `Ret_60d` | 보존 metadata |
| `Ret_month` | 보존 metadata |
| `run_seed` | 해당 prediction file의 seed |
| `checkpoint_path` | prediction에 사용한 checkpoint |
| `logit_down` | class `0` logit |
| `logit_up` | class `1` logit |
| `prob_down` | class `0` softmax probability |
| `prob_up` | class `1` softmax probability |
| `pred_class` | `prob_up >= 0.5` 기준 prediction |
| `correct` | `pred_class == label`이면 `1`, 아니면 `0` |

5-run averaged prediction CSV columns:
- 위의 non-logit metadata columns
- `prob_up_seed_42`
- `prob_up_seed_43`
- `prob_up_seed_44`
- `prob_up_seed_45`
- `prob_up_seed_46`
- `mean_prob_up`
- `std_prob_up`
- `pred_class`
- `correct`

Averaged file에서는 averaged prediction이 probability 기준이므로 logits는 optional입니다.

## Classification Metrics

Seed별 metrics file:

```text
outputs/metrics/{experiment_name}/seed_{run_seed}/test_metrics.json
```

5-run averaged metrics file:

```text
outputs/metrics/{experiment_name}/averaged/test_metrics.json
```

Core metrics:
- `num_samples`
- `positive_rate`
- `negative_rate`
- `predicted_positive_rate`
- `accuracy`
- `majority_class_accuracy`
- `accuracy_minus_majority_class_accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `average_precision`
- `brier_score`
- `log_loss`
- confusion matrix counts:
  - `tn`
  - `fp`
  - `fn`
  - `tp`

규칙:
- class가 하나만 있어서 metric이 정의되지 않으면 `null`로 저장하고 이유를 기록합니다.
- positive rate가 정확히 50%가 아니므로 `majority_class_accuracy`는 필수입니다.
- class imbalance와 predictive improvement를 혼동하지 않도록
  `accuracy_minus_majority_class_accuracy`를 보고합니다.

## Return Correlation Diagnostics

Re-image 논문은 classification과 stock-ranking style performance를 보고합니다.
1단계 diagnostic으로 아래를 저장합니다.
- `prob_up`과 `target_return`의 Pearson correlation
- `prob_up`과 `target_return`의 Spearman correlation
- date-wise cross-sectional Pearson correlation 평균
- date-wise cross-sectional Spearman correlation 평균

파일:

```text
outputs/metrics/{experiment_name}/seed_{run_seed}/correlation_metrics.json
outputs/metrics/{experiment_name}/averaged/correlation_metrics.json
```

date-wise correlation은 valid stock 수가 너무 적은 날짜는 skip합니다.

## Paper-style Decile / H-L Outputs

1단계는 stock cross-section reproduction이므로 paper-style ranking output도
evaluation plan에 포함합니다.

Ranking signal:
- single seed: `prob_up`
- averaged run: `mean_prob_up`

각 `Date`마다:
- ranking signal 기준으로 stock을 정렬
- decile 할당
- bottom decile: 가장 약하게 예측된 종목
- top decile: 가장 강하게 예측된 종목

Output files:

```text
outputs/predictions/{experiment_name}/seed_{run_seed}/test_decile_assignments.csv
outputs/predictions/{experiment_name}/averaged/test_decile_assignments.csv
outputs/metrics/{experiment_name}/seed_{run_seed}/portfolio_metrics.json
outputs/metrics/{experiment_name}/averaged/portfolio_metrics.json
```

최소 decile assignment columns:
- `Date`
- `StockID`
- `target_return`
- `MarketCap`
- ranking signal
- `decile`
- `is_bottom_decile`
- `is_top_decile`

계획할 portfolio metrics:
- equal-weight top decile return
- equal-weight bottom decile return
- equal-weight high-minus-low return
- `MarketCap` 기준 value-weight top decile return
- `MarketCap` 기준 value-weight bottom decile return
- value-weight high-minus-low return
- annualized return
- annualized volatility
- Sharpe ratio
- consecutive date-level holdings로 구현 가능하면 turnover

중요:
- 이것은 stock cross-sectional evaluation입니다.
- BTC는 단일 자산이므로 이후 BTC 단계에서는 이 H-L decile 구조를 그대로
  적용하면 안 됩니다.
- portfolio annualization convention은 구현과 보고서 작성 전에 다시 확인합니다.

## Summary Tables

예정 report tables:

```text
reports/tables/stage1_classification_metrics.csv
reports/tables/stage1_majority_baseline_comparison.csv
reports/tables/stage1_correlation_metrics.csv
reports/tables/stage1_portfolio_metrics.csv
```

Minimum classification table columns:
- `experiment_name`
- `image_window`
- `target_horizon`
- `run_type`: `seed` 또는 `averaged`
- `run_seed`, averaged output에서는 nullable
- `num_samples`
- `positive_rate`
- `predicted_positive_rate`
- `accuracy`
- `majority_class_accuracy`
- `accuracy_minus_majority_class_accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `brier_score`
- `log_loss`

## Config Fields

```yaml
evaluation:
  threshold: 0.5
  tie_break_class: 1
  probability_source: softmax_logits
  average_probabilities_across_seeds: true
  metrics:
    classification:
      - accuracy
      - majority_class_accuracy
      - precision
      - recall
      - f1
      - roc_auc
      - average_precision
      - brier_score
      - log_loss
      - confusion_matrix
    correlation:
      - pearson_prob_return
      - spearman_prob_return
      - datewise_pearson_prob_return
      - datewise_spearman_prob_return
    portfolio:
      enabled: true
      ranking_signal_single_seed: prob_up
      ranking_signal_averaged: mean_prob_up
      num_deciles: 10
      weights:
        - equal_weight
        - value_weight_market_cap
```

## 이후 단계로 넘길 항목

Implementation gate `1-I7`로 넘김:
- 실제 metric functions
- prediction CSV writer
- averaged prediction merger
- decile assignment와 portfolio metric code

`1-8`로 넘김:
- prediction file에서 Grad-CAM sample selection.
- Grad-CAM이 averaged prediction decision을 기준으로 할지, 대표 seed checkpoint를
  기준으로 할지.
