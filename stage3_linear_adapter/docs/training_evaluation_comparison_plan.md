# 3-3 Training and Evaluation Comparison Plan

## English

Purpose:
- Define how BTC baseline and BTC Linear results will be compared.
- Keep Stage 2 training/evaluation settings fixed unless a run is explicitly
  marked as a speed diagnostic.

Primary single-seed comparison:
- Run seed: `42`
- Grid: `36` runs
  - Image windows: `I5`, `I20`, `I60`
  - Return horizons: `R5`, `R20`, `R60`
  - Image specs: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- Model variants:
  - Stage 2 baseline CNN.
  - Stage 3 CNN + Linear adapter/head, `adapter_dim=128`, `bias=False`.

Training settings inherited from Stage 2:

| Setting | Value |
|:---|:---|
| Batch size | `128` |
| Max epochs | `100` |
| Optimizer | Adam |
| Learning rate | `1e-5` |
| Loss | CrossEntropyLoss |
| Early stopping | `val_loss`, mode `min`, patience `2` |
| Mixed precision | `false` by default |
| DataParallel | `false` by default |

Evaluation outputs:
- `test_predictions.csv`
- `test_metrics.json`
- `test_trading_metrics.json`
- run manifest and train metadata
- comparison table joining Stage 2 baseline and Stage 3 Linear by:
  `image_window`, `return_horizon`, `image_spec`, `run_seed`

Comparison columns:
- baseline accuracy, Linear accuracy, `delta_accuracy`
- baseline ROC-AUC, Linear ROC-AUC, `delta_roc_auc`
- baseline accuracy-minus-majority, Linear accuracy-minus-majority
- baseline long/flat Sharpe net, Linear long/flat Sharpe net
- baseline long/short Sharpe net, Linear long/short Sharpe net
- parameter count and adapter dimension

Fast order:
1. Smoke run on `I60/R20/ohlc_ma_vb`, seed `42`, tiny row caps.
2. Full single configuration `I60/R20/ohlc_ma_vb`, seed `42`.
3. Full 36-run single-seed grid if the first full configuration works.
4. Five-seed rerun later for leading configurations.

## 한국어

목적:
- BTC baseline과 BTC Linear 결과를 어떻게 비교할지 정의합니다.
- speed diagnostic이라고 명시하지 않는 한 Stage 2 training/evaluation 설정을
  그대로 유지합니다.

Primary single-seed comparison:
- Run seed: `42`
- Grid: `36` runs
  - Image windows: `I5`, `I20`, `I60`
  - Return horizons: `R5`, `R20`, `R60`
  - Image specs: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- Model variants:
  - Stage 2 baseline CNN.
  - Stage 3 CNN + Linear adapter/head, `adapter_dim=128`, `bias=False`.

Stage 2에서 가져오는 training setting:

| Setting | Value |
|:---|:---|
| Batch size | `128` |
| Max epochs | `100` |
| Optimizer | Adam |
| Learning rate | `1e-5` |
| Loss | CrossEntropyLoss |
| Early stopping | `val_loss`, mode `min`, patience `2` |
| Mixed precision | 기본 `false` |
| DataParallel | 기본 `false` |

Evaluation output:
- `test_predictions.csv`
- `test_metrics.json`
- `test_trading_metrics.json`
- run manifest와 train metadata
- Stage 2 baseline과 Stage 3 Linear를 아래 key로 join한 comparison table:
  `image_window`, `return_horizon`, `image_spec`, `run_seed`

비교 column:
- baseline accuracy, Linear accuracy, `delta_accuracy`
- baseline ROC-AUC, Linear ROC-AUC, `delta_roc_auc`
- baseline accuracy-minus-majority, Linear accuracy-minus-majority
- baseline long/flat Sharpe net, Linear long/flat Sharpe net
- baseline long/short Sharpe net, Linear long/short Sharpe net
- parameter count와 adapter dimension

빠른 실행 순서:
1. `I60/R20/ohlc_ma_vb`, seed `42`, tiny row cap으로 smoke run.
2. `I60/R20/ohlc_ma_vb`, seed `42` full single configuration.
3. 첫 full configuration이 정상 작동하면 36-run single-seed grid.
4. leading configuration에 대해서 five-seed rerun은 나중에 실행.
