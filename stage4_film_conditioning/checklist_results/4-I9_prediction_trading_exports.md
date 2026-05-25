# 4-I9 Prediction and Trading Exports

## English

Status: complete

Implemented Stage 4 prediction, classification metric, and trading metric
exports.

Added files:
- `src/stage4_film/evaluation/prediction.py`
- `src/stage4_film/evaluation/__init__.py`
- `scripts/evaluate_stage4_predictions.py`
- `scripts/evaluate_stage4_trading.py`

What changed:
- Stage 2 evaluation utilities assume `model(image)`.
- Stage 4 models require `model(image, context)`.
- The new Stage 4 prediction helper keeps the Stage 2 prediction CSV schema and
  adds:
  - `stage4_experiment_name`
  - `context_method`
  - normalized context columns prefixed as `context_*`
- Classification metrics reuse Stage 2 `compute_classification_metrics`.
- Trading metrics reuse Stage 2 `compute_trading_metrics`.

Output paths:
- Predictions:
  `outputs/stage4/predictions/{stage4_experiment}/seed_{seed}/{split}_predictions.csv`
- Classification metrics:
  `outputs/stage4/metrics/{stage4_experiment}/seed_{seed}/{split}_metrics.json`
- Trading metrics:
  `outputs/stage4/metrics/{stage4_experiment}/seed_{seed}/{split}_trading_metrics.json`

Local validation:
- `python -m py_compile` passed for:
  - `src/stage4_film/evaluation/prediction.py`
  - `scripts/evaluate_stage4_predictions.py`
  - `scripts/evaluate_stage4_trading.py`
- Prediction and trading export passed for the local `concat` smoke checkpoint:
  - experiment `stage4_concat_i60_ohlc_ma_vb_r20_c60`
  - split `test`
  - predictions: 8 rows
  - classification metrics written
  - trading metrics written
- Prediction and trading export passed for the local `film_gamma` smoke
  checkpoint:
  - experiment `stage4_film_gamma_i60_ohlc_ma_vb_r20_c60`
  - split `test`
  - predictions: 4 rows
  - classification metrics written
  - trading metrics written

Current boundary:
- 4-I9 exports prediction/classification/trading outputs.
- 4-I10 will add Grad-CAM plus context/gate/gamma/beta interpretation exports.

## 한국어

상태: 완료

Stage 4 prediction, classification metric, trading metric export를 구현했습니다.

추가한 파일:
- `src/stage4_film/evaluation/prediction.py`
- `src/stage4_film/evaluation/__init__.py`
- `scripts/evaluate_stage4_predictions.py`
- `scripts/evaluate_stage4_trading.py`

변경 내용:
- Stage 2 evaluation helper는 `model(image)`를 전제로 합니다.
- Stage 4 model은 `model(image, context)`가 필요합니다.
- 새 Stage 4 prediction helper는 Stage 2 prediction CSV schema를 유지하면서
  아래 column을 추가합니다.
  - `stage4_experiment_name`
  - `context_method`
  - `context_*` prefix가 붙은 normalized context columns
- Classification metric은 Stage 2 `compute_classification_metrics`를 재사용합니다.
- Trading metric은 Stage 2 `compute_trading_metrics`를 재사용합니다.

저장 위치:
- Predictions:
  `outputs/stage4/predictions/{stage4_experiment}/seed_{seed}/{split}_predictions.csv`
- Classification metrics:
  `outputs/stage4/metrics/{stage4_experiment}/seed_{seed}/{split}_metrics.json`
- Trading metrics:
  `outputs/stage4/metrics/{stage4_experiment}/seed_{seed}/{split}_trading_metrics.json`

Local 검증:
- 아래 파일의 `python -m py_compile` 통과:
  - `src/stage4_film/evaluation/prediction.py`
  - `scripts/evaluate_stage4_predictions.py`
  - `scripts/evaluate_stage4_trading.py`
- Local `concat` smoke checkpoint에서 prediction/trading export 통과:
  - experiment `stage4_concat_i60_ohlc_ma_vb_r20_c60`
  - split `test`
  - predictions 8 rows
  - classification metrics 저장
  - trading metrics 저장
- Local `film_gamma` smoke checkpoint에서도 prediction/trading export 통과:
  - experiment `stage4_film_gamma_i60_ohlc_ma_vb_r20_c60`
  - split `test`
  - predictions 4 rows
  - classification metrics 저장
  - trading metrics 저장

현재 경계:
- 4-I9는 prediction/classification/trading output 저장까지 담당합니다.
- 4-I10에서 Grad-CAM plus context/gate/gamma/beta interpretation export를
  추가합니다.
