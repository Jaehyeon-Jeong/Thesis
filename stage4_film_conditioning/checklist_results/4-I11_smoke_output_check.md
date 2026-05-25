# 4-I11. Local Smoke Output Check

## English

Status: complete.

Implemented file:
- `scripts/check_stage4_outputs.py`

Smoke summary:
- `reports/smoke_tests/stage4_smoke_summary.json`

Purpose:
- Prevent another incomplete-run situation where a checkpoint exists but
  predictions, metrics, Grad-CAM, or interpretation exports are missing.
- Stage 4 is considered complete only when the full artifact bundle exists and
  parses.

Checker requirements:
- `best.pt`
- `last.pt`
- `train_history.csv`
- `train_metadata.json`
- `{split}_predictions.csv`
- `{split}_metrics.json`
- `{split}_trading_metrics.json`
- `btc_context_gradcam_{split}_{N}perclass.png`
- `samples.csv`
- `modulation_summary.csv`
- `modulation_values.json`
- `context_features.csv`
- `context_scaler.json`
- `context_feature_audit.json`
- `context_feature_summary.csv`
- `run_manifest.json`

The checker validates:
- File existence.
- Non-empty file size.
- CSV parseability and minimum row count.
- JSON parseability.
- JSON list item count for `modulation_values.json`.

Local checks run:

```bash
python scripts/check_stage4_outputs.py \
  --config configs/env_local.yaml \
  --context-method concat \
  --image-window 60 \
  --image-spec ohlc_ma_vb \
  --return-horizon 20 \
  --run-seed 42 \
  --split test \
  --gradcam-samples-per-class 2
```

Result:
- `stage4_concat_i60_ohlc_ma_vb_r20_c60`
- Status: `ok`
- Prediction rows: `8`
- Grad-CAM sample rows: `4`
- Modulation summary rows: `4`
- Modulation JSON items: `4`
- Context feature rows: `32`

```bash
python scripts/check_stage4_outputs.py \
  --config configs/env_local.yaml \
  --context-method film_gamma \
  --image-window 60 \
  --image-spec ohlc_ma_vb \
  --return-horizon 20 \
  --run-seed 42 \
  --split test \
  --gradcam-samples-per-class 2
```

Result:
- `stage4_film_gamma_i60_ohlc_ma_vb_r20_c60`
- Status: `ok`
- Prediction rows: `4`
- Grad-CAM sample rows: `4`
- Modulation summary rows: `4`
- Modulation JSON items: `4`
- Context feature rows: `32`

Boundary:
- The smoke rows are intentionally tiny and are not performance results.
- `4-I12` should run the real Kaggle single-config/four-ablation experiment.

## 한국어

상태: 완료.

구현 파일:
- `scripts/check_stage4_outputs.py`

Smoke summary:
- `reports/smoke_tests/stage4_smoke_summary.json`

목적:
- checkpoint만 있고 prediction, metric, Grad-CAM, interpretation export가 빠지는
  incomplete-run 상황을 막습니다.
- Stage 4 완료 기준은 전체 artifact bundle이 존재하고 parse 가능한 상태입니다.

Checker 요구 산출물:
- `best.pt`
- `last.pt`
- `train_history.csv`
- `train_metadata.json`
- `{split}_predictions.csv`
- `{split}_metrics.json`
- `{split}_trading_metrics.json`
- `btc_context_gradcam_{split}_{N}perclass.png`
- `samples.csv`
- `modulation_summary.csv`
- `modulation_values.json`
- `context_features.csv`
- `context_scaler.json`
- `context_feature_audit.json`
- `context_feature_summary.csv`
- `run_manifest.json`

Checker가 확인하는 것:
- 파일 존재 여부.
- 비어 있지 않은 file size.
- CSV parse 가능 여부와 최소 row 수.
- JSON parse 가능 여부.
- `modulation_values.json`의 JSON list item 수.

로컬 실행:

```bash
python scripts/check_stage4_outputs.py \
  --config configs/env_local.yaml \
  --context-method concat \
  --image-window 60 \
  --image-spec ohlc_ma_vb \
  --return-horizon 20 \
  --run-seed 42 \
  --split test \
  --gradcam-samples-per-class 2
```

결과:
- `stage4_concat_i60_ohlc_ma_vb_r20_c60`
- Status: `ok`
- Prediction rows: `8`
- Grad-CAM sample rows: `4`
- Modulation summary rows: `4`
- Modulation JSON items: `4`
- Context feature rows: `32`

```bash
python scripts/check_stage4_outputs.py \
  --config configs/env_local.yaml \
  --context-method film_gamma \
  --image-window 60 \
  --image-spec ohlc_ma_vb \
  --return-horizon 20 \
  --run-seed 42 \
  --split test \
  --gradcam-samples-per-class 2
```

결과:
- `stage4_film_gamma_i60_ohlc_ma_vb_r20_c60`
- Status: `ok`
- Prediction rows: `4`
- Grad-CAM sample rows: `4`
- Modulation summary rows: `4`
- Modulation JSON items: `4`
- Context feature rows: `32`

경계:
- Smoke row는 의도적으로 작게 제한한 것이며 성능 결과가 아닙니다.
- `4-I12`에서 실제 Kaggle single-config/four-ablation 실험을 실행해야 합니다.
