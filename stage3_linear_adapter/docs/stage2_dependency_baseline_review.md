# 3-1 Stage 2 Dependency and Baseline-Output Review

## English

Purpose:
- Confirm exactly what Stage 3 inherits from Stage 2.
- Identify which Stage 2 baseline outputs are available for comparison.
- Lock the rule that Stage 3 changes only the model adapter/head path.

Sources checked:
- Root `PLAN.md`, Stage 3 section.
- `stage2_btc_extension/README.md`
- `stage2_btc_extension/reports/stage2_single_seed_result_report.md`
- `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py`
- `stage2_btc_extension/src/stage2_btc/runners/btc_baseline.py`
- `stage2_btc_extension/configs/env_kaggle.yaml`

Inherited fixed Stage 2 components:

| Component | Stage 3 rule |
|:---|:---|
| BTC OHLCV loader | Reuse unchanged. |
| BTC chart image generator | Reuse unchanged for `I5`, `I20`, `I60`. |
| Image specs | Reuse `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`. |
| Label rule | Reuse `future_return > 0 -> label=1`, otherwise `0`. |
| Split | Reuse Stage 2 split policy: train/validation pool `2018-01-01` to `2020-12-31`, test `2021-01-01` to `2024-12-31`, random train/validation split within train/validation pool. |
| Normalization | Reuse train-only pixel mean/std fitting. |
| CNN feature extractor | Reuse Stage-2 Stock_CNN-style `I5`, `I20`, `I60` variants. |
| Metrics | Reuse classification metrics and BTC trading metrics. |
| Grad-CAM | Reuse CNN-block Grad-CAM requirement. |

Current Stage 2 baseline result status:
- Single-seed grid exists for `36` BTC baseline experiments:
  `3 image windows x 3 return horizons x 4 image specs x seed 42`.
- Five-seed baseline rerun is still pending.
- Stage 2 best single-seed configuration:
  `I60/R20/ohlc_ma_vb`, accuracy `0.6031`, ROC-AUC `0.6169`.

Stage 3 implication:
- First Stage 3 comparison should match Stage 2 single-seed grid where possible:
  `I5/I20/I60 x R5/R20/R60 x 4 image specs x seed 42`.
- Five-seed Stage 3 is deferred until after the quick single-seed comparison.
- Final claims remain provisional until Stage 2 and Stage 3 five-seed checks are
  available.

## 한국어

목적:
- Stage 3가 Stage 2에서 정확히 무엇을 상속하는지 확인합니다.
- 비교에 사용할 Stage 2 baseline output 상태를 확인합니다.
- Stage 3에서는 model adapter/head path만 바꾼다는 규칙을 고정합니다.

확인한 근거:
- Root `PLAN.md`의 Stage 3 section.
- `stage2_btc_extension/README.md`
- `stage2_btc_extension/reports/stage2_single_seed_result_report.md`
- `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py`
- `stage2_btc_extension/src/stage2_btc/runners/btc_baseline.py`
- `stage2_btc_extension/configs/env_kaggle.yaml`

Stage 2에서 고정해서 가져오는 것:

| Component | Stage 3 규칙 |
|:---|:---|
| BTC OHLCV loader | 그대로 재사용합니다. |
| BTC chart image generator | `I5`, `I20`, `I60` 모두 그대로 재사용합니다. |
| Image specs | `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb` 그대로 재사용합니다. |
| Label rule | `future_return > 0 -> label=1`, 그 외 `0` 그대로 사용합니다. |
| Split | Stage 2 split 정책을 유지합니다: train/validation pool `2018-01-01` to `2020-12-31`, test `2021-01-01` to `2024-12-31`, train/validation pool 내부 random split. |
| Normalization | train-only pixel mean/std fitting을 그대로 사용합니다. |
| CNN feature extractor | Stage-2 Stock_CNN-style `I5`, `I20`, `I60` variant를 그대로 사용합니다. |
| Metrics | classification metric과 BTC trading metric을 그대로 사용합니다. |
| Grad-CAM | CNN block Grad-CAM requirement를 유지합니다. |

현재 Stage 2 baseline 결과 상태:
- seed `42` 기준 `36`개 BTC baseline 실험 결과가 있습니다:
  `3 image window x 3 return horizon x 4 image spec x seed 42`.
- baseline five-seed rerun은 아직 예정입니다.
- Stage 2 single-seed best configuration:
  `I60/R20/ohlc_ma_vb`, accuracy `0.6031`, ROC-AUC `0.6169`.

Stage 3에 주는 의미:
- 첫 Stage 3 비교는 가능하면 Stage 2 single-seed grid와 같은 축으로 맞춥니다:
  `I5/I20/I60 x R5/R20/R60 x image spec 4개 x seed 42`.
- Stage 3 five-seed는 빠른 single-seed 비교 이후로 미룹니다.
- Stage 2와 Stage 3 five-seed 확인 전까지 최종 결론은 provisional로 둡니다.
