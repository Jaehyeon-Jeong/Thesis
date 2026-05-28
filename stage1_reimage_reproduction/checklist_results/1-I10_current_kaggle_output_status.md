# 1-I10 Current Kaggle Output Status

## English

Status: seed-42 full-data fast diagnostics completed for all three I20 horizons

Checked/recorded Stage 1 outputs:
- `I20/R5`
- `I20/R20`
- `I20/R60`

Confirmed output status:

| Horizon | Status | Evidence | Interpretation |
|:---|:---|:---|:---|
| `I20/R5` | Full seed-42 fast diagnostic | `check_stage1_single_seed_outputs.py` returned `status: ok` with checkpoint, test predictions, metrics, correlation metrics, and test Grad-CAM. | Usable as current full-data diagnostic result, but not final paper-style reproduction. |
| `I20/R20` | Full seed-42 fast diagnostic | `check_stage1_single_seed_outputs.py` returned `status: ok` with checkpoint, test predictions, metrics, correlation metrics, and test Grad-CAM. | Usable as current full-data diagnostic result, but not final paper-style reproduction. |
| `I20/R60` | Full seed-42 fast diagnostic | Earlier archived result includes test metrics, test predictions, checkpoint, train history, and test Grad-CAM preview. | Usable as current diagnostic result, but not final paper-style reproduction. |

Metric snapshot:

| Horizon | Test rows | Accuracy | Majority-class accuracy | Accuracy minus majority | ROC-AUC | Average precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `I20/R5` | 1,399,933 | 0.5273 | 0.5078 | +0.0195 | 0.5373 | 0.5344 | 0.5489 |
| `I20/R20` | 1,393,845 | 0.5285 | 0.5222 | +0.0063 | 0.5339 | 0.5483 | 0.5817 |
| `I20/R60` | 1,376,215 | 0.5312 | 0.5408 | -0.0096 | 0.5298 | 0.5632 | 0.5982 |

Execution caveat:
- These runs used fast Kaggle settings: batch size `1024`, mixed precision,
  DataParallel, and fast cuDNN.
- Strict paper-style reproduction remains later work: batch size `128` and five
  independent runs/seeds.
- The current Grad-CAM previews have `2` predicted-up and `2` predicted-down
  examples. Final Figure-13-style output should use `10` up and `10` down.

Output/report files updated:
- `reports/stage1_current_status_report.md`
- `reports/tables/stage1_seed42_current_status.csv`

Kaggle backup zip names recorded from the run:
- `stage1_i20_r5_seed42_after_output_check_20260527T203319Z_outputs.zip`
- `stage1_i20_r20_seed42_after_output_check_20260527T223520Z_outputs.zip`
- `stage1_i20_r5_r20_seed42_final_all_outputs_20260527T223615Z_outputs.zip`

## 한국어

상태: I20 horizon 3개 전체 seed-42 full-data fast diagnostic 완료

확인/기록한 Stage 1 output:
- `I20/R5`
- `I20/R20`
- `I20/R60`

확인된 output 상태:

| Horizon | 상태 | 근거 | 해석 |
|:---|:---|:---|:---|
| `I20/R5` | Full seed-42 fast diagnostic | `check_stage1_single_seed_outputs.py`가 `status: ok`를 반환했고 checkpoint, test prediction, metric, correlation metric, test Grad-CAM이 모두 있습니다. | 현재 full-data diagnostic 결과로 사용 가능하지만 최종 paper-style reproduction은 아닙니다. |
| `I20/R20` | Full seed-42 fast diagnostic | `check_stage1_single_seed_outputs.py`가 `status: ok`를 반환했고 checkpoint, test prediction, metric, correlation metric, test Grad-CAM이 모두 있습니다. | 현재 full-data diagnostic 결과로 사용 가능하지만 최종 paper-style reproduction은 아닙니다. |
| `I20/R60` | Full seed-42 fast diagnostic | 이전에 보존한 결과에 test metric, test prediction, checkpoint, train history, test Grad-CAM preview가 있습니다. | 현재 diagnostic 결과로 사용 가능하지만 최종 paper-style reproduction은 아닙니다. |

Metric snapshot:

| Horizon | Test rows | Accuracy | Majority-class accuracy | Accuracy minus majority | ROC-AUC | Average precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `I20/R5` | 1,399,933 | 0.5273 | 0.5078 | +0.0195 | 0.5373 | 0.5344 | 0.5489 |
| `I20/R20` | 1,393,845 | 0.5285 | 0.5222 | +0.0063 | 0.5339 | 0.5483 | 0.5817 |
| `I20/R60` | 1,376,215 | 0.5312 | 0.5408 | -0.0096 | 0.5298 | 0.5632 | 0.5982 |

실행상 주의:
- 세 run 모두 fast Kaggle setting을 사용했습니다: batch size `1024`, mixed
  precision, DataParallel, fast cuDNN.
- 논문식 strict reproduction은 나중에 수행합니다: batch size `128`, five
  independent runs/seeds.
- 현재 Grad-CAM preview는 predicted-up 2개, predicted-down 2개입니다. 최종
  Figure 13 스타일 output은 up 10개, down 10개로 다시 만들어야 합니다.

업데이트한 output/report 파일:
- `reports/stage1_current_status_report.md`
- `reports/tables/stage1_seed42_current_status.csv`

Kaggle run에서 기록된 backup zip 이름:
- `stage1_i20_r5_seed42_after_output_check_20260527T203319Z_outputs.zip`
- `stage1_i20_r20_seed42_after_output_check_20260527T223520Z_outputs.zip`
- `stage1_i20_r5_r20_seed42_final_all_outputs_20260527T223615Z_outputs.zip`
