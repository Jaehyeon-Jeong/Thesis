# 1-I10 Current Kaggle Output Status

## English

Status: in progress, partial artifacts archived

Checked local Stage 1 output folders:
- `stage1_i20_r20_seed42_outputs`
- `stage1_i20_r60_seed42_outputs`

Confirmed output status:

| Horizon | Status | Evidence | Interpretation |
|:---|:---|:---|:---|
| `I20/R5` | Pending | No full seed-42 output is currently archived locally. | Must be re-run or re-downloaded before reporting. |
| `I20/R20` | Smoke validation only | `validation_metrics.json` has only `4` validation samples, and metadata says `run_mode: smoke`. | Do not report as a full reproduction result. |
| `I20/R60` | Full seed-42 fast diagnostic | `test_metrics.json`, `test_predictions.csv`, checkpoint, train history, and test Grad-CAM preview exist. | Usable as current diagnostic result, but not final paper-style reproduction. |

`I20/R60` current metric snapshot:
- Test rows: `1,376,215`
- Accuracy: `0.5312`
- Majority-class accuracy: `0.5408`
- Accuracy minus majority: `-0.0096`
- ROC-AUC: `0.5298`
- Average precision: `0.5632`
- F1: `0.5982`

Execution caveat:
- Current `I20/R60` used fast Kaggle settings: batch size `1024`, mixed
  precision, and DataParallel.
- Strict paper-style reproduction remains later work: batch size `128` and five
  independent runs/seeds.
- The current Grad-CAM preview has `2` predicted-up and `2` predicted-down
  examples. Final Figure-13-style output should use `10` up and `10` down.

Output created:
- `reports/stage1_current_status_report.md`
- `reports/tables/stage1_seed42_current_status.csv`
- `reports/figures/gradcam/stage1_i20_r60_seed42_test_2019_figure13_style.png`
- `reports/figures/gradcam/stage1_i20_r20_seed42_validation_1993_smoke_figure13_style.png`

## 한국어

상태: 진행 중, 일부 artifact만 보존됨

확인한 로컬 Stage 1 output 폴더:
- `stage1_i20_r20_seed42_outputs`
- `stage1_i20_r60_seed42_outputs`

확인된 output 상태:

| Horizon | 상태 | 근거 | 해석 |
|:---|:---|:---|:---|
| `I20/R5` | 대기 | 현재 로컬에 full seed-42 output이 보존되어 있지 않습니다. | 보고 전에 재실행 또는 재다운로드가 필요합니다. |
| `I20/R20` | Smoke validation only | `validation_metrics.json` sample 수가 `4`이고 metadata가 `run_mode: smoke`입니다. | full reproduction 결과로 보고하면 안 됩니다. |
| `I20/R60` | Full seed-42 fast diagnostic | `test_metrics.json`, `test_predictions.csv`, checkpoint, train history, test Grad-CAM preview가 있습니다. | 현재 diagnostic 결과로는 사용할 수 있지만 최종 논문식 재현은 아닙니다. |

`I20/R60` 현재 metric snapshot:
- Test rows: `1,376,215`
- Accuracy: `0.5312`
- Majority-class accuracy: `0.5408`
- Accuracy minus majority: `-0.0096`
- ROC-AUC: `0.5298`
- Average precision: `0.5632`
- F1: `0.5982`

실행상 주의:
- 현재 `I20/R60`은 fast Kaggle setting을 사용했습니다: batch size `1024`,
  mixed precision, DataParallel.
- 논문식 strict reproduction은 나중에 수행합니다: batch size `128`, five
  independent runs/seeds.
- 현재 Grad-CAM preview는 predicted-up 2개, predicted-down 2개입니다. 최종
  Figure 13 스타일 output은 up 10개, down 10개로 다시 만들어야 합니다.

생성한 output:
- `reports/stage1_current_status_report.md`
- `reports/tables/stage1_seed42_current_status.csv`
- `reports/figures/gradcam/stage1_i20_r60_seed42_test_2019_figure13_style.png`
- `reports/figures/gradcam/stage1_i20_r20_seed42_validation_1993_smoke_figure13_style.png`
