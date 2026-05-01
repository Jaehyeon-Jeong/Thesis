# 2-I12. Stage 2 Single-Seed Result Report and Figure-13-Style Grad-CAM

## English

Status: report draft prepared, Figure-13-style Grad-CAM generation cell prepared.

This checklist item records the Stage 2 single-seed result report handoff.

Completed:
- Copied the small Stage 2 result tables from `stage2_result_save/tables/` into
  `reports/tables/`.
- Prepared `reports/stage2_single_seed_result_report.md` with the single-seed
  result interpretation.
- Identified the best single-seed configuration:
  `I60 / R20 / ohlc_ma_vb / seed 42`.
- Added `notebooks/kaggle_stage2_best_gradcam_10_one_cell.md`.
  This Kaggle cell regenerates a Re-Imagining Figure-13-style Grad-CAM with:
  - `Predicted Up` (`pred_class = 1`): 10 BTC images
  - `Predicted Down` (`pred_class = 0`): 10 BTC images
  - Total: 20 original BTC chart images plus layer-wise Grad-CAM rows
- The Kaggle cell copies the Stage 2 code snapshot into `/kaggle/working` if it
  is missing and patches `env_kaggle.yaml` to point at the Kaggle BTC dataset.

Important limitation:
- The local `stage2_result_save` folder contains metrics and a quick
  `2`-per-class Grad-CAM preview.
- It does not contain the checkpoint or full prediction CSV for the best
  configuration.
- Therefore the final `10`-per-class Figure-13-style image cannot be regenerated
  locally from the current downloaded result folder.

Required final artifact after running the Kaggle cell:
- `reports/figures/gradcam/stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam_10perclass.png`
- `reports/figures/gradcam/stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam_10perclass_samples.csv`

After those files are downloaded from Kaggle and placed in
`stage2_btc_extension/reports/figures/gradcam/`, the Stage 2 single-seed result
report can be finalized and published as a complete result package.

## 한국어

상태: 결과 보고서 초안 준비 완료, Figure 13 스타일 Grad-CAM 생성 cell 준비 완료.

이 체크리스트 항목은 Stage 2 single-seed 결과 보고와 Grad-CAM handoff를 기록합니다.

완료한 것:
- `stage2_result_save/tables/`의 작은 Stage 2 결과표를 `reports/tables/`로 정리했습니다.
- `reports/stage2_single_seed_result_report.md`에 single-seed 결과 해석을 정리했습니다.
- single-seed 기준 best configuration을 확인했습니다:
  `I60 / R20 / ohlc_ma_vb / seed 42`.
- `notebooks/kaggle_stage2_best_gradcam_10_one_cell.md`를 추가했습니다.
  이 Kaggle cell은 Re-Imagining Figure 13 스타일 Grad-CAM을 다시 생성합니다.
  - `Predicted Up` (`pred_class = 1`): BTC image 10개
  - `Predicted Down` (`pred_class = 0`): BTC image 10개
  - 총 20개 원본 BTC chart image와 layer별 Grad-CAM row
- 이 Kaggle cell은 필요하면 Stage 2 code snapshot을 `/kaggle/working`으로 복사하고,
  `env_kaggle.yaml`이 Kaggle BTC dataset을 가리키도록 patch합니다.

중요한 제한사항:
- 현재 로컬 `stage2_result_save`에는 metric과 빠른 확인용 `2`-per-class Grad-CAM
  preview만 있습니다.
- best configuration의 checkpoint와 전체 prediction CSV는 포함되어 있지 않습니다.
- 따라서 지금 내려받은 결과 폴더만으로는 로컬에서 `10`-per-class Figure 13 스타일
  이미지를 다시 생성할 수 없습니다.

Kaggle cell 실행 뒤 필요한 최종 산출물:
- `reports/figures/gradcam/stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam_10perclass.png`
- `reports/figures/gradcam/stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam_10perclass_samples.csv`

이 두 파일을 Kaggle에서 내려받아
`stage2_btc_extension/reports/figures/gradcam/`에 넣으면 Stage 2 single-seed 결과
보고서를 완성된 결과 패키지로 최종 publish할 수 있습니다.
