# Stage 2 Notebooks

## English

Available Kaggle cells:
- `kaggle_stage2_btc_ohlcv_audit_one_cell.md`: runs the BTC OHLCV data audit
  inside Kaggle.
- `kaggle_stage2_btc_baseline_one_cell.md`: planned one-cell interface for
  BTC baseline training, evaluation, trading metrics, and quick Grad-CAM. It is
  executable only after the Stage 2 implementation scripts are added.
- `kaggle_stage2_grid_single_seed_one_cell.md`: runs the 36-run Stage 2 grid
  once with seed `42`.
- `kaggle_stage2_grid_five_seed_one_cell.md`: runs the 180-run Stage 2 grid
  with seeds `42, 43, 44, 45, 46`.
- `kaggle_stage2_i20_i60_r20_five_seed_one_cell.md`: runs the selected 40-run
  robustness check for `I20/R20` and `I60/R20` across all four image specs and
  five seeds.
- `kaggle_stage2_results_viewer_one_cell.md`: summarizes grid outputs into
  tables and displays selected Grad-CAM figures.
- `kaggle_stage2_best_gradcam_10_one_cell.md`: regenerates the best
  single-seed configuration's Grad-CAM in a Re-Imagining Figure-13-style layout
  with 10 predicted-up and 10 predicted-down BTC images.

## 한국어

사용 가능한 Kaggle cell:
- `kaggle_stage2_btc_ohlcv_audit_one_cell.md`: Kaggle 안에서 BTC OHLCV data audit을
  실행합니다.
- `kaggle_stage2_btc_baseline_one_cell.md`: BTC baseline training, evaluation,
  trading metric, quick Grad-CAM을 위한 예정 one-cell interface입니다. Stage 2 구현
  script가 추가된 뒤 실행 가능합니다.
- `kaggle_stage2_grid_single_seed_one_cell.md`: seed `42` 하나로 Stage 2 전체
  36-run grid를 실행합니다.
- `kaggle_stage2_grid_five_seed_one_cell.md`: seed `42, 43, 44, 45, 46`으로
  Stage 2 전체 180-run grid를 실행합니다.
- `kaggle_stage2_i20_i60_r20_five_seed_one_cell.md`: `I20/R20`과 `I60/R20`만
  두고 image spec 4개와 seed 5개를 실행하는 선별 40-run robustness check입니다.
- `kaggle_stage2_results_viewer_one_cell.md`: grid output을 표로 요약하고 선택한
  Grad-CAM figure를 표시합니다.
- `kaggle_stage2_best_gradcam_10_one_cell.md`: single-seed 기준 best
  configuration의 Grad-CAM을 Re-Imagining Figure 13 스타일로 다시 생성합니다.
  `Predicted Up` 10개와 `Predicted Down` 10개, 총 20개 BTC image를 포함합니다.
