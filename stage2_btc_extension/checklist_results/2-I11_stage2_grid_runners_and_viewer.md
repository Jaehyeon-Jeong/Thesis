# 2-I11 Stage 2 Grid Runners and Result Viewer

## English

Status: ready for Kaggle execution

Added grid execution scripts:
- `scripts/run_stage2_grid.py`
- `scripts/summarize_stage2_grid_results.py`

Added Kaggle notebook cells:
- `notebooks/kaggle_stage2_grid_single_seed_one_cell.md`
  - Runs 36 experiments:
    `3 image windows x 3 return horizons x 4 image specs x 1 seed`.
  - Seed: `42`.
- `notebooks/kaggle_stage2_grid_five_seed_one_cell.md`
  - Runs 180 experiments:
    `3 image windows x 3 return horizons x 4 image specs x 5 seeds`.
  - Seeds: `42, 43, 44, 45, 46`.
- `notebooks/kaggle_stage2_results_viewer_one_cell.md`
  - Reads current outputs and backup zips.
  - Writes seed-level and mean/std CSV tables.
  - Displays compact tables and selected Grad-CAM figures.

Experiment grid:
- Image windows: `I5`, `I20`, `I60`
- Return horizons: `R5`, `R20`, `R60`
- Image specs:
  - `ohlc`
  - `ohlc_vb`
  - `ohlc_ma`
  - `ohlc_ma_vb`

Model variants:
- `I5` uses the `stock_cnn_i5` variant: 2 convolution blocks, channels
  `64 -> 128`, expected params `155,138`.
- `I20` uses the `stock_cnn_i20` variant: 3 convolution blocks, channels
  `64 -> 128 -> 256`, expected params `708,866`.
- `I60` uses the `stock_cnn_i60` variant: 4 convolution blocks, channels
  `64 -> 128 -> 256 -> 512`, expected params `2,952,962`.
- The grid runner passes `image_window` to the model factory. It does not reuse
  the I20 model for I5/I60.

Grad-CAM display:
- Grad-CAM samples are split into `Predicted Up` and `Predicted Down` panels.
- Each column title records date, predicted class, `prob_up`, true label, and
  whether the prediction was correct.
- The heatmap is still computed from the target predicted class logit; the true
  label is displayed as interpretation metadata.

Output preservation:
- Grid runner creates per-experiment backup zips under
  `/kaggle/working/stage2_saved_outputs/`.
- It supports `--skip-completed` for resume-style execution.
- It supports `--continue-on-error` so one failed configuration does not erase
  completed outputs.

Validation:
- `python -m py_compile` passed for:
  - `scripts/run_stage2_grid.py`
  - `scripts/summarize_stage2_grid_results.py`
- A local dry-run of one grid item passed.
- A local summary read of the existing smoke output passed.

Important:
- The grid runners are execution wrappers. They do not change the CNN model,
  label construction, split rule, image generator, or evaluation implementation.
- The five-seed grid can be long. If necessary, set `RUN_GRADCAM = False` first
  to collect metric/trading results, then generate Grad-CAM for selected
  configurations.

## 한국어

상태: Kaggle 실행 준비 완료

추가한 grid 실행 script:
- `scripts/run_stage2_grid.py`
- `scripts/summarize_stage2_grid_results.py`

추가한 Kaggle notebook cell:
- `notebooks/kaggle_stage2_grid_single_seed_one_cell.md`
  - 36개 실험 실행:
    `3 image window x 3 return horizon x 4 image spec x 1 seed`.
  - Seed: `42`.
- `notebooks/kaggle_stage2_grid_five_seed_one_cell.md`
  - 180개 실험 실행:
    `3 image window x 3 return horizon x 4 image spec x 5 seed`.
  - Seeds: `42, 43, 44, 45, 46`.
- `notebooks/kaggle_stage2_results_viewer_one_cell.md`
  - 현재 output과 backup zip을 읽습니다.
  - seed-level table과 mean/std CSV table을 저장합니다.
  - compact table과 선택한 Grad-CAM figure를 표시합니다.

실험 grid:
- Image window: `I5`, `I20`, `I60`
- Return horizon: `R5`, `R20`, `R60`
- Image spec:
  - `ohlc`
  - `ohlc_vb`
  - `ohlc_ma`
  - `ohlc_ma_vb`

Model variant:
- `I5`는 `stock_cnn_i5` variant를 사용합니다: convolution block 2개,
  channel `64 -> 128`, expected params `155,138`.
- `I20`은 `stock_cnn_i20` variant를 사용합니다: convolution block 3개,
  channel `64 -> 128 -> 256`, expected params `708,866`.
- `I60`은 `stock_cnn_i60` variant를 사용합니다: convolution block 4개,
  channel `64 -> 128 -> 256 -> 512`, expected params `2,952,962`.
- Grid runner는 `image_window`를 model factory에 넘깁니다. I5/I60에서 I20
  model을 재사용하지 않습니다.

Grad-CAM 표시:
- Grad-CAM sample은 `Predicted Up` panel과 `Predicted Down` panel로 나눠
  보여줍니다.
- 각 column title에는 date, predicted class, `prob_up`, true label, correct
  여부를 같이 기록합니다.
- Heatmap 계산 자체는 target predicted class logit 기준으로 유지하고, true label은
  해석 metadata로 표시합니다.

Output 보존:
- Grid runner는 experiment별 backup zip을
  `/kaggle/working/stage2_saved_outputs/` 아래에 만듭니다.
- `--skip-completed`를 지원하므로 중단 후 이어서 실행할 수 있습니다.
- `--continue-on-error`를 지원하므로 한 configuration이 실패해도 완료된 output은
  남습니다.

검증:
- 아래 두 script의 `python -m py_compile` 통과:
  - `scripts/run_stage2_grid.py`
  - `scripts/summarize_stage2_grid_results.py`
- local dry-run으로 grid item 하나 통과.
- 기존 smoke output을 대상으로 summary read 통과.

중요:
- Grid runner는 실행 wrapper입니다. CNN model, label construction, split rule,
  image generator, evaluation 구현은 바꾸지 않습니다.
- five-seed grid는 오래 걸릴 수 있습니다. 필요하면 먼저 `RUN_GRADCAM = False`로
  metric/trading result를 모은 뒤, 선택된 configuration만 Grad-CAM을 다시 생성합니다.
