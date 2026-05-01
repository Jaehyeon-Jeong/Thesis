# 3-5 Kaggle Runner and Output Plan

## English

Purpose:
- Define the Kaggle-first execution path for Stage 3.
- Preserve outputs after every long-running step.

Kaggle runner pattern:
- Use the same pattern as Stage 2:
  - copy code snapshot to `/kaggle/working/stage3_linear_adapter`;
  - attach the public BTC OHLCV dataset;
  - optionally attach Stage 2 baseline result artifacts for comparison tables;
  - run train, prediction evaluation, trading evaluation, Grad-CAM, output check;
  - backup `outputs/stage3` after each major step.

Planned notebook cells:
- `notebooks/kaggle_stage3_linear_single_config_one_cell.md`
  - Runs one configuration, default `I60/R20/ohlc_ma_vb`, seed `42`.
- `notebooks/kaggle_stage3_linear_grid_single_seed_one_cell.md`
  - Runs `36` Linear experiments for seed `42`.
- `notebooks/kaggle_stage3_results_viewer_one_cell.md`
  - Builds baseline-vs-Linear tables and selected Grad-CAM displays.

Planned scripts:
- `scripts/run_stage3_linear.py`
- `scripts/evaluate_stage3_predictions.py`
- `scripts/evaluate_stage3_trading.py`
- `scripts/generate_stage3_gradcam.py`
- `scripts/compare_stage3_vs_stage2.py`
- `scripts/check_stage3_outputs.py`
- optional `scripts/run_stage3_grid.py`

Output roots:

```text
outputs/stage3/
  checkpoints/
  metrics/
  predictions/
  figures/
  run_manifests/

reports/
  tables/
  figures/gradcam/
```

Experiment naming:

```text
stage3_i{image_window}_{image_spec}_r{return_horizon}_linear_d{adapter_dim}
```

Example:

```text
stage3_i60_ohlc_ma_vb_r20_linear_d128
```

Backup rule:
- Backup after training.
- Backup after prediction evaluation.
- Backup after trading evaluation.
- Backup after Grad-CAM.
- Backup after final output check.
- Store backup zips under `/kaggle/working/stage3_saved_outputs`.

GitHub tracking rule:
- Track only small Markdown reports, configs, source code, small summary CSVs,
  and selected small Grad-CAM preview figures.
- Do not track checkpoints, large predictions, raw Kaggle backups, or zip files.

## 한국어

목적:
- Stage 3의 Kaggle-first 실행 경로를 정의합니다.
- 오래 걸리는 단계가 끝날 때마다 output을 보존합니다.

Kaggle runner pattern:
- Stage 2와 같은 pattern을 사용합니다:
  - code snapshot을 `/kaggle/working/stage3_linear_adapter`로 복사;
  - public BTC OHLCV dataset attach;
  - baseline comparison table을 위해 필요하면 Stage 2 baseline result artifact attach;
  - train, prediction evaluation, trading evaluation, Grad-CAM, output check 실행;
  - 각 주요 단계 뒤 `outputs/stage3`를 backup.

예정 notebook cell:
- `notebooks/kaggle_stage3_linear_single_config_one_cell.md`
  - 기본 `I60/R20/ohlc_ma_vb`, seed `42` 한 configuration 실행.
- `notebooks/kaggle_stage3_linear_grid_single_seed_one_cell.md`
  - seed `42` 기준 Linear 36개 실험 실행.
- `notebooks/kaggle_stage3_results_viewer_one_cell.md`
  - baseline-vs-Linear table과 선택 Grad-CAM display 생성.

예정 script:
- `scripts/run_stage3_linear.py`
- `scripts/evaluate_stage3_predictions.py`
- `scripts/evaluate_stage3_trading.py`
- `scripts/generate_stage3_gradcam.py`
- `scripts/compare_stage3_vs_stage2.py`
- `scripts/check_stage3_outputs.py`
- optional `scripts/run_stage3_grid.py`

Output root:

```text
outputs/stage3/
  checkpoints/
  metrics/
  predictions/
  figures/
  run_manifests/

reports/
  tables/
  figures/gradcam/
```

Experiment naming:

```text
stage3_i{image_window}_{image_spec}_r{return_horizon}_linear_d{adapter_dim}
```

예시:

```text
stage3_i60_ohlc_ma_vb_r20_linear_d128
```

Backup 규칙:
- 학습 직후 backup.
- prediction evaluation 직후 backup.
- trading evaluation 직후 backup.
- Grad-CAM 직후 backup.
- final output check 직후 backup.
- backup zip은 `/kaggle/working/stage3_saved_outputs`에 저장합니다.

GitHub tracking 규칙:
- 작은 Markdown report, config, source code, 작은 summary CSV, 선택된 작은
  Grad-CAM preview figure만 track합니다.
- checkpoint, 대용량 prediction, raw Kaggle backup, zip file은 track하지 않습니다.
