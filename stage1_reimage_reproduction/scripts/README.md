# Scripts

## English

Executable Stage 1 scripts are added only after the corresponding checklist
item is confirmed.

Current scripts:
- `check_scaffold.py`: verifies the 1-I1 package/config scaffold without
  loading image data or training a model.
- `check_data_loading.py`: verifies the 1-I2 monthly_20d shard discovery,
  row alignment, and sample image tensor shape without constructing labels or
  training a model.
- `check_label_split_normalization.py`: verifies the 1-I3 horizon labels,
  deterministic splits, and train-only normalization metadata writing without
  training a model.
- `check_model.py`: verifies the 1-I4 `StockCNNI20` model shape contract,
  parameter count, logits output, and Grad-CAM target-layer lookup.
- `check_training_loop.py`: verifies the 1-I5 training loop, backward pass,
  best/last checkpoints, history CSV, and metadata JSON on synthetic smoke data.
- `run_stage1_baseline.py`: runs the 1-I6 config-driven local/Kaggle baseline
  runner and writes checkpoints, train histories, and `run_manifest.json`.
- `evaluate_stage1_predictions.py`: runs the 1-I7 checkpoint evaluation path,
  writes prediction CSVs, metric JSONs, correlation JSONs, and averaged
  probability files.
- `generate_stage1_gradcam.py`: runs the 1-I8 Grad-CAM path, selecting Up/Down
  samples from prediction CSVs and writing Figure 13-style heatmap grids.
- `run_stage1_kaggle_single_seed.sh`: runs the 1-I10 Kaggle single-seed
  sequence: data check, seed-42 training, test evaluation, Grad-CAM, and output
  verification.
- `check_stage1_single_seed_outputs.py`: verifies that the expected 1-I10
  checkpoint, prediction, metric, manifest, and Grad-CAM files exist after a
  Kaggle run.

Example:

```bash
python scripts/check_scaffold.py --config configs/env_local.yaml --create-output-dirs
```

## 한국어

각 체크리스트 항목이 확인된 뒤에만 1단계 실행 스크립트를 추가합니다.

현재 script:
- `check_scaffold.py`: image data loading이나 model training 없이 1-I1
  package/config scaffold를 확인합니다.
- `check_data_loading.py`: label 생성이나 model training 없이 1-I2
  monthly_20d shard discovery, row alignment, sample image tensor shape를
  확인합니다.
- `check_label_split_normalization.py`: model training 없이 1-I3 horizon label,
  deterministic split, train-only normalization metadata writing을 확인합니다.
- `check_model.py`: 1-I4 `StockCNNI20` model shape contract, parameter count,
  logits output, Grad-CAM target-layer lookup을 확인합니다.
- `check_training_loop.py`: synthetic smoke data로 1-I5 training loop,
  backward pass, best/last checkpoint, history CSV, metadata JSON을 확인합니다.
- `run_stage1_baseline.py`: 1-I6 config 기반 local/Kaggle baseline runner를
  실행하고 checkpoint, train history, `run_manifest.json`을 작성합니다.
- `evaluate_stage1_predictions.py`: 1-I7 checkpoint evaluation path를 실행해
  prediction CSV, metric JSON, correlation JSON, averaged probability file을
  작성합니다.
- `generate_stage1_gradcam.py`: 1-I8 Grad-CAM path를 실행해 prediction CSV에서
  Up/Down sample을 고르고 Figure 13-style heatmap grid를 작성합니다.
- `run_stage1_kaggle_single_seed.sh`: 1-I10 Kaggle single-seed sequence를
  실행합니다. data check, seed-42 training, test evaluation, Grad-CAM, output
  verification을 한 번에 묶습니다.
- `check_stage1_single_seed_outputs.py`: Kaggle 실행 이후 1-I10에서 기대하는
  checkpoint, prediction, metric, manifest, Grad-CAM 파일이 실제로 있는지
  확인합니다.

예시:

```bash
python scripts/check_scaffold.py --config configs/env_local.yaml --create-output-dirs
```
