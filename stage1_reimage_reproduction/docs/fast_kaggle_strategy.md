# Stage 1 Fast Kaggle Strategy

## English

Purpose:
- Run the Stage 1 Re-image stock baseline on Kaggle as quickly and safely as
  possible without changing the research design.

Why the current stock run is much heavier than the BTC example:
- BTC example has only a few thousand daily windows.
- Stage 1 stock baseline has roughly 2.18M valid image rows for `I20/R20`.
- `I20/R20` alone has about 550K train rows, 236K validation rows, and 1.39M
  test rows.
- The BTC example keeps generated images in RAM.
- Stage 1 stock baseline reads public `.dat` shards through memmap and trains
  on many more samples.

Fast execution rule:
- Do not run all horizons and Grad-CAM in one command during debugging.
- Run one horizon first, preferably `stage1_i20_r20`.
- Run Grad-CAM only after prediction CSVs exist.
- Start Grad-CAM with `--samples-per-class 2`; use `10` only for report output.

One-cell Kaggle runner:
- `notebooks/kaggle_stage1_single_horizon_one_cell.md`
- This is the recommended copy-paste interface for Kaggle.
- It keeps the codebase modular but lets the notebook run code copy, data copy,
  config patching, training, evaluation, and Grad-CAM from one cell.

Recommended Kaggle setup:

```bash
df -h /tmp /kaggle/working
du -sh /kaggle/input/datasets/moskow/baseline-dataset
```

If disk space is enough, copy data to local temporary disk:

```bash
rm -rf /tmp/baseline-dataset
mkdir -p /tmp/baseline-dataset
cp -R /kaggle/input/datasets/moskow/baseline-dataset/* /tmp/baseline-dataset/
```

Patch the config in the Kaggle notebook:

```python
from pathlib import Path
import yaml

config_path = Path("/kaggle/working/stage1_reimage_reproduction/configs/env_kaggle.yaml")
cfg = yaml.safe_load(config_path.read_text())

cfg["paths"]["data_root"] = "/tmp/baseline-dataset"
cfg["data"]["monthly20_root"] = "/tmp/baseline-dataset"
cfg["runtime"]["num_workers"] = 4
cfg["runtime"]["pin_memory"] = True
cfg["runtime"]["persistent_workers"] = True
cfg["training"]["log_every_batches"] = 20

config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
```

First full single-horizon run:

```bash
python -u scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_single_seed \
  --horizons stage1_i20_r20 \
  --run-seeds 42
```

Then evaluate only that horizon:

```bash
python -u scripts/evaluate_stage1_predictions.py \
  --config configs/env_kaggle.yaml \
  --horizon stage1_i20_r20 \
  --run-seed 42 \
  --split test
```

Then quick Grad-CAM:

```bash
python -u scripts/generate_stage1_gradcam.py \
  --config configs/env_kaggle.yaml \
  --horizon stage1_i20_r20 \
  --run-seed 42 \
  --split test \
  --year 2019 \
  --samples-per-class 2 \
  --write-report-copy
```

Only after this works:
- Repeat for `stage1_i20_r5`.
- Repeat for `stage1_i20_r60`.
- Generate report Grad-CAM with `--samples-per-class 10`.

Performance code change:
- Training/validation DataLoader no longer collates metadata.
- Metadata is still retained for evaluation/prediction CSV.
- This keeps the research design unchanged while reducing avoidable CPU
  overhead during training.

Fast runtime fix added on 2026-05-01:
- Training/validation images can now be pre-loaded into RAM before training.
  This changes data access speed, not the model, label, split, or target.
- The Kaggle one-cell runner enables:
  - `preload_train_val_images=True`
  - `batch_size=1024`
  - `mixed_precision=True`
  - `data_parallel=True`
  - `cudnn_benchmark=True`
- If CUDA runs out of memory, set `BATCH_SIZE=512` first. If the run is still
  unstable, set `DATA_PARALLEL=False`.
- The expected fast-run log must include:
  - `[preload:train] ...`
  - `[preload:validation] ...`
  - `[train] mixed_precision=True ...`
  - `[train] enabling DataParallel ...` when Kaggle provides two GPUs.
- If the log only prints `batch=20 loss=...`, the Kaggle code snapshot is old.
  Re-upload the latest `stage1_reimage_reproduction` folder or zip.

Expected runtime, not guaranteed:
- Old lazy-memmap path: often 20+ minutes per epoch on Kaggle, sometimes worse.
- Fast RAM-preload path on Kaggle T4 x2:
  - code/data copy: about 3-10 minutes depending on Kaggle input storage.
  - train/validation preload: about 2-8 minutes.
  - one epoch: target about 2-6 minutes after preload.
  - one horizon including training, test prediction, and quick Grad-CAM:
    about 25-60 minutes.
  - all three `I20/R5`, `I20/R20`, `I20/R60`: about 1.5-3.5 hours.
- Strict reproduction mode with batch 128 and FP32 is slower. Keep it for the
  final audit run, not for debugging the pipeline.

Paid compute decision:
- Colab Pro/Pro+ may be faster only if it actually gives an L4/A100-class GPU
  and the dataset sits on fast local disk. It is not guaranteed.
- RunPod/Lambda/GCP with A10/A100 and local NVMe is more likely to be faster
  than Colab for this workload.
- Immediate recommendation: first run the new Kaggle fast path. Pay only if
  one horizon still takes more than about one hour after the `[preload:*]`
  messages appear.

## 한국어

목적:
- 연구 설계를 바꾸지 않고 Stage 1 Re-image stock baseline을 Kaggle에서 최대한
  빠르고 안전하게 실행합니다.

지금 stock run이 BTC 예시보다 훨씬 무거운 이유:
- BTC 예시는 daily window가 수천 개 수준입니다.
- Stage 1 stock baseline은 `I20/R20`만 해도 valid image row가 약 218만 개입니다.
- `I20/R20` 하나만 봐도 train 약 55만 row, validation 약 23만 row, test 약
  139만 row입니다.
- BTC 예시는 생성된 이미지를 RAM에 올려둔 구조입니다.
- Stage 1 stock baseline은 공개 `.dat` shard를 memmap으로 읽고 훨씬 많은 sample을
  학습합니다.

빠른 실행 원칙:
- debugging 중에는 모든 horizon과 Grad-CAM을 한 명령으로 실행하지 않습니다.
- 먼저 horizon 하나, 가능하면 `stage1_i20_r20`부터 실행합니다.
- Grad-CAM은 prediction CSV가 생긴 뒤에만 실행합니다.
- Grad-CAM은 먼저 `--samples-per-class 2`로 확인하고, 보고용만 `10`으로 만듭니다.

Kaggle one-cell runner:
- `notebooks/kaggle_stage1_single_horizon_one_cell.md`
- Kaggle에서 복붙해서 실행하는 권장 interface입니다.
- codebase는 모듈 구조를 유지하면서, Notebook에서는 code copy, data copy, config
  patching, training, evaluation, Grad-CAM을 한 셀에서 실행합니다.

권장 Kaggle setup:

```bash
df -h /tmp /kaggle/working
du -sh /kaggle/input/datasets/moskow/baseline-dataset
```

disk 여유가 있으면 데이터를 local temporary disk로 복사합니다.

```bash
rm -rf /tmp/baseline-dataset
mkdir -p /tmp/baseline-dataset
cp -R /kaggle/input/datasets/moskow/baseline-dataset/* /tmp/baseline-dataset/
```

Kaggle Notebook에서 config를 수정합니다.

```python
from pathlib import Path
import yaml

config_path = Path("/kaggle/working/stage1_reimage_reproduction/configs/env_kaggle.yaml")
cfg = yaml.safe_load(config_path.read_text())

cfg["paths"]["data_root"] = "/tmp/baseline-dataset"
cfg["data"]["monthly20_root"] = "/tmp/baseline-dataset"
cfg["runtime"]["num_workers"] = 4
cfg["runtime"]["pin_memory"] = True
cfg["runtime"]["persistent_workers"] = True
cfg["training"]["log_every_batches"] = 20

config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
```

첫 full single-horizon run:

```bash
python -u scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_single_seed \
  --horizons stage1_i20_r20 \
  --run-seeds 42
```

그 다음 같은 horizon만 evaluation합니다.

```bash
python -u scripts/evaluate_stage1_predictions.py \
  --config configs/env_kaggle.yaml \
  --horizon stage1_i20_r20 \
  --run-seed 42 \
  --split test
```

그 다음 빠른 Grad-CAM:

```bash
python -u scripts/generate_stage1_gradcam.py \
  --config configs/env_kaggle.yaml \
  --horizon stage1_i20_r20 \
  --run-seed 42 \
  --split test \
  --year 2019 \
  --samples-per-class 2 \
  --write-report-copy
```

이게 정상 작동한 뒤:
- `stage1_i20_r5`를 반복합니다.
- `stage1_i20_r60`을 반복합니다.
- 보고용 Grad-CAM은 `--samples-per-class 10`으로 생성합니다.

성능 관련 코드 변경:
- training/validation DataLoader가 더 이상 metadata를 collate하지 않게 했습니다.
- evaluation/prediction CSV에서는 metadata를 그대로 유지합니다.
- 연구 설계는 유지하면서 training 중 불필요한 CPU overhead만 줄인 변경입니다.

2026-05-01 추가 fast runtime 수정:
- training/validation 이미지를 학습 전에 RAM으로 pre-load할 수 있게 했습니다.
  이는 data access 속도만 바꾸며 model, label, split, target은 바꾸지 않습니다.
- Kaggle one-cell runner는 아래 fast setting을 켭니다.
  - `preload_train_val_images=True`
  - `batch_size=1024`
  - `mixed_precision=True`
  - `data_parallel=True`
  - `cudnn_benchmark=True`
- CUDA OOM이 나면 먼저 `BATCH_SIZE=512`로 낮춥니다. 그래도 불안정하면
  `DATA_PARALLEL=False`로 둡니다.
- 정상 fast run log에는 반드시 아래 문구가 보여야 합니다.
  - `[preload:train] ...`
  - `[preload:validation] ...`
  - `[train] mixed_precision=True ...`
  - Kaggle에서 GPU 두 개가 잡히면 `[train] enabling DataParallel ...`
- log가 여전히 `batch=20 loss=...`만 찍히면 Kaggle code snapshot이 예전
  버전입니다. 최신 `stage1_reimage_reproduction` 폴더나 zip을 다시 업로드해야 합니다.

예상 시간:
- 기존 lazy-memmap path: Kaggle에서 epoch 하나가 20분 이상 걸릴 수 있습니다.
- Kaggle T4 x2 fast RAM-preload path:
  - code/data copy: Kaggle input storage 상태에 따라 약 3-10분.
  - train/validation preload: 약 2-8분.
  - preload 이후 epoch 하나: 목표 약 2-6분.
  - horizon 하나의 training, test prediction, quick Grad-CAM: 약 25-60분.
  - `I20/R5`, `I20/R20`, `I20/R60` 세 개 전체: 약 1.5-3.5시간.
- batch 128, FP32 strict reproduction mode는 더 느립니다. 이 설정은 pipeline
  debugging이 아니라 최종 audit run용으로 남깁니다.

유료 compute 판단:
- Colab Pro/Pro+는 L4/A100급 GPU가 실제로 배정되고 데이터가 빠른 local disk에
  있을 때만 빨라질 수 있습니다. 배정이 보장되지는 않습니다.
- RunPod/Lambda/GCP처럼 A10/A100과 local NVMe를 고르는 환경이 이 workload에는
  Colab보다 더 확실하게 빠릅니다.
- 지금 당장 권장하는 선택은 새 Kaggle fast path를 먼저 돌리는 것입니다.
  `[preload:*]`가 보이는데도 horizon 하나가 1시간을 크게 넘으면 그때 유료 GPU를
  판단합니다.
