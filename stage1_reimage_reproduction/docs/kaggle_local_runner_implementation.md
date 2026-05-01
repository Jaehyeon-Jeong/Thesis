# 1-I6 Kaggle/Local Runner Implementation

## English

Status:
- Completed on 2026-05-01.

Purpose:
- Connect the shared Stage 1 code into one config-driven runner for local smoke
  tests and Kaggle full runs.
- Write a run manifest for provenance.

Implemented files:
- `src/stage1_reimage/runners/__init__.py`
- `src/stage1_reimage/runners/stage1_baseline.py`
- `scripts/run_stage1_baseline.py`
- `notebooks/kaggle_stage1_single_horizon_one_cell.md`
- Updated `src/stage1_reimage/data/label_split.py`

Implemented behavior:
- One runner script supports local and Kaggle configs.
- `smoke`, `full_single_seed`, and `full_paper_style` run modes.
- Local non-smoke runs are blocked unless `--allow-local-full` is passed.
- Horizon-specific labels/splits/normalization are built inside the runner.
- Dataloaders are built from the memmap-backed public I20 dataset.
- `StockCNNI20` is trained through the shared training loop.
- `outputs/run_manifests/run_manifest.json` is written.

Local smoke command:

```bash
python scripts/run_stage1_baseline.py \
  --config configs/env_local.yaml \
  --run-mode smoke \
  --horizons stage1_i20_r20 \
  --max-train-rows 8 \
  --max-val-rows 4 \
  --normalization-max-images 128 \
  --max-epochs 1
```

Kaggle command:

```bash
python scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_single_seed
```

Validation result:
- Local smoke runner passed.
- `run_manifest.json` was written under `outputs/run_manifests/`.
- Smoke checkpoints and histories were written under `outputs/`.
- Smoke details:
  - horizon: `stage1_i20_r20`
  - run seed: `42`
  - train rows used: `8`
  - validation rows used: `4`
  - normalization images used: `128`
  - max epochs: `1`
  - device: `cpu`

Scope limits:
- This gate does not implement final evaluation metrics or prediction CSVs.
- This gate does not implement portfolio outputs.
- This gate does not implement Grad-CAM.
- Smoke outputs under `outputs/` are not reproduction results and are excluded
  from GitHub.

## 한국어

상태:
- 2026-05-01 완료.

목적:
- local smoke test와 Kaggle full run이 같은 Stage 1 code를 쓰도록 config 기반
  runner를 구현합니다.
- provenance 기록을 위해 run manifest를 작성합니다.

구현한 파일:
- `src/stage1_reimage/runners/__init__.py`
- `src/stage1_reimage/runners/stage1_baseline.py`
- `scripts/run_stage1_baseline.py`
- `notebooks/kaggle_stage1_single_horizon_one_cell.md`
- `src/stage1_reimage/data/label_split.py` 업데이트

구현한 동작:
- 하나의 runner script가 local config와 Kaggle config를 모두 지원합니다.
- `smoke`, `full_single_seed`, `full_paper_style` run mode 지원.
- local에서 non-smoke run은 `--allow-local-full` 없이는 막습니다.
- runner 내부에서 horizon별 label/split/normalization을 만듭니다.
- memmap 기반 public I20 dataset에서 DataLoader를 만듭니다.
- shared training loop로 `StockCNNI20`을 학습합니다.
- `outputs/run_manifests/run_manifest.json`을 작성합니다.

Local smoke command:

```bash
python scripts/run_stage1_baseline.py \
  --config configs/env_local.yaml \
  --run-mode smoke \
  --horizons stage1_i20_r20 \
  --max-train-rows 8 \
  --max-val-rows 4 \
  --normalization-max-images 128 \
  --max-epochs 1
```

Kaggle command:

```bash
python scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_single_seed
```

검증 결과:
- local smoke runner가 통과했습니다.
- `outputs/run_manifests/` 아래 `run_manifest.json`을 작성했습니다.
- smoke checkpoint와 history는 `outputs/` 아래 작성됐습니다.
- Smoke detail:
  - horizon: `stage1_i20_r20`
  - run seed: `42`
  - train rows used: `8`
  - validation rows used: `4`
  - normalization images used: `128`
  - max epochs: `1`
  - device: `cpu`

범위 제한:
- 이 gate에서는 final evaluation metric과 prediction CSV를 구현하지 않았습니다.
- portfolio output도 아직 구현하지 않았습니다.
- Grad-CAM도 아직 구현하지 않았습니다.
- `outputs/` 아래 smoke output은 reproduction result가 아니며 GitHub에서 제외합니다.
