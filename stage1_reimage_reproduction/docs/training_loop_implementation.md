# 1-I5 Training Loop and Checkpoint Implementation

## English

Status:
- Completed on 2026-04-30.

Purpose:
- Implement the Stage 1 training loop and checkpoint machinery for
  `StockCNNI20`.
- Verify forward/backward, validation, early stopping state, checkpoint writing,
  history CSV, and metadata JSON with a synthetic local smoke test.

Implemented files:
- `src/stage1_reimage/training/__init__.py`
- `src/stage1_reimage/training/loop.py`
- `scripts/check_training_loop.py`
- Updated `configs/env_local.yaml`
- Updated `configs/env_kaggle.yaml`

Implemented behavior:
- Xavier uniform initialization for `Conv2d` and `Linear` weights.
- Zero bias for `Conv2d` and `Linear`.
- BatchNorm weight one and bias zero.
- `torch.nn.CrossEntropyLoss` on logits.
- Adam optimizer with learning rate `1e-5`.
- Per-epoch training and validation loss/accuracy.
- Validation-loss best checkpoint: `best.pt`.
- Final checkpoint: `last.pt`.
- Per-run `train_history.csv`.
- Per-run `train_metadata.json`.

Validation command:

```bash
python scripts/check_training_loop.py --config configs/env_local.yaml --max-epochs 2 --train-samples 8 --val-samples 4 --batch-size 2
```

Validation result:
- Passed locally on synthetic smoke data.
- Wrote `best.pt`, `last.pt`, `train_history.csv`, and `train_metadata.json`
  under `outputs/`.
- Smoke result:
  - device: `cpu`
  - best epoch: `1`
  - stopped epoch: `2`
  - stopped early: `false`

Scope limits:
- This gate does not run real `monthly_20d` training.
- This gate does not implement the full local/Kaggle runner.
- This gate does not implement final evaluation metrics, prediction CSVs,
  portfolio outputs, or Grad-CAM.
- Smoke outputs under `outputs/` are not reproduction results and are excluded
  from GitHub tracking.

## 한국어

상태:
- 2026-04-30 완료.

목적:
- `StockCNNI20`용 Stage 1 training loop와 checkpoint machinery를 구현합니다.
- synthetic local smoke test로 forward/backward, validation, early stopping
  state, checkpoint writing, history CSV, metadata JSON을 검증합니다.

구현한 파일:
- `src/stage1_reimage/training/__init__.py`
- `src/stage1_reimage/training/loop.py`
- `scripts/check_training_loop.py`
- `configs/env_local.yaml` 업데이트
- `configs/env_kaggle.yaml` 업데이트

구현한 동작:
- `Conv2d`, `Linear` weight에 Xavier uniform initialization.
- `Conv2d`, `Linear` bias는 zero.
- BatchNorm weight는 one, bias는 zero.
- logits에 `torch.nn.CrossEntropyLoss` 적용.
- Adam optimizer, learning rate `1e-5`.
- epoch별 training/validation loss와 accuracy.
- validation loss 기준 best checkpoint: `best.pt`.
- 종료 시점 checkpoint: `last.pt`.
- run별 `train_history.csv`.
- run별 `train_metadata.json`.

검증 명령:

```bash
python scripts/check_training_loop.py --config configs/env_local.yaml --max-epochs 2 --train-samples 8 --val-samples 4 --batch-size 2
```

검증 결과:
- synthetic smoke data로 로컬 통과했습니다.
- `outputs/` 아래 `best.pt`, `last.pt`, `train_history.csv`,
  `train_metadata.json`을 작성했습니다.
- Smoke result:
  - device: `cpu`
  - best epoch: `1`
  - stopped epoch: `2`
  - stopped early: `false`

범위 제한:
- 이 gate에서는 실제 `monthly_20d` training을 실행하지 않았습니다.
- full local/Kaggle runner는 아직 구현하지 않았습니다.
- 최종 evaluation metric, prediction CSV, portfolio output, Grad-CAM도
  아직 구현하지 않았습니다.
- `outputs/` 아래 smoke output은 reproduction result가 아니며 GitHub tracking에서 제외합니다.
