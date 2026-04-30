# Stage 1 Training Loop Detail Plan

## English

Status:
- Stage 1-6 completed as a detail plan.
- No training code has been implemented yet.

## Purpose

Define the Stage 1 baseline CNN training loop before implementation. This plan
fixes the loss, optimizer, initialization, early stopping, checkpointing, run
seeds, logging, and full-run versus local smoke-test behavior.

## Sources Checked

Process source:
- `../PLAN.md`
- `docs/stage1_checklist.md`
- `docs/source_map.md`
- `docs/split_normalization_plan.md`
- `docs/baseline_cnn_implementation_plan.md`

Paper/source references:
- `../자료조사/Re-image 요약.md`
- `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
- `../stage0_data_check/docs/source_reference_check.md`

Reference implementation:
- `https://github.com/lich99/Stock_CNN`
- Commit: `415e2acf2a5013afca67e383acd3edc61fced840`
- `models/baseline.py`

## Paper Training Details to Follow

From the local Re-image summary and Stage 0 source audit:

| Item | Paper/source value |
| --- | --- |
| Task | Binary classification of future return sign |
| Loss | Cross-entropy |
| Output interpretation | Softmax Up probability, threshold 50% |
| Optimizer | Adam |
| Initial learning rate | `1e-5` |
| Batch size | `128` |
| Dropout | `0.5` in fully connected classifier |
| Initialization | Xavier initializer |
| Early stopping | Stop after validation loss does not improve for 2 epochs |
| Standardization | Use training-data pixel mean/std |
| Split | 1993-2000 train/validation, 2001-2019 test |
| Independent runs | 5 independent retrainings, then average predictions |

Paper-unreported items:
- Exact random seeds.
- Exact epoch count.
- Hardware.
- Weight decay.
- Learning-rate schedule.
- Gradient clipping.
- Mixed precision.
- DataLoader worker count.

## Training Scope

Train the Stage 1 I20 baseline for:
- `stage1_i20_r5`
- `stage1_i20_r20`
- `stage1_i20_r60`

Each horizon uses:
- Its own horizon-specific label filtering.
- Its own split metadata from `1-4`.
- Its own training-only pixel mean/std from `1-4`.
- The same `StockCNNI20` architecture from `1-5`.

## Full Paper-style Run

Full Stage 1 reproduction should train 5 independent models per horizon:

```text
horizon in [r5, r20, r60]
  run_seed in [42, 43, 44, 45, 46]
    initialize model with run_seed
    train with same horizon split and normalization rules
    keep best checkpoint by validation loss
```

Prediction aggregation:
- For each seed-run checkpoint, compute softmax probabilities on the test set.
- Average `prob_up` across the 5 independent runs.
- Final paper-style prediction uses averaged probability:
  - `pred_class = 1 if mean_prob_up >= 0.5 else 0`
- Save both individual-run predictions and averaged predictions.

Important seed distinction:
- `split_seed = 42` controls the train/validation split.
- `run_seed in [42, 43, 44, 45, 46]` controls model initialization and training
  stochasticity.

## Local Smoke-test Run

Local smoke tests are not reproduction results.

Local smoke-test defaults:
- `run_seeds: [42]`
- tiny subset only
- small number of train/validation batches
- used only to verify:
  - data loading
  - tensor shapes
  - forward/backward pass
  - checkpoint writing
  - metric file writing

Smoke-test outputs must be marked with:

```yaml
run_mode: smoke
report_as_reproduction: false
```

## Loss and Output

Training loss:
- `torch.nn.CrossEntropyLoss`
- Input: logits `(batch_size, 2)`
- Target: integer labels `(batch_size,)`, values `0` or `1`

No class weighting by default:
- The paper does not report class weighting.
- Positive rates are close to balanced for the current horizons.

Softmax:
- Do not apply softmax inside `model.forward`.
- Apply `torch.softmax(logits, dim=1)` only for metrics and prediction files.

## Optimizer

Use Adam:

```yaml
optimizer:
  name: adam
  learning_rate: 1.0e-5
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.0
```

Notes:
- The paper reports Adam and `1e-5`.
- Adam betas/eps and weight decay are not reported, so PyTorch defaults are used
  and recorded.
- No learning-rate schedule by default because the paper does not report one.

## Initialization

Use Xavier initialization before training:
- Apply Xavier initialization to `Conv2d` and `Linear` weights.
- Set `Conv2d` and `Linear` biases to zero when present.
- Set `BatchNorm2d` weights to one and biases to zero.

Default variant:
- `torch.nn.init.xavier_uniform_`

Reason:
- The paper reports Xavier initializer but does not specify uniform versus
  normal. The implementation must record the chosen variant.

## Epoch and Early Stopping

Epoch cap:
- `max_epochs: 100`

Reason:
- The paper does not report the exact number of epochs.
- `100` is a safety cap; early stopping is expected to determine the actual
  stopping epoch.

Validation:
- Run validation once after every training epoch.
- Monitor mean validation loss.

Early stopping:
- `patience: 2`
- `min_delta: 0.0`
- If validation loss does not improve for 2 consecutive epochs, stop.
- Save the best checkpoint whenever validation loss improves.
- Restore/use the best validation-loss checkpoint for test prediction.

## DataLoader Behavior

Training DataLoader:
- `batch_size: 128`
- `shuffle: true`
- seeded generator using the current `run_seed`
- `drop_last: false`

Validation/test DataLoader:
- `batch_size: 128`
- `shuffle: false`
- `drop_last: false`

Memory behavior:
- Use the lazy/memmap image loading plan from `1-2`.
- Do not eager-load all images into RAM.

DataLoader performance knobs:
- `num_workers`, `pin_memory`, and Kaggle-specific path/performance choices are
  deferred to `1-6K`.

## Device and Determinism

Device selection:
- Full run target: Kaggle CUDA GPU.
- Implementation should support `cuda`, `mps`, and `cpu`.
- Config default: `device: auto`.

Seed setup per run:
- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA, when available

Determinism defaults:

```yaml
determinism:
  enabled: true
  cudnn_deterministic: true
  cudnn_benchmark: false
```

Note:
- Exact bitwise reproducibility can still vary by GPU, driver, PyTorch version,
  and some CUDA kernels. This must be recorded in environment metadata.

Mixed precision:
- Disabled by default because the paper does not report AMP/mixed precision.

Gradient clipping:
- Disabled by default because the paper does not report it.

## Checkpoint Plan

Checkpoint root:

```text
outputs/checkpoints/{experiment_name}/seed_{run_seed}/
```

Example:

```text
outputs/checkpoints/stage1_i20_r20/seed_42/best.pt
outputs/checkpoints/stage1_i20_r20/seed_42/last.pt
```

Best checkpoint contents:
- model state dict
- optimizer state dict
- epoch
- best validation loss
- run seed
- split seed
- horizon/target name
- config snapshot
- normalization metadata
- source reference metadata

Save policy:
- Save `best.pt` whenever validation loss improves.
- Save `last.pt` at training stop.
- Test prediction uses `best.pt`.

## Logging and Metadata

Per-run history:

```text
outputs/metrics/{experiment_name}/seed_{run_seed}/train_history.csv
```

Minimum columns:
- `epoch`
- `train_loss`
- `val_loss`
- `train_accuracy`
- `val_accuracy`
- `learning_rate`
- `epoch_seconds`
- `best_so_far`

Per-run metadata:

```text
outputs/metrics/{experiment_name}/seed_{run_seed}/train_metadata.json
```

Minimum fields:
- experiment name
- target return name
- run mode: `full` or `smoke`
- run seed
- split seed
- model reference repo/commit
- paper source note
- optimizer settings
- initialization settings
- early stopping settings
- device
- package/environment summary
- start/end timestamp
- best epoch
- best validation loss

Console logging:
- Batch-level logging every `100` training batches by default.
- Epoch-level logging after every train and validation epoch.

## Planned Config Fields

```yaml
training:
  run_mode: full
  run_seeds: [42, 43, 44, 45, 46]
  batch_size: 128
  max_epochs: 100
  loss: cross_entropy
  class_weight: null
  optimizer:
    name: adam
    learning_rate: 1.0e-5
    betas: [0.9, 0.999]
    eps: 1.0e-8
    weight_decay: 0.0
  initialization:
    name: xavier_uniform
    conv_linear_bias: zero
    batchnorm_weight: one
    batchnorm_bias: zero
  early_stopping:
    monitor: val_loss
    mode: min
    patience: 2
    min_delta: 0.0
    restore_best: true
  dataloader:
    train_shuffle: true
    eval_shuffle: false
    drop_last: false
  device: auto
  determinism:
    enabled: true
    cudnn_deterministic: true
    cudnn_benchmark: false
  mixed_precision: false
  gradient_clipping: null
  log_every_batches: 100
```

## Deferred Items

Deferred to `1-6K`:
- Kaggle input/output paths.
- Kaggle dataset mount names.
- `num_workers` and `pin_memory` defaults for Kaggle.
- Notebook runner structure.
- Environment export and package/version recording details.

Deferred to `1-7`:
- Exact final metric list and prediction CSV schemas.
- Majority-class baseline comparison.
- 5-run averaged prediction file naming.

Deferred to `1-8`:
- Grad-CAM generation from trained checkpoints.
- Whether to generate Grad-CAM for individual seed models, averaged model
  outputs, or the best representative seed.

## 한국어

상태:
- 1-6을 Training loop 세부계획으로 완료했습니다.
- training code는 아직 구현하지 않았습니다.

## 목적

구현 전에 1단계 baseline CNN의 학습 루프를 고정합니다. 여기서는 loss,
optimizer, initialization, early stopping, checkpoint, run seed, logging,
full-run과 local smoke-test 구분을 정합니다.

## 확인한 근거

진행 기준:
- `../PLAN.md`
- `docs/stage1_checklist.md`
- `docs/source_map.md`
- `docs/split_normalization_plan.md`
- `docs/baseline_cnn_implementation_plan.md`

논문/근거:
- `../자료조사/Re-image 요약.md`
- `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
- `../stage0_data_check/docs/source_reference_check.md`

기준 구현:
- `https://github.com/lich99/Stock_CNN`
- commit: `415e2acf2a5013afca67e383acd3edc61fced840`
- `models/baseline.py`

## 논문에서 따라야 할 학습 설정

로컬 Re-image 요약과 0단계 source audit 기준:

| 항목 | 논문/근거 값 |
| --- | --- |
| Task | future return sign binary classification |
| Loss | cross-entropy |
| 출력 해석 | softmax Up probability, threshold 50% |
| Optimizer | Adam |
| Initial learning rate | `1e-5` |
| Batch size | `128` |
| Dropout | fully connected classifier에서 `0.5` |
| Initialization | Xavier initializer |
| Early stopping | validation loss가 2 epoch 개선되지 않으면 중단 |
| Standardization | training-data pixel mean/std 사용 |
| Split | 1993-2000 train/validation, 2001-2019 test |
| 독립 학습 | 5회 independent retraining 후 prediction average |

논문 미보고 항목:
- 정확한 random seed.
- 정확한 epoch 수.
- hardware.
- weight decay.
- learning-rate schedule.
- gradient clipping.
- mixed precision.
- DataLoader worker 수.

## 학습 범위

1단계 I20 baseline을 아래 세 horizon에 대해 학습합니다.
- `stage1_i20_r5`
- `stage1_i20_r20`
- `stage1_i20_r60`

각 horizon은:
- horizon별 label filtering을 사용합니다.
- `1-4`의 horizon별 split metadata를 사용합니다.
- `1-4`의 horizon별 training-only pixel mean/std를 사용합니다.
- `1-5`의 동일한 `StockCNNI20` 구조를 사용합니다.

## Full Paper-style Run

1단계 full reproduction은 horizon마다 5개 독립 모델을 학습합니다.

```text
horizon in [r5, r20, r60]
  run_seed in [42, 43, 44, 45, 46]
    run_seed로 model initialize
    같은 horizon split과 normalization 규칙으로 train
    validation loss 기준 best checkpoint 저장
```

Prediction aggregation:
- seed별 checkpoint로 test set softmax probability를 계산합니다.
- 5개 run의 `prob_up`을 평균합니다.
- 최종 paper-style prediction:
  - `pred_class = 1 if mean_prob_up >= 0.5 else 0`
- individual-run prediction과 averaged prediction을 모두 저장합니다.

중요한 seed 구분:
- `split_seed = 42`는 train/validation split을 결정합니다.
- `run_seed in [42, 43, 44, 45, 46]`는 model initialization과 training
  stochasticity를 결정합니다.

## Local Smoke-test Run

local smoke test는 reproduction result가 아닙니다.

local smoke-test 기본값:
- `run_seeds: [42]`
- tiny subset만 사용
- train/validation batch 수를 작게 제한
- 확인 목적:
  - data loading
  - tensor shape
  - forward/backward pass
  - checkpoint writing
  - metric file writing

Smoke-test output에는 반드시 아래를 남깁니다.

```yaml
run_mode: smoke
report_as_reproduction: false
```

## Loss와 Output

Training loss:
- `torch.nn.CrossEntropyLoss`
- Input: logits `(batch_size, 2)`
- Target: integer labels `(batch_size,)`, 값은 `0` 또는 `1`

Class weighting:
- 기본 사용하지 않습니다.
- 논문이 class weighting을 보고하지 않았고, 현재 horizon positive rate가 대체로 균형에 가깝기 때문입니다.

Softmax:
- `model.forward` 안에서 softmax를 적용하지 않습니다.
- metrics와 prediction file 생성 시에만 `torch.softmax(logits, dim=1)`를 적용합니다.

## Optimizer

Adam을 사용합니다.

```yaml
optimizer:
  name: adam
  learning_rate: 1.0e-5
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.0
```

메모:
- 논문은 Adam과 `1e-5`를 보고합니다.
- Adam betas/eps와 weight decay는 보고하지 않으므로 PyTorch default를 쓰고 기록합니다.
- learning-rate schedule은 논문에 없으므로 기본 사용하지 않습니다.

## Initialization

학습 전 Xavier initialization을 적용합니다.
- `Conv2d`, `Linear` weight에 Xavier initialization 적용.
- `Conv2d`, `Linear` bias가 있으면 zero.
- `BatchNorm2d` weight는 one, bias는 zero.

기본 variant:
- `torch.nn.init.xavier_uniform_`

이유:
- 논문은 Xavier initializer를 보고하지만 uniform/normal 여부는 명시하지 않습니다.
- 따라서 구현에서는 선택한 variant를 기록해야 합니다.

## Epoch와 Early Stopping

Epoch cap:
- `max_epochs: 100`

이유:
- 논문은 정확한 epoch 수를 보고하지 않습니다.
- `100`은 safety cap이고, 실제 종료는 early stopping이 결정하게 둡니다.

Validation:
- 매 epoch 학습 후 validation을 1회 수행합니다.
- mean validation loss를 monitor합니다.

Early stopping:
- `patience: 2`
- `min_delta: 0.0`
- validation loss가 2 epoch 연속 개선되지 않으면 중단합니다.
- validation loss가 개선될 때마다 best checkpoint를 저장합니다.
- test prediction은 best validation-loss checkpoint를 사용합니다.

## DataLoader 동작

Training DataLoader:
- `batch_size: 128`
- `shuffle: true`
- 현재 `run_seed`를 쓰는 seeded generator
- `drop_last: false`

Validation/test DataLoader:
- `batch_size: 128`
- `shuffle: false`
- `drop_last: false`

Memory behavior:
- `1-2`의 lazy/memmap image loading plan을 사용합니다.
- 모든 image를 RAM에 eager-load하지 않습니다.

DataLoader performance knobs:
- `num_workers`, `pin_memory`, Kaggle-specific path/performance 선택은 `1-6K`로 넘깁니다.

## Device와 Determinism

Device selection:
- full run target은 Kaggle CUDA GPU입니다.
- 구현은 `cuda`, `mps`, `cpu`를 지원합니다.
- config 기본값은 `device: auto`입니다.

Run별 seed setup:
- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA, 가능할 때

Determinism 기본값:

```yaml
determinism:
  enabled: true
  cudnn_deterministic: true
  cudnn_benchmark: false
```

주의:
- GPU, driver, PyTorch version, 일부 CUDA kernel에 따라 완전한 bitwise
  reproducibility는 달라질 수 있습니다. 이 정보는 environment metadata에 기록합니다.

Mixed precision:
- 논문에 AMP/mixed precision 보고가 없으므로 기본 비활성화합니다.

Gradient clipping:
- 논문에 보고가 없으므로 기본 비활성화합니다.

## Checkpoint 계획

Checkpoint root:

```text
outputs/checkpoints/{experiment_name}/seed_{run_seed}/
```

예시:

```text
outputs/checkpoints/stage1_i20_r20/seed_42/best.pt
outputs/checkpoints/stage1_i20_r20/seed_42/last.pt
```

Best checkpoint 내용:
- model state dict
- optimizer state dict
- epoch
- best validation loss
- run seed
- split seed
- horizon/target name
- config snapshot
- normalization metadata
- source reference metadata

Save policy:
- validation loss가 개선될 때 `best.pt` 저장.
- training stop 시 `last.pt` 저장.
- test prediction은 `best.pt` 사용.

## Logging과 Metadata

Per-run history:

```text
outputs/metrics/{experiment_name}/seed_{run_seed}/train_history.csv
```

최소 columns:
- `epoch`
- `train_loss`
- `val_loss`
- `train_accuracy`
- `val_accuracy`
- `learning_rate`
- `epoch_seconds`
- `best_so_far`

Per-run metadata:

```text
outputs/metrics/{experiment_name}/seed_{run_seed}/train_metadata.json
```

최소 fields:
- experiment name
- target return name
- run mode: `full` 또는 `smoke`
- run seed
- split seed
- model reference repo/commit
- paper source note
- optimizer settings
- initialization settings
- early stopping settings
- device
- package/environment summary
- start/end timestamp
- best epoch
- best validation loss

Console logging:
- batch-level logging은 기본 `100` training batches마다 수행합니다.
- epoch-level logging은 매 train/validation epoch 뒤에 수행합니다.

## 예정 Config Fields

```yaml
training:
  run_mode: full
  run_seeds: [42, 43, 44, 45, 46]
  batch_size: 128
  max_epochs: 100
  loss: cross_entropy
  class_weight: null
  optimizer:
    name: adam
    learning_rate: 1.0e-5
    betas: [0.9, 0.999]
    eps: 1.0e-8
    weight_decay: 0.0
  initialization:
    name: xavier_uniform
    conv_linear_bias: zero
    batchnorm_weight: one
    batchnorm_bias: zero
  early_stopping:
    monitor: val_loss
    mode: min
    patience: 2
    min_delta: 0.0
    restore_best: true
  dataloader:
    train_shuffle: true
    eval_shuffle: false
    drop_last: false
  device: auto
  determinism:
    enabled: true
    cudnn_deterministic: true
    cudnn_benchmark: false
  mixed_precision: false
  gradient_clipping: null
  log_every_batches: 100
```

## 이후 단계로 넘길 항목

`1-6K`로 넘김:
- Kaggle input/output path.
- Kaggle dataset mount name.
- Kaggle용 `num_workers`, `pin_memory` 기본값.
- notebook runner 구조.
- environment export와 package/version 기록 세부사항.

`1-7`로 넘김:
- 최종 metric list와 prediction CSV schema.
- majority-class baseline 비교.
- 5-run averaged prediction file naming.

`1-8`로 넘김:
- trained checkpoint에서 Grad-CAM 생성.
- 개별 seed model, averaged model output, 대표 seed 중 어느 기준으로 Grad-CAM을 만들지.
