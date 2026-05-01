# Stage 1 Progress Logging

## English

Purpose:
- Make long Kaggle runs visibly progress instead of looking frozen.

Added progress output:
- Stage start:
  - selected run mode
  - horizon list
  - seed list
  - device
- Dataset setup:
  - shard count
  - total row count
  - metadata row count
- Horizon setup:
  - labeled row count
  - train/validation/test split counts
- Normalization:
  - total train images used
  - processed image count
  - percentage complete
  - final train pixel mean/std
- Training:
  - epoch start
  - train/validation batch counts
  - periodic batch progress with percentage
  - epoch-end loss/accuracy/time
  - early stopping message

Recommended Kaggle interface:
- Use `notebooks/kaggle_stage1_single_horizon_one_cell.md`.
- The cell already calls scripts with `python -u`.

Manual command, if debugging:

```bash
python -u scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_single_seed \
  --horizons stage1_i20_r20 \
  --run-seeds 42
```

Why `python -u`:
- It reduces stdout buffering in notebook environments.
- The code also uses `flush=True` for progress messages.

If the output is too frequent, increase this config value:

```yaml
training:
  log_every_batches: 100
```

If the output is too sparse, reduce it:

```yaml
training:
  log_every_batches: 10
```

## 한국어

목적:
- Kaggle long run이 멈춘 것처럼 보이지 않도록 진행 상황을 출력합니다.

추가한 진행 출력:
- Stage 시작:
  - 선택한 run mode
  - horizon 목록
  - seed 목록
  - device
- Dataset 준비:
  - shard 수
  - 전체 row 수
  - metadata row 수
- Horizon 준비:
  - label 생성 후 row 수
  - train/validation/test split count
- Normalization:
  - 사용할 train image 수
  - 처리된 image 수
  - 진행률 percentage
  - 최종 train pixel mean/std
- Training:
  - epoch 시작
  - train/validation batch 수
  - 주기적 batch 진행률과 percentage
  - epoch 종료 loss/accuracy/time
  - early stopping message

권장 Kaggle interface:
- `notebooks/kaggle_stage1_single_horizon_one_cell.md`를 사용합니다.
- 이 cell은 이미 script들을 `python -u`로 호출합니다.

한 단계씩 debug할 때의 manual command:

```bash
python -u scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_single_seed \
  --horizons stage1_i20_r20 \
  --run-seeds 42
```

`python -u`를 쓰는 이유:
- Notebook 환경에서 stdout buffering을 줄입니다.
- 코드 내부 progress message도 `flush=True`로 출력합니다.

출력이 너무 잦으면 config 값을 키우면 됩니다.

```yaml
training:
  log_every_batches: 100
```

출력이 너무 드물면 줄이면 됩니다.

```yaml
training:
  log_every_batches: 10
```
