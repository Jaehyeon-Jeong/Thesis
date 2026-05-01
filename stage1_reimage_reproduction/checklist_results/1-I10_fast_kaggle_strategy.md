# 1-I10 Fast Kaggle Strategy Update

## English

Date:
- 2026-05-01

Reason:
- The initial all-in-one Kaggle command was too heavy because it attempted
  multiple horizons, evaluation, and Grad-CAM together.
- The earlier BTC all-in-one example was much smaller and RAM-backed, so it is
  not comparable to the 2.18M-row public stock image shard.

Changed files:
- `src/stage1_reimage/data/label_split.py`
- `src/stage1_reimage/runners/stage1_baseline.py`
- `docs/fast_kaggle_strategy.md`
- `docs/stage1_execution_map.md`
- `workflow_diagram.md`
- `notebooks/kaggle_stage1_single_horizon_one_cell.md`

Performance change:
- Training and validation datasets now avoid metadata collation.
- Evaluation datasets still include metadata for prediction CSV output.
- Additional 2026-05-01 runtime fix:
  - Added RAM pre-load dataset for train/validation images.
  - Added optional mixed precision training.
  - Added optional DataParallel training for Kaggle T4 x2.
  - Added one-cell guard that fails if the Kaggle code snapshot is old.

Recommended execution:
- Run one horizon first: `stage1_i20_r20`.
- Evaluate after the checkpoint exists.
- Generate Grad-CAM after prediction CSV exists.
- Repeat for `stage1_i20_r5` and `stage1_i20_r60`.
- For Kaggle usability, use
  `notebooks/kaggle_stage1_single_horizon_one_cell.md` as the copy-paste cell.
- Fast one-cell defaults:
  - `BATCH_SIZE=1024`
  - `PRELOAD_TRAIN_VAL_IMAGES=True`
  - `MIXED_PRECISION=True`
  - `DATA_PARALLEL=True`
  - `FAST_CUDNN=True`

Expected runtime:
- Lazy memmap path can take 20+ minutes per epoch.
- Fast pre-load path target is about 25-60 minutes for one horizon including
  train, test prediction, and quick Grad-CAM.
- All three Stage 1 horizons are expected to take about 1.5-3.5 hours on Kaggle
  T4 x2 if Kaggle I/O is normal.

Validation:
- Python compile passed.
- Tiny local smoke run passed.
- Smoke log: `reports/smoke_tests/1-I10_training_metadata_free_smoke.log`
- Fast pre-load smoke path passed locally.
- Fast pre-load smoke log:
  `reports/smoke_tests/1-I10_fast_preload_smoke.log`

## 한국어

날짜:
- 2026-05-01

이유:
- 처음 만든 all-in-one Kaggle 명령은 여러 horizon, evaluation, Grad-CAM을 한 번에
  실행해서 너무 무거웠습니다.
- 이전 BTC all-in-one 예시는 dataset이 훨씬 작고 RAM 기반이라, 218만 row stock
  image shard와 직접 비교하기 어렵습니다.

수정 파일:
- `src/stage1_reimage/data/label_split.py`
- `src/stage1_reimage/runners/stage1_baseline.py`
- `docs/fast_kaggle_strategy.md`
- `docs/stage1_execution_map.md`
- `workflow_diagram.md`
- `notebooks/kaggle_stage1_single_horizon_one_cell.md`

성능 변경:
- training/validation dataset은 metadata collation을 하지 않도록 바꿨습니다.
- evaluation dataset은 prediction CSV 생성을 위해 metadata를 그대로 유지합니다.
- 2026-05-01 추가 runtime 수정:
  - train/validation image를 RAM에 pre-load하는 dataset을 추가했습니다.
  - mixed precision training option을 추가했습니다.
  - Kaggle T4 x2용 DataParallel training option을 추가했습니다.
  - Kaggle code snapshot이 오래된 경우 one-cell runner가 바로 실패하도록 guard를
    추가했습니다.

권장 실행:
- 먼저 `stage1_i20_r20` horizon 하나만 실행합니다.
- checkpoint가 생긴 뒤 evaluation합니다.
- prediction CSV가 생긴 뒤 Grad-CAM을 만듭니다.
- 이후 `stage1_i20_r5`, `stage1_i20_r60`을 반복합니다.
- Kaggle에서 편하게 실행하려면
  `notebooks/kaggle_stage1_single_horizon_one_cell.md`의 cell을 복붙합니다.
- fast one-cell 기본값:
  - `BATCH_SIZE=1024`
  - `PRELOAD_TRAIN_VAL_IMAGES=True`
  - `MIXED_PRECISION=True`
  - `DATA_PARALLEL=True`
  - `FAST_CUDNN=True`

예상 시간:
- 기존 lazy memmap path는 epoch 하나가 20분 이상 걸릴 수 있습니다.
- fast pre-load path는 horizon 하나의 train, test prediction, quick Grad-CAM까지
  약 25-60분을 목표로 합니다.
- Kaggle I/O가 정상이라면 Stage 1 horizon 세 개는 T4 x2에서 약 1.5-3.5시간을
  예상합니다.

검증:
- Python compile 통과.
- tiny local smoke run 통과.
- Smoke log: `reports/smoke_tests/1-I10_training_metadata_free_smoke.log`
- fast pre-load smoke path가 local에서 통과했습니다.
- fast pre-load smoke log:
  `reports/smoke_tests/1-I10_fast_preload_smoke.log`
