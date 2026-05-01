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

Recommended execution:
- Run one horizon first: `stage1_i20_r20`.
- Evaluate after the checkpoint exists.
- Generate Grad-CAM after prediction CSV exists.
- Repeat for `stage1_i20_r5` and `stage1_i20_r60`.
- For Kaggle usability, use
  `notebooks/kaggle_stage1_single_horizon_one_cell.md` as the copy-paste cell.

Validation:
- Python compile passed.
- Tiny local smoke run passed.
- Smoke log: `reports/smoke_tests/1-I10_training_metadata_free_smoke.log`

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

권장 실행:
- 먼저 `stage1_i20_r20` horizon 하나만 실행합니다.
- checkpoint가 생긴 뒤 evaluation합니다.
- prediction CSV가 생긴 뒤 Grad-CAM을 만듭니다.
- 이후 `stage1_i20_r5`, `stage1_i20_r60`을 반복합니다.
- Kaggle에서 편하게 실행하려면
  `notebooks/kaggle_stage1_single_horizon_one_cell.md`의 cell을 복붙합니다.

검증:
- Python compile 통과.
- tiny local smoke run 통과.
- Smoke log: `reports/smoke_tests/1-I10_training_metadata_free_smoke.log`
