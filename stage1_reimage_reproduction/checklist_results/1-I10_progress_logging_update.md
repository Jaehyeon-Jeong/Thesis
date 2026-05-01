# 1-I10 Progress Logging Update

## English

Date:
- 2026-05-01

Reason:
- Kaggle full/single-horizon runs looked frozen because normalization and long
  epochs did not print enough progress.

Changed files:
- `src/stage1_reimage/data/label_split.py`
- `src/stage1_reimage/runners/stage1_baseline.py`
- `src/stage1_reimage/training/loop.py`
- `docs/progress_logging.md`

What changed:
- Added progress logs for dataset setup, horizon setup, normalization,
  epoch start/end, batch progress, and early stopping.
- Added `flush=True` to progress messages.
- The canonical Kaggle one-cell runner calls scripts with `python -u`.

Local validation:
- `python -m py_compile src/stage1_reimage/data/label_split.py src/stage1_reimage/runners/stage1_baseline.py src/stage1_reimage/training/loop.py`
- Tiny local smoke run confirmed visible progress messages.
- Smoke log: `reports/smoke_tests/1-I10_progress_logging_smoke.log`

Kaggle note:
- The running Kaggle notebook will not automatically get this update.
- Re-upload or refresh the `stage1_reimage_reproduction/` code snapshot before
  using the new progress output.

## 한국어

날짜:
- 2026-05-01

이유:
- Kaggle full/single-horizon run이 normalization과 긴 epoch 구간에서 진행률을
  충분히 출력하지 않아 멈춘 것처럼 보였습니다.

수정 파일:
- `src/stage1_reimage/data/label_split.py`
- `src/stage1_reimage/runners/stage1_baseline.py`
- `src/stage1_reimage/training/loop.py`
- `docs/progress_logging.md`

변경 내용:
- dataset 준비, horizon 준비, normalization, epoch 시작/종료, batch 진행률,
  early stopping message를 출력하도록 했습니다.
- progress message에 `flush=True`를 추가했습니다.
- 표준 Kaggle one-cell runner가 script들을 `python -u`로 호출합니다.

로컬 검증:
- `python -m py_compile src/stage1_reimage/data/label_split.py src/stage1_reimage/runners/stage1_baseline.py src/stage1_reimage/training/loop.py`
- tiny local smoke run에서 진행률 메시지가 출력되는 것을 확인했습니다.
- Smoke log: `reports/smoke_tests/1-I10_progress_logging_smoke.log`

Kaggle 주의:
- 이미 실행 중인 Kaggle Notebook에는 이 수정이 자동 반영되지 않습니다.
- 새 진행률 출력을 쓰려면 `stage1_reimage_reproduction/` code snapshot을 다시
  업로드하거나 새로 복사해야 합니다.
