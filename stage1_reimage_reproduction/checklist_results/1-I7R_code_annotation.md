# 1-I7R Code Annotation Pass

## English

Status:
- Completed on 2026-05-01.

Purpose:
- Add explanatory comments/docstrings so the Stage 1 code can be read as a
  learning artifact, not only as runnable experiment code.
- Explain what each important function receives, what it returns, what tensor
  or DataFrame shape it handles, and where the value moves next.
- Keep explanatory comments/docstrings primarily in Korean so the project owner
  can review the code line by line while learning the implementation.

Updated files:
- `src/stage1_reimage/config.py`
- `src/stage1_reimage/paths.py`
- `src/stage1_reimage/runtime.py`
- `src/stage1_reimage/seed.py`
- `src/stage1_reimage/data/monthly20.py`
- `src/stage1_reimage/data/label_split.py`
- `src/stage1_reimage/models/stock_cnn.py`
- `src/stage1_reimage/training/loop.py`
- `src/stage1_reimage/runners/stage1_baseline.py`
- `src/stage1_reimage/evaluation/prediction.py`
- `scripts/run_stage1_baseline.py`
- `scripts/evaluate_stage1_predictions.py`
- `scripts/check_scaffold.py`
- `scripts/check_data_loading.py`
- `scripts/check_label_split_normalization.py`
- `scripts/check_model.py`
- `scripts/check_training_loop.py`
- Root `PLAN.md`

Main annotation topics:
- How raw `.dat` bytes become image tensors `(1, 64, 60)`.
- How DataLoader batches become `(batch_size, 1, 64, 60)`.
- How `Ret_5d`, `Ret_20d`, and `Ret_60d` become binary labels.
- How split metadata is used to select train/validation/test rows.
- How train-only pixel normalization avoids leakage.
- How CNN layers transform tensor shapes.
- How logits `(batch_size, 2)` become loss during training.
- How logits become `prob_up` and prediction CSV rows during evaluation.

Validation:
- `python -m compileall src scripts`
- `python scripts/check_scaffold.py --config configs/env_local.yaml`
- `python scripts/check_model.py --config configs/env_local.yaml --batch-size 2`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --max-rows 4`

Result:
- Code behavior is unchanged.
- Local smoke evaluation still writes prediction and metric outputs.
- The root code-writing rule now explicitly requires detailed explanatory
  comments for all code.
- The root code-writing rule now also states that explanatory comments and
  docstrings should be written primarily in Korean.

## 한국어

상태:
- 2026-05-01 완료.

목적:
- Stage 1 코드를 단순히 실행 가능한 실험 코드가 아니라, 읽으면서 배울 수 있는
  코드로 만들기 위해 설명 주석/docstring을 추가했습니다.
- 중요한 함수가 무엇을 입력받고, 무엇을 반환하고, 어떤 tensor/DataFrame shape를
  다루고, 그 값이 다음 어디로 이동하는지 설명했습니다.
- 프로젝트 작성자가 코드 한 줄씩 읽으면서 구현을 따라갈 수 있도록 설명 주석과
  docstring은 기본적으로 한국어로 남기도록 정리했습니다.

수정한 파일:
- `src/stage1_reimage/config.py`
- `src/stage1_reimage/paths.py`
- `src/stage1_reimage/runtime.py`
- `src/stage1_reimage/seed.py`
- `src/stage1_reimage/data/monthly20.py`
- `src/stage1_reimage/data/label_split.py`
- `src/stage1_reimage/models/stock_cnn.py`
- `src/stage1_reimage/training/loop.py`
- `src/stage1_reimage/runners/stage1_baseline.py`
- `src/stage1_reimage/evaluation/prediction.py`
- `scripts/run_stage1_baseline.py`
- `scripts/evaluate_stage1_predictions.py`
- `scripts/check_scaffold.py`
- `scripts/check_data_loading.py`
- `scripts/check_label_split_normalization.py`
- `scripts/check_model.py`
- `scripts/check_training_loop.py`
- Root `PLAN.md`

주요 주석 내용:
- raw `.dat` byte가 image tensor `(1, 64, 60)`가 되는 방식.
- DataLoader batch가 `(batch_size, 1, 64, 60)`가 되는 방식.
- `Ret_5d`, `Ret_20d`, `Ret_60d`가 binary label이 되는 방식.
- split metadata로 train/validation/test row를 고르는 방식.
- train-only pixel normalization이 leakage를 막는 방식.
- CNN layer별 tensor shape 변화.
- logits `(batch_size, 2)`가 training loss로 들어가는 방식.
- logits가 evaluation에서 `prob_up`과 prediction CSV row가 되는 방식.

검증:
- `python -m compileall src scripts`
- `python scripts/check_scaffold.py --config configs/env_local.yaml`
- `python scripts/check_model.py --config configs/env_local.yaml --batch-size 2`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --max-rows 4`

결과:
- 코드 동작은 바뀌지 않았습니다.
- local smoke evaluation은 여전히 prediction/metric output을 정상 작성합니다.
- root code-writing rule에는 모든 코드에 자세한 설명 주석을 남긴다는 규칙을
  명시적으로 추가했습니다.
- root code-writing rule에는 설명 주석과 docstring을 기본적으로 한국어로 작성한다는
  규칙도 추가했습니다.
