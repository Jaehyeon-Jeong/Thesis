# 1-I10 Kaggle Full Single-seed Run

## English

Date:
- 2026-05-01

Status:
- Kaggle execution package prepared.
- Actual full Kaggle execution is still pending because it must run inside a
  Kaggle Notebook with the `monthly_20d` dataset attached.
- The Stage 1 checklist remains open until Kaggle outputs are returned and
  checked.

What was added:
- `scripts/run_stage1_kaggle_single_seed.sh`
- `scripts/check_stage1_single_seed_outputs.py`
- `docs/kaggle_single_seed_runbook.md`

Validation done locally:
- Bash syntax check for `run_stage1_kaggle_single_seed.sh`.
- Python bytecode compilation for `check_stage1_single_seed_outputs.py`.
- Output checker help text generation.

Local validation logs:
- `reports/smoke_tests/1-I10_bash_syntax.log`
- `reports/smoke_tests/1-I10_py_compile.log`
- `reports/smoke_tests/1-I10_output_checker_help.txt`

Kaggle command:

```bash
bash scripts/run_stage1_kaggle_single_seed.sh
```

Expected result:
- Full seed `42` checkpoints for `I20/R5`, `I20/R20`, and `I20/R60`.
- Test prediction CSV and metric JSON for each horizon.
- 2019 Figure 13-style Grad-CAM figure for each horizon.
- `outputs/run_manifests/run_manifest.json`.

Limitation:
- This item cannot be honestly marked complete until the Kaggle run produces
  outputs and `scripts/check_stage1_single_seed_outputs.py` returns `status:
  ok`.

## 한국어

날짜:
- 2026-05-01

상태:
- Kaggle 실행 package를 준비했습니다.
- 실제 full Kaggle 실행은 아직 pending입니다. 이 실행은 `monthly_20d`
  dataset이 attach된 Kaggle Notebook 안에서 돌아야 합니다.
- Kaggle output을 받아서 확인하기 전까지 Stage 1 checklist의 `1-I10`은 open
  상태로 둡니다.

추가한 것:
- `scripts/run_stage1_kaggle_single_seed.sh`
- `scripts/check_stage1_single_seed_outputs.py`
- `docs/kaggle_single_seed_runbook.md`

로컬에서 확인한 것:
- `run_stage1_kaggle_single_seed.sh` Bash syntax check.
- `check_stage1_single_seed_outputs.py` Python bytecode compile.
- Output checker help text generation.

로컬 검증 log:
- `reports/smoke_tests/1-I10_bash_syntax.log`
- `reports/smoke_tests/1-I10_py_compile.log`
- `reports/smoke_tests/1-I10_output_checker_help.txt`

Kaggle 실행 명령:

```bash
bash scripts/run_stage1_kaggle_single_seed.sh
```

예상 결과:
- `I20/R5`, `I20/R20`, `I20/R60` 각각 seed `42` full checkpoint.
- 각 horizon의 test prediction CSV와 metric JSON.
- 각 horizon의 2019년 Figure 13-style Grad-CAM figure.
- `outputs/run_manifests/run_manifest.json`.

제한:
- Kaggle에서 실제 output이 생성되고
  `scripts/check_stage1_single_seed_outputs.py`가 `status: ok`를 반환하기 전에는
  이 항목을 완료 처리하지 않습니다.
