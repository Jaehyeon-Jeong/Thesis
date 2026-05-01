# Stage 1 Checklist

## English

Proceed one item at a time. Do not implement the next item until the current
item has been checked and reported.

Progress:

Planning phase:
- [x] 1-0. Stage 1 folder and planning documents
- [x] 1-1. Source and constraint re-check
- [x] 1-2. Data loading detail plan
- [x] 1-3. Label construction detail plan
- [x] 1-4. Split and normalization detail plan
- [x] 1-5. Baseline CNN implementation detail plan
- [x] 1-6. Training loop detail plan
- [x] 1-6K. Kaggle runner and environment config detail plan
- [x] 1-7. Evaluation and prediction-output detail plan
- [x] 1-8. Grad-CAM detail plan
- [x] 1-9. Stage 1 report plan

Implementation phase:
- [x] 1-I0. Implementation readiness review
- [x] 1-I1. Shared code/config scaffold implementation
- [x] 1-I2. Data loading implementation
- [x] 1-I3. Label, split, and normalization implementation
- [x] 1-I4. Baseline CNN model implementation
- [x] 1-I5. Training loop and checkpoint implementation
- [x] 1-I6. Kaggle/local runner implementation
- [x] 1-I7. Evaluation and prediction-output implementation
- [x] 1-I7R. Code annotation/readability pass
- [x] 1-I8. Grad-CAM implementation
- [x] 1-I9. Local smoke test
- [ ] 1-I10. Kaggle full single-seed run
- [ ] 1-I11. Kaggle full paper-style 5-run reproduction
- [ ] 1-I12. Stage 1 report outputs

Important:
- Items `1-0` through `1-9` are planning and design gates.
- Items `1-I0` through `1-I12` are actual implementation and execution gates.
- Do not mark Stage 1 reproduction complete until the implementation phase
  produces code, smoke-test outputs, Kaggle outputs, predictions, metrics, and
  Grad-CAM figures.

## 1-0. Stage 1 Folder and Planning Documents

Purpose:
- Create a separate Stage 1 workspace.
- Write Stage 1 pipeline, checklist, and initial source map.
- Carry forward Stage 0 feasibility limits.

Status:
- Completed.

Outputs:
- `README.md`
- `docs/stage1_pipeline.md`
- `docs/stage1_checklist.md`
- `docs/source_map.md`

## 1-1. Source and Constraint Re-check

Purpose:
- Re-read required sources before coding.
- Confirm that Stage 1 is still limited to public I20 full-spec reproduction.

Must check:
- `../PLAN.md`
- `../stage0_data_check/docs/monthly20_data_check.md`
- `../stage0_data_check/docs/source_reference_check.md`
- `../자료조사/Re-image 요약.md`
- `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
- `../자료조사/Grad-CAM요약.md`
- `../자료조사/Grad-CAM.pdf`

Output:
- Updated `docs/source_map.md`
- Short report to user before implementation

Status:
- Completed on 2026-04-30.

## 1-2. Data Loading Detail Plan

Purpose:
- Decide exact data loader behavior before writing code.

Questions to settle:
- File discovery rule by year
- `.dat` dtype and reshape path
- label matching rule
- row alignment validation
- memory loading vs lazy loading

Output:
- Detailed plan section or config draft

Status:
- Completed on 2026-04-30.

Output produced:
- `docs/data_loading_plan.md`

## 1-3. Label Construction Detail Plan

Purpose:
- Define exactly how `Ret_5d`, `Ret_20d`, and `Ret_60d` become labels.

Questions to settle:
- NaN filtering
- zero return handling
- retained metadata columns
- per-horizon dataset naming

Output:
- Detailed label construction plan

Status:
- Completed on 2026-04-30.

Output produced:
- `docs/label_construction_plan.md`

## 1-4. Split and Normalization Detail Plan

Purpose:
- Lock train/validation/test split and normalization before training.

Questions to settle:
- train/validation split inside 1993-2000
- random seed
- validation ratio
- train-only pixel mean/std calculation

Output:
- Detailed split/normalization plan

Status:
- Completed on 2026-04-30.

Output produced:
- `docs/split_normalization_plan.md`
- `data_inventory/stage1_horizon_counts_by_year.csv`
- `data_inventory/stage1_horizon_counts_by_period.csv`

## 1-5. Baseline CNN Implementation Detail Plan

Purpose:
- Write exact model implementation plan before coding.

Questions to settle:
- Whether to copy GitHub layer values exactly for I20
- How to expose intermediate activations for Grad-CAM
- Naming of model class and config fields

Output:
- Detailed model implementation plan

Status:
- Completed on 2026-04-30.

Output produced:
- `docs/baseline_cnn_implementation_plan.md`

## 1-6. Training Loop Detail Plan

Purpose:
- Define training loop behavior before coding.

Questions to settle:
- epoch cap
- checkpoint rule
- early stopping state
- logging frequency
- device and deterministic settings

Output:
- Detailed training plan

Status:
- Completed on 2026-04-30.

Output produced:
- `docs/training_loop_plan.md`

## 1-6K. Kaggle Runner and Environment Config Detail Plan

Purpose:
- Define how full Stage 1 training/evaluation will run on Kaggle Notebook.
- Keep local execution as a smoke-test path.
- Avoid separate Kaggle-only and local-only codebases.

Questions to settle:
- Kaggle input paths for `monthly_20d`.
- Kaggle output paths for metrics, predictions, checkpoints, and figures.
- GitHub install/clone method inside Kaggle.
- Environment config names: `env_kaggle.yaml`, `env_local.yaml`.
- How to record GitHub commit hash, Kaggle notebook version, dataset version, and seed.
- Whether Colab runner is needed now or deferred.

Output:
- Detailed Kaggle runner plan
- Config file skeletons if confirmed

Status:
- Completed on 2026-04-30.

Output produced:
- `docs/kaggle_runner_plan.md`
- `configs/env_kaggle.yaml`
- `configs/env_local.yaml`
- `notebooks/README.md`

## 1-7. Evaluation and Prediction-output Detail Plan

Purpose:
- Define metrics and saved prediction schema before coding.

Questions to settle:
- metrics list
- prediction CSV schema
- probability calculation from logits
- per-horizon output directory names

Output:
- Detailed evaluation plan

Status:
- Completed on 2026-04-30.

Output produced:
- `docs/evaluation_prediction_plan.md`

## 1-8. Grad-CAM Detail Plan

Purpose:
- Define Figure 13-style Grad-CAM reproduction before coding.

Questions to settle:
- target layer list
- target class selection
- Up/Down sample selection rule
- heatmap normalization and colormap
- output figure layout

Output:
- Detailed Grad-CAM plan

Status:
- Completed on 2026-04-30.

Output produced:
- `docs/gradcam_plan.md`

## 1-9. Stage 1 Report Plan

Purpose:
- Define final Stage 1 reporting tables and limitations.

Questions to settle:
- result table columns
- limitation wording
- comparison to paper tables

Output:
- Stage 1 report outline

Status:
- Completed on 2026-04-30.

Output produced:
- `docs/stage1_report_plan.md`

## 1-I0. Implementation Readiness Review

Purpose:
- Confirm that the planning phase is sufficient to start code implementation.
- Check that no implementation task violates `../PLAN.md`.

Must check:
- `../PLAN.md`
- `docs/stage1_pipeline.md`
- `docs/source_map.md`
- all completed Stage 1 planning docs

Output:
- Short readiness note before coding begins

Status:
- Completed on 2026-04-30.

Output produced:
- `docs/implementation_readiness_review.md`

## 1-I1. Shared Code/Config Scaffold Implementation

Purpose:
- Create the shared package/module structure used by local and Kaggle runs.
- Add config loading and output directory helpers.

Expected implementation targets:
- `src/`
- `configs/`
- `scripts/`
- optional `notebooks/`

Output:
- Importable project scaffold
- Config files for local/Kaggle paths

Status:
- Completed on 2026-04-30.

Output produced:
- `src/stage1_reimage/__init__.py`
- `src/stage1_reimage/config.py`
- `src/stage1_reimage/paths.py`
- `src/stage1_reimage/runtime.py`
- `src/stage1_reimage/seed.py`
- `scripts/check_scaffold.py`
- `docs/shared_code_config_scaffold.md`

Validation:
- `python scripts/check_scaffold.py --config configs/env_local.yaml --create-output-dirs`
- `python -m compileall src scripts/check_scaffold.py`

## 1-I2. Data Loading Implementation

Purpose:
- Implement lazy/memmap loading of `monthly_20d` `.dat` image shards and
  matching `.feather` label shards.

Expected implementation targets:
- data discovery
- row alignment checks
- image tensor conversion to `(batch, 1, 64, 60)`
- metadata preservation

Output:
- Dataset/data module
- Local smoke check for file counts and sample shapes

Status:
- Completed on 2026-04-30.

Output produced:
- `src/stage1_reimage/data/__init__.py`
- `src/stage1_reimage/data/monthly20.py`
- `scripts/check_data_loading.py`
- `docs/data_loading_implementation.md`

Validation:
- `python scripts/check_data_loading.py --config configs/env_local.yaml --sample-indices 0 -1`
- `python -m compileall src scripts/check_data_loading.py`

## 1-I3. Label, Split, and Normalization Implementation

Purpose:
- Implement horizon-specific labels, 1993-2000 train/validation split,
  2001-2019 test split, and train-only pixel normalization.

Expected implementation targets:
- `Ret_5d`, `Ret_20d`, `Ret_60d` label creation
- NaN filtering
- split metadata
- normalization metadata

Output:
- `split_summary.json`
- `normalization.json`
- split/index metadata if needed

Status:
- Completed on 2026-04-30.

Output produced:
- `src/stage1_reimage/data/label_split.py`
- `scripts/check_label_split_normalization.py`
- `docs/label_split_normalization_implementation.md`
- Updated `configs/env_local.yaml`
- Updated `configs/env_kaggle.yaml`

Validation:
- `python scripts/check_label_split_normalization.py --config configs/env_local.yaml --normalization-max-images 2048`
- `python -m compileall src scripts/check_label_split_normalization.py`

## 1-I4. Baseline CNN Model Implementation

Purpose:
- Implement `StockCNNI20` following `lich99/Stock_CNN/models/baseline.py`.

Expected implementation targets:
- `src/models/stock_cnn.py`
- source comments citing paper/GitHub
- shape and parameter-count checks
- Grad-CAM hookable conv layer names

Output:
- Model module
- Local random tensor smoke test

Status:
- Completed on 2026-04-30.

Output produced:
- `src/stage1_reimage/models/__init__.py`
- `src/stage1_reimage/models/stock_cnn.py`
- `scripts/check_model.py`
- `docs/baseline_cnn_model_implementation.md`
- Updated `configs/env_local.yaml`
- Updated `configs/env_kaggle.yaml`

Validation:
- `python scripts/check_model.py --config configs/env_local.yaml --batch-size 2`
- `python -m compileall src scripts/check_model.py`

## 1-I5. Training Loop and Checkpoint Implementation

Purpose:
- Implement training loop, Xavier initialization, Adam, CrossEntropyLoss,
  early stopping, checkpointing, and train history logging.

Expected implementation targets:
- training module
- checkpoint module
- seed/determinism helpers

Output:
- `best.pt`
- `last.pt`
- `train_history.csv`
- `train_metadata.json`

Status:
- Completed on 2026-04-30.

Output produced:
- `src/stage1_reimage/training/__init__.py`
- `src/stage1_reimage/training/loop.py`
- `scripts/check_training_loop.py`
- `docs/training_loop_implementation.md`
- Updated `configs/env_local.yaml`
- Updated `configs/env_kaggle.yaml`

Validation:
- `python scripts/check_training_loop.py --config configs/env_local.yaml --max-epochs 2 --train-samples 8 --val-samples 4 --batch-size 2`
- `python -m compileall src scripts/check_training_loop.py`

## 1-I6. Kaggle/Local Runner Implementation

Purpose:
- Implement runner scripts or notebooks that use the same source code with
  environment-specific configs.

Expected implementation targets:
- local smoke runner
- Kaggle runner or notebook
- run manifest writer

Output:
- runnable local smoke command
- Kaggle runner skeleton
- `run_manifest.json`

Status:
- Completed on 2026-05-01.

Output produced:
- `src/stage1_reimage/runners/__init__.py`
- `src/stage1_reimage/runners/stage1_baseline.py`
- `scripts/run_stage1_baseline.py`
- `notebooks/kaggle_stage1_single_horizon_one_cell.md`
- `docs/kaggle_local_runner_implementation.md`
- Updated `src/stage1_reimage/data/label_split.py`

Validation:
- `python scripts/run_stage1_baseline.py --config configs/env_local.yaml --run-mode smoke --horizons stage1_i20_r20 --max-train-rows 8 --max-val-rows 4 --normalization-max-images 128 --max-epochs 1`
- `python -m compileall src scripts/run_stage1_baseline.py`

## 1-I7. Evaluation and Prediction-output Implementation

Purpose:
- Implement metrics and prediction CSV export for each horizon and seed.

Expected implementation targets:
- classification metrics
- majority-class baseline comparison
- individual seed predictions
- 5-run averaged predictions

Output:
- `predictions.csv`
- `metrics.json`
- tables under `reports/tables/`

Status:
- Completed on 2026-05-01.

Output produced:
- `src/stage1_reimage/evaluation/__init__.py`
- `src/stage1_reimage/evaluation/prediction.py`
- `scripts/evaluate_stage1_predictions.py`
- `docs/evaluation_prediction_implementation.md`
- Updated `configs/env_local.yaml`
- Updated `configs/env_kaggle.yaml`

Validation:
- `python -m compileall src scripts`
- `python scripts/check_scaffold.py --config configs/env_local.yaml`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --max-rows 4`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --split validation --average-seed-predictions 42`

## 1-I7R. Code Annotation/Readability Pass

Purpose:
- Add explanatory comments/docstrings so the user can read the code and
  understand what each function receives, returns, and passes to the next stage.

Expected implementation targets:
- comments for tensor/DataFrame shapes
- comments for data movement between functions
- comments for leakage-sensitive fields
- root `PLAN.md` code-writing rule update

Output:
- More explanatory code comments
- `checklist_results/1-I7R_code_annotation.md`

Status:
- Completed on 2026-05-01.

Validation:
- `python -m compileall src scripts`
- `python scripts/check_scaffold.py --config configs/env_local.yaml`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --max-rows 4`

## 1-I8. Grad-CAM Implementation

Purpose:
- Implement Re-image Figure 13-style Grad-CAM outputs.

Expected implementation targets:
- Grad-CAM module
- layer hooks for `layer1[0]`, `layer2[0]`, `layer3[0]`
- 2019 Up/Down sample selection
- I20/R5, I20/R20, I20/R60 Grad-CAM extension

Output:
- Grad-CAM figures under `outputs/figures/gradcam/`
- report figures under `reports/figures/gradcam/`

Status:
- Completed on 2026-05-01.

Output produced:
- `src/stage1_reimage/interpretability/__init__.py`
- `src/stage1_reimage/interpretability/gradcam.py`
- `scripts/generate_stage1_gradcam.py`
- `docs/gradcam_implementation.md`
- `checklist_results/1-I8_gradcam_implementation.md`
- `reports/figures/gradcam/stage1_i20_r20_seed_42_validation_1993_figure13_style.png`

Validation:
- `python -m compileall src scripts`
- `python scripts/generate_stage1_gradcam.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --year 1993 --samples-per-class 1 --write-report-copy`

## 1-I9. Local Smoke Test

Purpose:
- Verify that code paths work locally on a tiny subset.

Must verify:
- data loading
- label/split/normalization
- model forward/backward
- checkpoint writing
- prediction/metric writing

Output:
- Smoke-test logs and outputs clearly marked as non-reproduction results

Status:
- Completed on 2026-05-01.

Output produced:
- `docs/local_smoke_test.md`
- `checklist_results/1-I9_local_smoke_test.md`
- `reports/smoke_tests/1-I9_*.json`
- `reports/smoke_tests/1-I9_compileall.log`

Validation:
- `python scripts/check_scaffold.py --config configs/env_local.yaml`
- `python scripts/check_data_loading.py --config configs/env_local.yaml --sample-indices 0 -1`
- `python scripts/check_label_split_normalization.py --config configs/env_local.yaml --normalization-max-images 128`
- `python scripts/check_model.py --config configs/env_local.yaml --batch-size 2`
- `python scripts/check_training_loop.py --config configs/env_local.yaml --max-epochs 2 --train-samples 8 --val-samples 4 --batch-size 2`
- `python scripts/run_stage1_baseline.py --config configs/env_local.yaml --run-mode smoke --horizons stage1_i20_r20 --max-train-rows 8 --max-val-rows 4 --normalization-max-images 128 --max-epochs 1`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --max-rows 4`
- `python scripts/generate_stage1_gradcam.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --year 1993 --samples-per-class 1 --write-report-copy`
- `python -m compileall src scripts`

## 1-I10. Kaggle Full Single-seed Run

Purpose:
- Run the first full Kaggle baseline with seed `[42]`.

Output:
- Full-run checkpoint, predictions, metrics, and manifest for one seed

Status:
- Kaggle one-cell single-horizon execution standard prepared on 2026-05-01.
- This item is not marked complete yet because the actual full run must execute
  inside Kaggle with the `monthly_20d` dataset attached.

Prepared output:
- `scripts/check_stage1_single_seed_outputs.py`
- `docs/progress_logging.md`
- `docs/fast_kaggle_strategy.md`
- `docs/stage1_execution_map.md`
- `workflow_diagram.md`
- `notebooks/kaggle_stage1_single_horizon_one_cell.md`
- `checklist_results/1-I10_progress_logging_update.md`
- `checklist_results/1-I10_fast_kaggle_strategy.md`
- `reports/smoke_tests/1-I10_progress_logging_smoke.log`
- `reports/smoke_tests/1-I10_training_metadata_free_smoke.log`

Kaggle command/interface:
- Copy the Python cell from `notebooks/kaggle_stage1_single_horizon_one_cell.md`
  into Kaggle.
- Run one horizon at a time:
  - `stage1_i20_r20`
  - `stage1_i20_r5`
  - `stage1_i20_r60`

Completion condition:
- `scripts/check_stage1_single_seed_outputs.py` returns `status: ok` from the
  Kaggle run output directory after all three horizon outputs exist.

## 1-I11. Kaggle Full Paper-style 5-run Reproduction

Purpose:
- Run the paper-style 5 independent training runs per horizon.

Output:
- Per-seed outputs
- Averaged prediction outputs
- Final Stage 1 reproduction metrics

## 1-I12. Stage 1 Report Outputs

Purpose:
- Assemble final Stage 1 tables, limitations, and figures.

Output:
- report tables
- Grad-CAM figures
- Stage 1 reproduction summary

## 한국어

한 번에 하나씩만 진행합니다. 현재 항목을 확인하고 보고하기 전에는 다음 항목을 구현하지 않습니다.

진행 상태:

계획 단계:
- [x] 1-0. 1단계 폴더와 계획 문서 작성
- [x] 1-1. 근거와 제한사항 재확인
- [x] 1-2. Data loading 세부계획
- [x] 1-3. Label construction 세부계획
- [x] 1-4. Split and normalization 세부계획
- [x] 1-5. Baseline CNN 구현 세부계획
- [x] 1-6. Training loop 세부계획
- [x] 1-6K. Kaggle runner와 환경 config 세부계획
- [x] 1-7. Evaluation and prediction-output 세부계획
- [x] 1-8. Grad-CAM 세부계획
- [x] 1-9. 1단계 보고 계획

구현 단계:
- [x] 1-I0. 구현 시작 전 readiness review
- [x] 1-I1. 공통 code/config scaffold 구현
- [x] 1-I2. Data loading 구현
- [x] 1-I3. Label, split, normalization 구현
- [x] 1-I4. Baseline CNN model 구현
- [x] 1-I5. Training loop와 checkpoint 구현
- [x] 1-I6. Kaggle/local runner 구현
- [x] 1-I7. Evaluation과 prediction-output 구현
- [x] 1-I7R. 코드 주석/가독성 보강
- [x] 1-I8. Grad-CAM 구현
- [x] 1-I9. Local smoke test
- [ ] 1-I10. Kaggle full single-seed run
- [ ] 1-I11. Kaggle full paper-style 5-run reproduction
- [ ] 1-I12. 1단계 report outputs

중요:
- `1-0`부터 `1-9`까지는 planning/design gate입니다.
- `1-I0`부터 `1-I12`까지가 실제 code implementation과 execution gate입니다.
- code, smoke-test output, Kaggle output, predictions, metrics, Grad-CAM figure가
  나오기 전에는 1단계 재현을 완료로 보지 않습니다.

## 1-0. 1단계 폴더와 계획 문서 작성

목적:
- 1단계 전용 작업 공간을 만듭니다.
- 1단계 pipeline, checklist, initial source map을 작성합니다.
- 0단계에서 확인한 가능 범위와 제한사항을 이어받습니다.

상태:
- 완료.

산출물:
- `README.md`
- `docs/stage1_pipeline.md`
- `docs/stage1_checklist.md`
- `docs/source_map.md`

## 1-1. 근거와 제한사항 재확인

목적:
- 코드 작성 전에 필요한 근거를 다시 읽습니다.
- 1단계가 public I20 full-spec reproduction으로 제한된다는 점을 재확인합니다.

반드시 확인:
- `../PLAN.md`
- `../stage0_data_check/docs/monthly20_data_check.md`
- `../stage0_data_check/docs/source_reference_check.md`
- `../자료조사/Re-image 요약.md`
- `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
- `../자료조사/Grad-CAM요약.md`
- `../자료조사/Grad-CAM.pdf`

산출물:
- 업데이트된 `docs/source_map.md`
- 구현 전 사용자에게 짧은 보고

상태:
- 2026-04-30 완료.

## 1-2. Data Loading 세부계획

목적:
- 코드 작성 전에 data loader 동작을 정확히 정합니다.

결정할 것:
- 연도별 파일 탐색 규칙
- `.dat` dtype과 reshape 방식
- label matching rule
- row alignment 검증
- memory loading vs lazy loading

산출물:
- 세부계획 문서 또는 config 초안

상태:
- 2026-04-30 완료.

생성한 산출물:
- `docs/data_loading_plan.md`

## 1-3. Label Construction 세부계획

목적:
- `Ret_5d`, `Ret_20d`, `Ret_60d`를 label로 바꾸는 방식을 정확히 정의합니다.

결정할 것:
- NaN filtering
- zero return 처리
- 보존할 metadata columns
- horizon별 dataset naming

산출물:
- label construction 세부계획

상태:
- 2026-04-30 완료.

생성한 산출물:
- `docs/label_construction_plan.md`

## 1-4. Split and Normalization 세부계획

목적:
- 학습 전에 train/validation/test split과 normalization을 고정합니다.

결정할 것:
- 1993-2000 내부 train/validation split
- random seed
- validation ratio
- train-only pixel mean/std 계산

산출물:
- split/normalization 세부계획

상태:
- 2026-04-30 완료.

생성한 산출물:
- `docs/split_normalization_plan.md`
- `data_inventory/stage1_horizon_counts_by_year.csv`
- `data_inventory/stage1_horizon_counts_by_period.csv`

## 1-5. Baseline CNN 구현 세부계획

목적:
- 코드 작성 전에 model implementation 계획을 정확히 씁니다.

결정할 것:
- I20 layer 값을 GitHub와 완전히 동일하게 둘지
- Grad-CAM용 intermediate activation 노출 방식
- model class 이름과 config field 이름

산출물:
- model implementation 세부계획

상태:
- 2026-04-30 완료.

생성한 산출물:
- `docs/baseline_cnn_implementation_plan.md`

## 1-6. Training Loop 세부계획

목적:
- 코드 작성 전에 training loop 동작을 정의합니다.

결정할 것:
- epoch cap
- checkpoint rule
- early stopping state
- logging frequency
- device와 deterministic settings

산출물:
- training 세부계획

상태:
- 2026-04-30 완료.

생성한 산출물:
- `docs/training_loop_plan.md`

## 1-6K. Kaggle Runner와 환경 Config 세부계획

목적:
- 1단계 full training/evaluation을 Kaggle Notebook에서 어떻게 실행할지 정의합니다.
- 로컬 실행은 smoke-test path로 유지합니다.
- Kaggle 전용 코드와 local 전용 코드를 따로 만들지 않습니다.

결정할 것:
- Kaggle의 `monthly_20d` input path.
- metrics, predictions, checkpoints, figures의 Kaggle output path.
- Kaggle 안에서 GitHub code를 install/clone하는 방식.
- 환경 config 이름: `env_kaggle.yaml`, `env_local.yaml`.
- GitHub commit hash, Kaggle notebook version, dataset version, seed 기록 방식.
- Colab runner를 지금 만들지, 나중으로 미룰지.

산출물:
- Kaggle runner 세부계획
- 확인되면 config skeleton

상태:
- 2026-04-30 완료.

생성한 산출물:
- `docs/kaggle_runner_plan.md`
- `configs/env_kaggle.yaml`
- `configs/env_local.yaml`
- `notebooks/README.md`

## 1-7. Evaluation and Prediction-output 세부계획

목적:
- 코드 작성 전에 metric과 prediction 저장 schema를 정의합니다.

결정할 것:
- metrics list
- prediction CSV schema
- logits에서 probability 계산 방식
- horizon별 output directory 이름

산출물:
- evaluation 세부계획

상태:
- 2026-04-30 완료.

생성한 산출물:
- `docs/evaluation_prediction_plan.md`

## 1-8. Grad-CAM 세부계획

목적:
- 코드 작성 전에 Figure 13 스타일 Grad-CAM 재현 방식을 정의합니다.

결정할 것:
- target layer list
- target class selection
- Up/Down sample selection rule
- heatmap normalization과 colormap
- output figure layout

산출물:
- Grad-CAM 세부계획

상태:
- 2026-04-30 완료.

생성한 산출물:
- `docs/gradcam_plan.md`

## 1-9. 1단계 보고 계획

목적:
- 최종 1단계 보고 table과 limitation 문구를 정의합니다.

결정할 것:
- result table columns
- limitation wording
- paper table과 비교 방식

산출물:
- 1단계 report outline

상태:
- 2026-04-30 완료.

생성한 산출물:
- `docs/stage1_report_plan.md`
