# Stage 1 Kaggle Runner and Environment Config Detail Plan

## English

Status:
- Stage 1-6K completed as a detail plan.
- No training/evaluation code has been implemented yet.

## Purpose

Define how Stage 1 full training and evaluation will run on Kaggle Notebook
while keeping one shared codebase for local smoke tests and Kaggle full runs.

## Sources Checked

Process source:
- `../PLAN.md`
- `docs/stage1_checklist.md`
- `docs/stage1_pipeline.md`
- `docs/source_map.md`
- `docs/training_loop_plan.md`

Data/source constraints:
- Stage 1 currently uses local `monthly_20d` rendered image shards.
- Direct feasible stock experiments remain `I20/R5`, `I20/R20`, `I20/R60`.
- Full run target is Kaggle Notebook.
- Local execution is only for smoke tests and documentation checks by default.

## Core Decision

Do not maintain separate Kaggle-only and local-only codebases.

Use:
- one shared `src/` implementation
- environment config files for paths/runtime
- Kaggle Notebook as a runner/wrapper

Created environment config skeletons:

```text
configs/env_local.yaml
configs/env_kaggle.yaml
```

Optional later:

```text
configs/env_colab.yaml
```

Colab is deferred because Stage 1 uses author/public stock image shards rather
than Hugging Face/LLM-heavy data.

## Kaggle Data Input Plan

Current Stage 1 data is not a Kaggle public dataset. Therefore, the `monthly_20d`
folder must be uploaded to Kaggle as a private or project dataset and attached
to the notebook.

Recommended Kaggle Dataset name:

```text
reimage-monthly-20d
```

Recommended Kaggle input layout:

```text
/kaggle/input/reimage-monthly-20d/monthly_20d/
  20d_month_has_vb_20_ma_1993_images.dat
  20d_month_has_vb_20_ma_1993_labels_w_delay.feather
  ...
  20d_month_has_vb_20_ma_2019_images.dat
  20d_month_has_vb_20_ma_2019_labels_w_delay.feather
```

Kaggle config value:

```yaml
data:
  monthly20_root: /kaggle/input/reimage-monthly-20d/monthly_20d
```

Important:
- The notebook must check the existence of all 27 `.dat` files and all 27
  `.feather` files before training.
- If the attached Kaggle Dataset has a different folder name, only
  `configs/env_kaggle.yaml` should change.

## Kaggle Code Input Plan

Preferred reproducibility mode:
- Attach a project code snapshot to Kaggle.
- Record the project Git commit or snapshot id in metadata.
- Copy the attached code snapshot into `/kaggle/working` before running.

Fallback development mode:
- Clone the GitHub repository in Kaggle.
- Checkout the exact commit used for the run.
- Record the commit hash in metadata.

Reason:
- Kaggle internet availability can vary by notebook settings.
- A code snapshot is more reproducible than relying on live `git clone`.

Recommended Kaggle code layout:

```text
/kaggle/input/stage1-reimage-code/stage1_reimage_reproduction/
```

Working copy in Kaggle:

```text
/kaggle/working/stage1_reimage_reproduction/
```

The notebook should not write inside read-only `/kaggle/input`.

## Kaggle Output Plan

Kaggle output root:

```text
/kaggle/working/stage1_reimage_reproduction/outputs
```

Required output folders:

```text
outputs/checkpoints/
outputs/metrics/
outputs/predictions/
outputs/figures/sample_images/
outputs/figures/gradcam/
outputs/logs/
outputs/run_manifests/
```

Stage 1-6K only defines paths. Exact metric and prediction schemas are deferred
to `1-7`; Grad-CAM figure generation is deferred to `1-8`.

## Kaggle Full-run Matrix

Full paper-style run:

| Horizon | Experiment name | Target return | Run seeds |
| --- | --- | --- | --- |
| `I20/R5` | `stage1_i20_r5` | `Ret_5d` | `[42, 43, 44, 45, 46]` |
| `I20/R20` | `stage1_i20_r20` | `Ret_20d` | `[42, 43, 44, 45, 46]` |
| `I20/R60` | `stage1_i20_r60` | `Ret_60d` | `[42, 43, 44, 45, 46]` |

Practical run modes:

```yaml
run_mode: smoke
run_seeds: [42]
```

```yaml
run_mode: full_single_seed
run_seeds: [42]
```

```yaml
run_mode: full_paper_style
run_seeds: [42, 43, 44, 45, 46]
```

Only `full_paper_style` should be treated as the paper-style reproduction run.

## Kaggle Runtime Defaults

Device:
- `device: cuda`
- Fail clearly if CUDA is requested but unavailable in full run.

DataLoader:
- `num_workers: 2` as a conservative Kaggle default.
- `pin_memory: true` when CUDA is available.
- `persistent_workers: true` when `num_workers > 0`.

Reason:
- The dataset is large and image loading is memmap-based.
- Worker count is environment-sensitive; `2` is a safe starting point.

Mixed precision:
- `false` by default because the paper does not report AMP.

Internet:
- Not required in preferred snapshot mode.
- If `git clone` mode is used, notebook internet must be enabled and the exact
  commit must be checked out.

## Required Run Manifest

Every Kaggle run must write:

```text
outputs/run_manifests/run_manifest.json
```

Minimum fields:
- run timestamp
- run mode
- project code source: snapshot path or git URL
- project commit hash or snapshot id
- Kaggle notebook title/version/run id, if available
- Kaggle dataset names and versions, if available
- data root
- output root
- config files used
- Python version
- package versions: `torch`, `numpy`, `pandas`, `pyarrow`, `sklearn` if used
- CUDA availability
- GPU name
- split seed
- run seeds
- experiment list
- source references:
  - Re-image paper summary/PDF
  - `lich99/Stock_CNN` commit
  - Grad-CAM paper summary/PDF for later interpretability output

## Notebook Runner Structure

Planned notebook:

```text
notebooks/kaggle_stage1_runner.ipynb
```

Notebook sections:
1. Print environment and GPU information.
2. Copy or locate project code.
3. Load `configs/env_kaggle.yaml`.
4. Validate `monthly_20d` Kaggle input files.
5. Run data audit checks from Stage 1.
6. Build split and normalization metadata.
7. Run smoke check if requested.
8. Train selected horizon/seed matrix.
9. Save checkpoints and train histories.
10. Run evaluation and prediction export after `1-7` is implemented.
11. Run Grad-CAM after `1-8` is implemented.
12. Write `run_manifest.json`.

At this stage, this is a runner plan only. The notebook file itself is deferred
until implementation starts.

## Local Config Role

Local config is for smoke tests only by default.

Local data root:

```text
/Users/jaehyeonjeong/Desktop/논문/테스트/Test/img_data/monthly_20d
```

Local output root:

```text
/Users/jaehyeonjeong/Desktop/논문/stage1_reimage_reproduction/outputs
```

Local default run mode:

```yaml
run_mode: smoke
run_seeds: [42]
```

Local metrics must not be reported as reproduction results unless a full run is
explicitly performed under the same rules as Kaggle.

## Deferred Items

Deferred to implementation gates:
- Actual Kaggle notebook creation.
- Actual training command or runner script.
- Actual data validation command.

Deferred to `1-7`:
- Final metric calculation and prediction CSV schema.
- Aggregated 5-run prediction file naming.

Deferred to `1-8`:
- Grad-CAM figure generation in Kaggle.

## 한국어

상태:
- 1-6K를 Kaggle runner와 environment config 세부계획으로 완료했습니다.
- training/evaluation code는 아직 구현하지 않았습니다.

## 목적

1단계 full training/evaluation을 Kaggle Notebook에서 어떻게 실행할지 정의합니다.
동시에 local smoke test와 Kaggle full run이 하나의 `src/` 코드베이스를 공유하도록
경로와 실행 환경만 config로 분리합니다.

## 확인한 근거

진행 기준:
- `../PLAN.md`
- `docs/stage1_checklist.md`
- `docs/stage1_pipeline.md`
- `docs/source_map.md`
- `docs/training_loop_plan.md`

데이터/source 제한:
- 1단계는 현재 local `monthly_20d` rendered image shard를 사용합니다.
- 직접 가능한 stock experiment는 `I20/R5`, `I20/R20`, `I20/R60`입니다.
- full run target은 Kaggle Notebook입니다.
- local execution은 기본적으로 smoke test와 문서/구조 확인 용도입니다.

## 핵심 결정

Kaggle 전용 코드와 local 전용 코드를 따로 만들지 않습니다.

사용할 구조:
- 하나의 공유 `src/` implementation
- path/runtime만 environment config로 분리
- Kaggle Notebook은 runner/wrapper 역할

생성한 environment config skeleton:

```text
configs/env_local.yaml
configs/env_kaggle.yaml
```

필요하면 나중에:

```text
configs/env_colab.yaml
```

Colab은 1단계에서는 보류합니다. 1단계는 Hugging Face/LLM-heavy data가 아니라
author/public stock image shard를 쓰기 때문입니다.

## Kaggle Data Input 계획

현재 1단계 데이터는 Kaggle public dataset이 아닙니다. 따라서 `monthly_20d`
폴더를 private/project Kaggle Dataset으로 upload하고 notebook에 attach해야 합니다.

권장 Kaggle Dataset 이름:

```text
reimage-monthly-20d
```

권장 Kaggle input layout:

```text
/kaggle/input/reimage-monthly-20d/monthly_20d/
  20d_month_has_vb_20_ma_1993_images.dat
  20d_month_has_vb_20_ma_1993_labels_w_delay.feather
  ...
  20d_month_has_vb_20_ma_2019_images.dat
  20d_month_has_vb_20_ma_2019_labels_w_delay.feather
```

Kaggle config 값:

```yaml
data:
  monthly20_root: /kaggle/input/reimage-monthly-20d/monthly_20d
```

중요:
- notebook은 학습 전에 27개 `.dat`와 27개 `.feather`가 모두 있는지 확인해야 합니다.
- Kaggle Dataset folder 이름이 다르면 `configs/env_kaggle.yaml`만 수정합니다.

## Kaggle Code Input 계획

권장 reproducibility mode:
- 이 project code snapshot을 Kaggle에 upload/attach합니다.
- project Git commit 또는 snapshot id를 metadata에 기록합니다.
- attach된 code snapshot을 `/kaggle/working`으로 복사한 뒤 실행합니다.

Fallback development mode:
- Kaggle에서 GitHub repository를 clone합니다.
- 정확한 commit을 checkout합니다.
- commit hash를 metadata에 기록합니다.

이유:
- Kaggle notebook internet 설정은 달라질 수 있습니다.
- live `git clone`보다 code snapshot attach가 더 재현성이 좋습니다.

권장 Kaggle code layout:

```text
/kaggle/input/stage1-reimage-code/stage1_reimage_reproduction/
```

Kaggle working copy:

```text
/kaggle/working/stage1_reimage_reproduction/
```

read-only인 `/kaggle/input` 내부에는 쓰지 않습니다.

## Kaggle Output 계획

Kaggle output root:

```text
/kaggle/working/stage1_reimage_reproduction/outputs
```

필수 output folders:

```text
outputs/checkpoints/
outputs/metrics/
outputs/predictions/
outputs/figures/sample_images/
outputs/figures/gradcam/
outputs/logs/
outputs/run_manifests/
```

1-6K에서는 path만 정의합니다. 정확한 metric/prediction schema는 `1-7`,
Grad-CAM figure generation은 `1-8`로 넘깁니다.

## Kaggle Full-run Matrix

Full paper-style run:

| Horizon | Experiment name | Target return | Run seeds |
| --- | --- | --- | --- |
| `I20/R5` | `stage1_i20_r5` | `Ret_5d` | `[42, 43, 44, 45, 46]` |
| `I20/R20` | `stage1_i20_r20` | `Ret_20d` | `[42, 43, 44, 45, 46]` |
| `I20/R60` | `stage1_i20_r60` | `Ret_60d` | `[42, 43, 44, 45, 46]` |

실제 실행 모드:

```yaml
run_mode: smoke
run_seeds: [42]
```

```yaml
run_mode: full_single_seed
run_seeds: [42]
```

```yaml
run_mode: full_paper_style
run_seeds: [42, 43, 44, 45, 46]
```

`full_paper_style`만 paper-style reproduction run으로 취급합니다.

## Kaggle Runtime 기본값

Device:
- `device: cuda`
- full run에서 CUDA를 요청했는데 사용할 수 없으면 명확히 실패시킵니다.

DataLoader:
- `num_workers: 2`를 보수적 Kaggle 기본값으로 둡니다.
- CUDA가 있으면 `pin_memory: true`.
- `num_workers > 0`이면 `persistent_workers: true`.

이유:
- dataset이 크고 image loading이 memmap 기반입니다.
- worker 수는 환경에 민감하므로 `2`를 안전한 시작점으로 둡니다.

Mixed precision:
- 논문이 AMP를 보고하지 않았으므로 기본값은 `false`입니다.

Internet:
- 권장 snapshot mode에서는 필요 없습니다.
- `git clone` mode를 쓰면 notebook internet을 켜고 정확한 commit을 checkout해야 합니다.

## 필수 Run Manifest

모든 Kaggle run은 아래 파일을 저장해야 합니다.

```text
outputs/run_manifests/run_manifest.json
```

최소 fields:
- run timestamp
- run mode
- project code source: snapshot path 또는 git URL
- project commit hash 또는 snapshot id
- 가능하면 Kaggle notebook title/version/run id
- 가능하면 Kaggle dataset names and versions
- data root
- output root
- 사용한 config files
- Python version
- package versions: `torch`, `numpy`, `pandas`, `pyarrow`, 필요하면 `sklearn`
- CUDA availability
- GPU name
- split seed
- run seeds
- experiment list
- source references:
  - Re-image paper summary/PDF
  - `lich99/Stock_CNN` commit
  - 추후 Grad-CAM output용 Grad-CAM paper summary/PDF

## Notebook Runner 구조

예정 notebook:

```text
notebooks/kaggle_stage1_runner.ipynb
```

Notebook sections:
1. environment와 GPU 정보 출력
2. project code copy 또는 locate
3. `configs/env_kaggle.yaml` load
4. `monthly_20d` Kaggle input file 검증
5. Stage 1 data audit checks 실행
6. split과 normalization metadata 생성
7. 요청 시 smoke check 실행
8. 선택한 horizon/seed matrix 학습
9. checkpoint와 train history 저장
10. `1-7` 구현 후 evaluation/prediction export 실행
11. `1-8` 구현 후 Grad-CAM 실행
12. `run_manifest.json` 저장

현재 단계에서는 runner plan만 작성합니다. notebook 파일 자체는 구현 시작 때 만듭니다.

## Local Config 역할

Local config는 기본적으로 smoke test 전용입니다.

Local data root:

```text
/Users/jaehyeonjeong/Desktop/논문/테스트/Test/img_data/monthly_20d
```

Local output root:

```text
/Users/jaehyeonjeong/Desktop/논문/stage1_reimage_reproduction/outputs
```

Local default run mode:

```yaml
run_mode: smoke
run_seeds: [42]
```

동일한 full run 규칙으로 명시적으로 돌린 경우가 아니라면 local metric은 reproduction
result로 보고하지 않습니다.

## 이후 단계로 넘길 항목

구현 gate로 넘김:
- 실제 Kaggle notebook 생성.
- 실제 training command 또는 runner script.
- 실제 data validation command.

`1-7`로 넘김:
- 최종 metric 계산과 prediction CSV schema.
- 5-run averaged prediction file naming.

`1-8`로 넘김:
- Kaggle에서 Grad-CAM figure generation.
