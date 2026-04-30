# Stage 1 Pipeline: Re-image Reproduction

## English

## Scope

Stage 1 reproduces the Re-image stock-image CNN pipeline before any BTC, Linear,
or FiLM extension.

Stage 1 starts from the current local public data:
- Rendered images: `../테스트/Test/img_data/monthly_20d/*_images.dat`
- Labels: `../테스트/Test/img_data/monthly_20d/*_labels_w_delay.feather`
- Confirmed image shape: `(N, 64, 60)`
- Confirmed image spec: 20-day full specification, interpreted as
  `OHLC + 20-day MA + volume`

Feasible Stage 1 experiments:
- `I20/R5`
- `I20/R20`
- `I20/R60`

Out of scope for the first Stage 1 implementation:
- BTC experiments
- Linear adapter
- FiLM
- News/LLM conditioning
- Direct `I5` or `I60` stock reproduction
- A/B/C/D image-spec ablations from the current `.dat` files

## Execution Environment

Default full-run environment:
- Kaggle Notebook.

Reason:
- Stage 1 CNN training on large image shards is heavy for local execution.
- Kaggle is the primary target for full training/evaluation runs.
- Local execution should remain available for small smoke tests.

Code organization rule:
- Do not make two independent codebases for Kaggle and local.
- Keep one shared `src/` implementation.
- Use environment configs and runner notebooks/scripts to adapt paths and runtime.

Planned environment files/runners:
- `configs/env_local.yaml`
- `configs/env_kaggle.yaml`
- `configs/env_colab.yaml` only if needed later
- `notebooks/kaggle_stage1_runner.ipynb`
- `notebooks/colab_stage1_runner.ipynb` only if needed later

Reproducibility records:
- GitHub commit hash.
- Kaggle notebook version or run id.
- Kaggle dataset name/version.
- Config files used for each run.
- Random seed.
- Package/environment information.
- Saved `metrics.json`, `predictions.csv`, checkpoints, and Grad-CAM figures.

## Fixed References

Every implementation task must cite the relevant source in code comments and
documentation.

Core sources:
- Root plan: `../PLAN.md`
- Stage 0 data audit: `../stage0_data_check/docs/monthly20_data_check.md`
- Stage 0 source audit: `../stage0_data_check/docs/source_reference_check.md`
- Re-image local summary: `../자료조사/Re-image 요약.md`
- Re-image PDF: `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
- Grad-CAM local summary: `../자료조사/Grad-CAM요약.md`
- Grad-CAM PDF: `../자료조사/Grad-CAM.pdf`
- GitHub reference: `https://github.com/lich99/Stock_CNN`
- Checked GitHub commit: `415e2acf2a5013afca67e383acd3edc61fced840`

## Pipeline Steps

Stage 1 is split into two gates:
- Planning/design gates: `1-0` through `1-9`.
- Implementation/execution gates: `1-I0` through `1-I12`.

The Re-image pipeline is not considered reproduced when planning is complete.
It is considered reproduced only after implementation, smoke testing, Kaggle
full runs, prediction outputs, metrics, report tables, and Grad-CAM figures are
produced.

### 1-0. Stage 1 Planning Scaffold

Purpose:
- Create a separate Stage 1 folder.
- Write the Stage 1 pipeline, checklist, and source map.
- Carry forward Stage 0 limitations before implementation.

Status:
- Completed by this planning step.

### 1-1. Source and Constraint Re-check

Purpose:
- Re-read the root `PLAN.md`.
- Re-check Stage 0 data feasibility.
- Re-check the paper/GitHub mismatch log before writing code.

Must confirm:
- Current direct experiments are only `I20/R5`, `I20/R20`, and `I20/R60`.
- Current images are full-spec I20 images.
- GitHub I20 model applies `dilation=(2,1)` to all three conv layers.
- Stage 1 will follow GitHub for the I20 model core unless the user decides otherwise.
- Full execution target is Kaggle Notebook, while local execution is only a smoke-test target.

Output:
- Updated note in `docs/source_map.md` if anything changes.

### 1-2. Data Loading Plan

Purpose:
- Load `.dat` image files and matching `.feather` labels.
- Preserve row alignment between image index and label row.

Rules:
- Image tensor convention must be `(batch, channel, height=64, width=60)`.
- Raw `.dat` rows are interpreted as grayscale images.
- Any conversion or normalization must be documented.
- No shuffling before split metadata is attached.

Expected output:
- A data-loading module or notebook section that can verify:
  - year
  - image row count
  - label row count
  - image shape
  - label columns

### 1-3. Label Construction Plan

Purpose:
- Build binary labels for each horizon from future holding-period returns.

Supported targets:
- `Ret_5d` -> `I20/R5`
- `Ret_20d` -> `I20/R20`
- `Ret_60d` -> `I20/R60`

Rules:
- Filter missing target-return rows separately for each horizon.
- Label is `1` if selected future return is greater than zero.
- Label is `0` otherwise.
- The return itself must be preserved in prediction outputs.

Leakage rule:
- Image data ends at date `t`.
- Label uses future return after `t`.
- No future return or future metadata may be used as model input.

### 1-4. Split Plan

Purpose:
- Match the Re-image stock reproduction split as closely as possible.

Status:
- Detail plan completed in `docs/split_normalization_plan.md`.

Default split:
- Train/validation period: 1993-2000
- Test period: 2001-2019

Fixed detail:
- Apply horizon-specific NaN filtering first.
- Use deterministic 70/30 random split inside 1993-2000.
- Use split seed `42` because the paper summary does not report a seed.
- Use non-stratified sample-level split by default because the paper summary
  only says random 70/30 split.

Rules:
- Test years must not leak into train/validation normalization or early stopping.
- Pixel mean/std must be computed from train data only if standardization is used.

### 1-5. Baseline CNN Plan

Purpose:
- Implement the I20 CNN baseline following `lich99/Stock_CNN/models/baseline.py`.

Status:
- Detail plan completed in `docs/baseline_cnn_implementation_plan.md`.

GitHub I20 reference:
- Input reshape: `x.reshape(-1, 1, 64, 60)`
- Layer 1: `Conv2d(1,64,kernel=(5,3),stride=(3,1),dilation=(2,1),padding=(12,1))`
- Layer 2: `Conv2d(64,128,kernel=(5,3),stride=(3,1),dilation=(2,1),padding=(12,1))`
- Layer 3: `Conv2d(128,256,kernel=(5,3),stride=(3,1),dilation=(2,1),padding=(12,1))`
- Each block: Conv -> BN -> LeakyReLU(0.01) -> MaxPool `(2,1)`
- Classifier: Dropout `0.5` -> Linear `46080 -> 2`
- Forward returns logits.

Known mismatch to document in code:
- Paper summary emphasizes first-layer dilation.
- GitHub applies I20 dilation to all three conv layers.
- Stage 1 I20 baseline follows GitHub unless user changes this decision.

Implementation guardrails:
- Planned model file: `src/models/stock_cnn.py`.
- Planned model class: `StockCNNI20`.
- Forward returns logits, not softmax probabilities.
- Expected parameter count: `708,866`.
- Grad-CAM target layers should remain hookable as `layer1[0]`, `layer2[0]`,
  and `layer3[0]`.

### 1-6. Training Plan

Purpose:
- Train one baseline model per horizon.
- Design the full training path for Kaggle Notebook execution.
- Keep a small local smoke-test mode for checking code and tensor shapes.

Status:
- Detail plan completed in `docs/training_loop_plan.md`.

Default settings from paper/source audit:
- Loss: cross-entropy on logits
- Optimizer: Adam
- Learning rate: `1e-5`
- Batch size: `128`
- Dropout: `0.5`
- Early stopping: validation loss not improving for 2 epochs
- Xavier initialization
- 5 independent retrainings and averaged predictions

Fixed implementation details:
- Full run seeds: `[42, 43, 44, 45, 46]`.
- Epoch cap: `100`, because the paper does not report exact epochs.
- Best checkpoint criterion: lowest validation loss.
- Test prediction uses `best.pt`.
- Local smoke-test uses one seed and tiny subsets only.

Deferred:
- Kaggle input/output paths and performance knobs are handled in `1-6K`.

### 1-6K. Kaggle Runner and Environment Config Plan

Purpose:
- Define Kaggle full-run paths, runtime defaults, code snapshot behavior, and
  environment config files.

Status:
- Detail plan completed in `docs/kaggle_runner_plan.md`.

Fixed environment files:
- `configs/env_kaggle.yaml`
- `configs/env_local.yaml`

Kaggle defaults:
- Data root: `/kaggle/input/reimage-monthly-20d/monthly_20d`
- Output root: `/kaggle/working/stage1_reimage_reproduction/outputs`
- Device: `cuda`
- DataLoader workers: `2`
- Mixed precision: `false`

Run modes:
- `smoke`: one seed, tiny subset, not a reproduction result.
- `full_single_seed`: seed `[42]`, first full baseline check.
- `full_paper_style`: seeds `[42, 43, 44, 45, 46]`, paper-style averaged result.

Rules:
- Upload/attach `monthly_20d` as a Kaggle Dataset.
- Prefer code snapshot attach over live `git clone`.
- Save `outputs/run_manifests/run_manifest.json` for every Kaggle run.
- Actual notebook/runner implementation is deferred until coding starts.

### 1-7. Evaluation and Prediction Output Plan

Purpose:
- Save reproducible prediction outputs and metrics for each horizon.

Status:
- Detail plan completed in `docs/evaluation_prediction_plan.md`.
- Implementation completed in `docs/evaluation_prediction_implementation.md`.

Required prediction columns:
- `Date`
- `StockID`
- target return column
- binary label
- predicted class
- probability of Up
- logits or score columns if useful

Core metrics:
- Accuracy
- Majority-class baseline accuracy
- Class balance
- Confusion matrix
- Precision
- Recall
- F1
- AUC, if both classes are present
- Brier score and log loss as probability diagnostics
- Pearson/Spearman prediction-return correlations

Paper-style outputs to prepare later:
- Decile/portfolio-style stock analysis is planned for Stage 1 stock
  cross-sectional evaluation.
- This must be handled carefully because labels are individual stock returns,
  while portfolio returns are constructed after prediction.
- BTC later cannot directly reuse this H-L decile setup because BTC is a single
  asset.

Implemented command:

```bash
python scripts/evaluate_stage1_predictions.py \
  --config configs/env_kaggle.yaml \
  --horizon stage1_i20_r20 \
  --run-seed 42 \
  --split test
```

### 1-8. Grad-CAM Plan

Purpose:
- Reproduce Re-image Figure 13 style interpretability for the trained I20/R20 model.

Status:
- Detail plan completed in `docs/gradcam_plan.md`.

Required references before coding:
- `../자료조사/Grad-CAM요약.md`
- `../자료조사/Grad-CAM.pdf`
- `../자료조사/Re-image 요약.md`
- `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`

Rules:
- Grad-CAM is computed after training.
- Target score is the softmax-before logit for the target class.
- Target class is the predicted class by default.
- Store activations and gradients for the selected convolution module.
- Channel weight is the spatial average of gradients.
- Heatmap is `ReLU(sum_k alpha_k^c A^k)`.
- Upsample heatmap to the input image size with bilinear interpolation.
- This is not a raw feature map.

Figure 13-style output:
- Primary reproduction: `stage1_i20_r20`, 2019 samples.
- Extended outputs: `stage1_i20_r5`, `stage1_i20_r20`, and `stage1_i20_r60`.
- Select 10 Up predictions and 10 Down predictions.
- Save original image row plus `layer1_conv`, `layer2_conv`, and `layer3_conv`
  Grad-CAM rows.
- For `full_paper_style`, select samples from averaged predictions and average
  normalized heatmaps across seed checkpoints.

Source distinction:
- Re-image Figure 13 motivates the layout and 2019 Up/Down grouping.
- Grad-CAM pp.4-6 motivate the logit target, activation gradients, spatial
  averaged channel weights, ReLU heatmap, and bilinear upsampling.
- Exact sample-selection rule, heatmap normalization, colormap, and ensemble
  heatmap averaging are implementation choices documented in
  `docs/gradcam_plan.md`.

### 1-9. Stage 1 Report Plan

Purpose:
- Summarize what was reproduced and what remains limited by available data.

Status:
- Detail plan completed in `docs/stage1_report_plan.md`.

Required report statements:
- Stage 1 is a public-data reproduction of the author-provided 20-day full-spec
  image pipeline.
- It is not a complete reproduction of all Re-image windows/spec ablations.
- `I5`, `I60`, and A/B/C/D ablations require additional data or raw OHLCV regeneration.

Required report outputs:
- `reports/stage1_reproduction_report.md`
- `reports/tables/stage1_dataset_summary.csv`
- `reports/tables/stage1_split_summary.csv`
- `reports/tables/stage1_classification_metrics.csv`
- `reports/tables/stage1_majority_baseline_comparison.csv`
- `reports/tables/stage1_correlation_metrics.csv`
- `reports/tables/stage1_portfolio_metrics.csv`
- `reports/tables/stage1_paper_comparison.csv`
- `reports/tables/stage1_gradcam_samples.csv`
- `reports/tables/stage1_artifact_manifest.csv`

Report rules:
- Compare only like-for-like cells against the Re-image paper.
- Do not report local smoke-test metrics as reproduction results.
- Mark unavailable metrics or non-comparable paper cells explicitly.
- State that Grad-CAM is a post-hoc class-discriminative heatmap, not a raw
  feature map.

### 1-I0 to 1-I12. Implementation and Execution Gates

Purpose:
- Implement and run the Stage 1 Re-image reproduction pipeline after the
  planning gates are confirmed.

Implementation gates:
- `1-I0`: implementation readiness review
- `1-I1`: shared code/config scaffold implementation
- `1-I2`: data loading implementation
- `1-I3`: label, split, and normalization implementation
- `1-I4`: baseline CNN model implementation
- `1-I5`: training loop and checkpoint implementation
- `1-I6`: Kaggle/local runner implementation
- `1-I7`: evaluation and prediction-output implementation
- `1-I8`: Grad-CAM implementation
- `1-I9`: local smoke test
- `1-I10`: Kaggle full single-seed run
- `1-I11`: Kaggle full paper-style 5-run reproduction
- `1-I12`: Stage 1 report outputs

Current implementation status:
- `1-I0` through `1-I9` are completed.
- `1-I9` local smoke test passed through data loading, training, evaluation,
  and Grad-CAM generation on tiny non-reproduction settings.
- The `1-I10` Kaggle single-seed execution wrapper is prepared.
- The next action is to run `bash scripts/run_stage1_kaggle_single_seed.sh`
  inside Kaggle and verify the returned output receipt.
- The readiness decision is limited to the public `monthly_20d` I20 full-spec
  reproduction path. Raw OHLC image-generator work remains a separate gate if
  needed before claiming a paper-wide pipeline.

Rule:
- Code implementation starts only after the relevant planning gates are
  confirmed by the user.

## Required Outputs

Planned outputs under this folder:
- `configs/*.yaml`
- `notebooks/kaggle_stage1_runner.ipynb`
- `outputs/predictions/*`
- `outputs/metrics/*`
- `outputs/checkpoints/*`
- `outputs/figures/sample_images/*`
- `outputs/figures/gradcam/*`
- `reports/tables/*`
- `reports/figures/gradcam/*`

## 한국어

## 범위

1단계는 BTC, Linear, FiLM으로 넘어가기 전에 Re-image stock-image CNN
파이프라인을 재현하는 단계입니다.

1단계는 현재 로컬 public data에서 시작합니다.
- 렌더링 이미지: `../테스트/Test/img_data/monthly_20d/*_images.dat`
- 라벨: `../테스트/Test/img_data/monthly_20d/*_labels_w_delay.feather`
- 확인된 image shape: `(N, 64, 60)`
- 확인된 image spec: 20-day full specification,
  즉 `OHLC + 20-day MA + volume`으로 해석

현재 가능한 1단계 실험:
- `I20/R5`
- `I20/R20`
- `I20/R60`

첫 1단계 구현 범위 밖:
- BTC 실험
- Linear adapter
- FiLM
- News/LLM conditioning
- 직접적인 `I5` 또는 `I60` stock 재현
- 현재 `.dat` 파일에서 A/B/C/D image-spec ablation 분리

1단계는 두 gate로 나눕니다.
- Planning/design gate: `1-0`부터 `1-9`.
- Implementation/execution gate: `1-I0`부터 `1-I12`.

계획이 끝났다고 Re-image pipeline 재현이 완료된 것은 아닙니다. 코드 구현,
smoke test, Kaggle full run, prediction output, metrics, report table,
Grad-CAM figure까지 만들어져야 1단계 재현으로 봅니다.

## 실행 환경

기본 full-run 환경:
- Kaggle Notebook.

이유:
- 큰 image shard로 CNN을 학습하는 1단계는 로컬 실행이 무겁습니다.
- full training/evaluation은 Kaggle을 기본 실행 대상으로 둡니다.
- 로컬 실행은 작은 smoke test 용도로 유지합니다.

코드 구성 원칙:
- Kaggle용 코드와 local용 코드를 따로 만들지 않습니다.
- 하나의 공통 `src/` 구현을 유지합니다.
- 환경별 config와 runner notebook/script로 path와 runtime만 바꿉니다.

계획된 환경 파일/runner:
- `configs/env_local.yaml`
- `configs/env_kaggle.yaml`
- `configs/env_colab.yaml`은 나중에 필요할 때만 추가
- `notebooks/kaggle_stage1_runner.ipynb`
- `notebooks/colab_stage1_runner.ipynb`는 나중에 필요할 때만 추가

재현성 기록:
- GitHub commit hash.
- Kaggle notebook version 또는 run id.
- Kaggle dataset name/version.
- 각 run에 사용한 config.
- random seed.
- package/environment 정보.
- 저장된 `metrics.json`, `predictions.csv`, checkpoints, Grad-CAM figures.

## 고정 참고자료

구현 작업마다 관련 근거를 코드 주석과 문서에 남깁니다.

핵심 자료:
- 루트 계획: `../PLAN.md`
- 0단계 데이터 감사: `../stage0_data_check/docs/monthly20_data_check.md`
- 0단계 근거 감사: `../stage0_data_check/docs/source_reference_check.md`
- Re-image 로컬 요약: `../자료조사/Re-image 요약.md`
- Re-image PDF: `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
- Grad-CAM 로컬 요약: `../자료조사/Grad-CAM요약.md`
- Grad-CAM PDF: `../자료조사/Grad-CAM.pdf`
- GitHub reference: `https://github.com/lich99/Stock_CNN`
- 확인한 GitHub commit: `415e2acf2a5013afca67e383acd3edc61fced840`

## 파이프라인 단계

### 1-0. 1단계 계획 구조 생성

목적:
- 1단계 전용 폴더를 만듭니다.
- 1단계 파이프라인, 체크리스트, source map을 작성합니다.
- 구현 전에 0단계 제한사항을 그대로 이어받습니다.

상태:
- 이번 planning step으로 완료.

### 1-1. 근거와 제한사항 재확인

목적:
- 루트 `PLAN.md`를 다시 읽습니다.
- 0단계 데이터 가능 범위를 다시 확인합니다.
- 코드 작성 전에 논문/GitHub mismatch log를 다시 확인합니다.

반드시 확인할 것:
- 현재 직접 가능한 실험은 `I20/R5`, `I20/R20`, `I20/R60`뿐입니다.
- 현재 이미지는 full-spec I20 image입니다.
- GitHub I20 모델은 `dilation=(2,1)`을 세 conv layer 모두에 적용합니다.
- 사용자가 바꾸지 않는 한 I20 model core는 GitHub를 따릅니다.
- full 실행 대상은 Kaggle Notebook이고, 로컬 실행은 smoke test 대상입니다.

산출물:
- 바뀐 사항이 있으면 `docs/source_map.md`에 업데이트합니다.

### 1-2. Data Loading 계획

목적:
- `.dat` image와 matching `.feather` label을 읽습니다.
- image index와 label row alignment를 보존합니다.

규칙:
- image tensor convention은 `(batch, channel, height=64, width=60)`입니다.
- `.dat` row는 grayscale image로 해석합니다.
- 변환이나 normalization은 반드시 문서화합니다.
- split metadata를 붙이기 전에는 무작위 shuffling을 하지 않습니다.

예상 산출물:
- 다음을 검증할 수 있는 data-loading module 또는 notebook section:
  - year
  - image row count
  - label row count
  - image shape
  - label columns

### 1-3. Label Construction 계획

목적:
- future holding-period return에서 horizon별 binary label을 만듭니다.

지원 target:
- `Ret_5d` -> `I20/R5`
- `Ret_20d` -> `I20/R20`
- `Ret_60d` -> `I20/R60`

규칙:
- horizon별로 target return 결측 row를 따로 제거합니다.
- 선택한 future return이 0보다 크면 label은 `1`입니다.
- 그렇지 않으면 label은 `0`입니다.
- prediction output에는 return 원값도 보존합니다.

Leakage 방지:
- image data는 date `t`까지입니다.
- label은 `t` 이후 future return입니다.
- future return 또는 future metadata는 model input으로 사용하지 않습니다.

### 1-4. Split 계획

목적:
- Re-image stock reproduction split을 최대한 맞춥니다.

상태:
- `docs/split_normalization_plan.md`에 세부계획을 완료했습니다.

기본 split:
- train/validation period: 1993-2000
- test period: 2001-2019

고정한 디테일:
- horizon별 NaN filtering을 먼저 적용합니다.
- 1993-2000 내부에서 deterministic 70/30 random split을 사용합니다.
- 논문 요약에 seed가 없으므로 split seed는 `42`로 명시합니다.
- 논문 요약이 random 70/30만 말하므로 기본값은 non-stratified
  sample-level split입니다.

규칙:
- test year가 train/validation normalization이나 early stopping에 섞이면 안 됩니다.
- standardization을 쓰면 pixel mean/std는 train data에서만 계산합니다.

### 1-5. Baseline CNN 계획

목적:
- `lich99/Stock_CNN/models/baseline.py`를 따르는 I20 CNN baseline을 구현합니다.

상태:
- `docs/baseline_cnn_implementation_plan.md`에 세부계획을 완료했습니다.

GitHub I20 기준:
- Input reshape: `x.reshape(-1, 1, 64, 60)`
- Layer 1: `Conv2d(1,64,kernel=(5,3),stride=(3,1),dilation=(2,1),padding=(12,1))`
- Layer 2: `Conv2d(64,128,kernel=(5,3),stride=(3,1),dilation=(2,1),padding=(12,1))`
- Layer 3: `Conv2d(128,256,kernel=(5,3),stride=(3,1),dilation=(2,1),padding=(12,1))`
- 각 block: Conv -> BN -> LeakyReLU(0.01) -> MaxPool `(2,1)`
- Classifier: Dropout `0.5` -> Linear `46080 -> 2`
- Forward는 logits를 반환합니다.

코드에 남길 mismatch:
- 논문 요약은 first-layer dilation을 강조합니다.
- GitHub는 I20 dilation을 세 conv layer 모두에 적용합니다.
- 사용자가 다르게 정하지 않는 한 1단계 I20 baseline은 GitHub를 따릅니다.

구현 guardrail:
- 예정 model file: `src/models/stock_cnn.py`.
- 예정 model class: `StockCNNI20`.
- `forward`는 softmax probability가 아니라 logits를 반환합니다.
- 기대 parameter count: `708,866`.
- Grad-CAM target layer는 `layer1[0]`, `layer2[0]`, `layer3[0]`로 hook
  가능하게 유지합니다.

### 1-6. Training 계획

목적:
- horizon별 baseline model을 하나씩 학습합니다.
- full training path는 Kaggle Notebook 실행 기준으로 설계합니다.
- 로컬에서는 code와 tensor shape 확인용 small smoke-test mode를 둡니다.

상태:
- `docs/training_loop_plan.md`에 세부계획을 완료했습니다.

논문/근거 감사 기준 기본값:
- Loss: logits에 대한 cross-entropy
- Optimizer: Adam
- Learning rate: `1e-5`
- Batch size: `128`
- Dropout: `0.5`
- Early stopping: validation loss 2 epoch 미개선 시 중단
- Xavier initialization
- 5회 independent retraining 후 averaged prediction

고정한 구현 디테일:
- Full run seeds: `[42, 43, 44, 45, 46]`.
- Epoch cap: 논문에 exact epoch가 없으므로 `100`.
- Best checkpoint 기준: 가장 낮은 validation loss.
- Test prediction은 `best.pt` 사용.
- local smoke-test는 1개 seed와 tiny subset만 사용합니다.

이후로 넘김:
- Kaggle input/output path와 performance knob은 `1-6K`에서 다룹니다.

### 1-6K. Kaggle Runner와 Environment Config 계획

목적:
- Kaggle full-run path, runtime default, code snapshot 방식, environment config
  file을 정의합니다.

상태:
- `docs/kaggle_runner_plan.md`에 세부계획을 완료했습니다.

고정한 environment files:
- `configs/env_kaggle.yaml`
- `configs/env_local.yaml`

Kaggle 기본값:
- Data root: `/kaggle/input/reimage-monthly-20d/monthly_20d`
- Output root: `/kaggle/working/stage1_reimage_reproduction/outputs`
- Device: `cuda`
- DataLoader workers: `2`
- Mixed precision: `false`

Run modes:
- `smoke`: one seed, tiny subset, reproduction result 아님.
- `full_single_seed`: seed `[42]`, 첫 full baseline check.
- `full_paper_style`: seeds `[42, 43, 44, 45, 46]`, 논문식 averaged result.

규칙:
- `monthly_20d`를 Kaggle Dataset으로 upload/attach합니다.
- live `git clone`보다 code snapshot attach를 우선합니다.
- 모든 Kaggle run은 `outputs/run_manifests/run_manifest.json`을 저장합니다.
- 실제 notebook/runner 구현은 coding 시작 시점으로 넘깁니다.

### 1-7. Evaluation and Prediction Output 계획

목적:
- horizon별 prediction output과 metric을 재현 가능하게 저장합니다.

상태:
- `docs/evaluation_prediction_plan.md`에 세부계획을 완료했습니다.
- `docs/evaluation_prediction_implementation.md`에 구현 결과를 기록했습니다.

필수 prediction columns:
- `Date`
- `StockID`
- target return column
- binary label
- predicted class
- probability of Up
- 필요하면 logits 또는 score columns

기본 metrics:
- Accuracy
- Majority-class baseline accuracy
- Class balance
- Confusion matrix
- Precision
- Recall
- F1
- AUC, 양 class가 모두 있을 때
- probability diagnostic으로 Brier score와 log loss
- prediction-return Pearson/Spearman correlation

추후 paper-style output:
- 1단계 stock cross-sectional evaluation으로 decile/portfolio-style 분석을 계획합니다.
- label은 individual stock return이고, portfolio return은 prediction 이후 구성된다는 점을 구분해야 합니다.
- BTC는 단일 자산이므로 이 H-L decile setup을 그대로 재사용하면 안 됩니다.

구현된 command:

```bash
python scripts/evaluate_stage1_predictions.py \
  --config configs/env_kaggle.yaml \
  --horizon stage1_i20_r20 \
  --run-seed 42 \
  --split test
```

### 1-8. Grad-CAM 계획

목적:
- 학습된 I20/R20 모델로 Re-image Figure 13 스타일 해석 그림을 재현합니다.

상태:
- `docs/gradcam_plan.md`에 세부계획을 완료했습니다.

코딩 전 필수 참고:
- `../자료조사/Grad-CAM요약.md`
- `../자료조사/Grad-CAM.pdf`
- `../자료조사/Re-image 요약.md`
- `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`

규칙:
- Grad-CAM은 학습 이후 계산합니다.
- target score는 softmax 이전 target class logit을 사용합니다.
- target class는 기본적으로 predicted class입니다.
- 선택한 convolution module의 activation과 gradient를 저장합니다.
- channel weight는 gradient의 spatial average입니다.
- heatmap은 `ReLU(sum_k alpha_k^c A^k)`입니다.
- heatmap은 bilinear interpolation으로 input image size에 맞춥니다.
- 이것은 raw feature map이 아닙니다.

Figure 13 스타일 산출물:
- primary reproduction은 `stage1_i20_r20`, 2019년 sample입니다.
- 확장 산출은 `stage1_i20_r5`, `stage1_i20_r20`, `stage1_i20_r60`입니다.
- Up 예측 10개와 Down 예측 10개를 선택합니다.
- original image row와 `layer1_conv`, `layer2_conv`, `layer3_conv`
  Grad-CAM row를 함께 저장합니다.
- `full_paper_style`에서는 averaged prediction에서 sample을 고르고,
  seed별 checkpoint의 normalized heatmap을 평균합니다.

근거 구분:
- Re-image Figure 13은 layout과 2019년 Up/Down grouping의 근거입니다.
- Grad-CAM pp.4-6은 logit target, activation gradient, spatial average
  channel weight, ReLU heatmap, bilinear upsampling의 근거입니다.
- 정확한 sample-selection rule, heatmap normalization, colormap, ensemble
  heatmap averaging은 `docs/gradcam_plan.md`에 구현상 선택으로 남겼습니다.

### 1-9. 1단계 보고 계획

목적:
- 무엇을 재현했고, 무엇이 데이터 제한으로 남았는지 정리합니다.

상태:
- `docs/stage1_report_plan.md`에 세부계획을 완료했습니다.

필수 보고 문장:
- 1단계는 저자 공개 20-day full-spec image pipeline에 대한 public-data reproduction입니다.
- 모든 Re-image window/spec ablation을 완전히 재현한 것은 아닙니다.
- `I5`, `I60`, A/B/C/D ablation은 추가 데이터 또는 raw OHLCV 재생성이 필요합니다.

필수 보고 산출물:
- `reports/stage1_reproduction_report.md`
- `reports/tables/stage1_dataset_summary.csv`
- `reports/tables/stage1_split_summary.csv`
- `reports/tables/stage1_classification_metrics.csv`
- `reports/tables/stage1_majority_baseline_comparison.csv`
- `reports/tables/stage1_correlation_metrics.csv`
- `reports/tables/stage1_portfolio_metrics.csv`
- `reports/tables/stage1_paper_comparison.csv`
- `reports/tables/stage1_gradcam_samples.csv`
- `reports/tables/stage1_artifact_manifest.csv`

보고 규칙:
- Re-image paper와 비교할 때는 같은 조건의 cell끼리만 비교합니다.
- local smoke-test metric은 reproduction result로 보고하지 않습니다.
- 사용할 수 없는 metric이나 직접 비교 불가능한 paper cell은 명시적으로 표시합니다.
- Grad-CAM은 raw feature map이 아니라 post-hoc class-discriminative heatmap이라고
  설명합니다.

### 1-I0 to 1-I12. 구현과 실행 Gate

목적:
- planning gate를 확인한 뒤 1단계 Re-image reproduction pipeline을 실제로
  구현하고 실행합니다.

구현 gate:
- `1-I0`: 구현 시작 전 readiness review
- `1-I1`: 공통 code/config scaffold 구현
- `1-I2`: data loading 구현
- `1-I3`: label, split, normalization 구현
- `1-I4`: baseline CNN model 구현
- `1-I5`: training loop와 checkpoint 구현
- `1-I6`: Kaggle/local runner 구현
- `1-I7`: evaluation과 prediction-output 구현
- `1-I8`: Grad-CAM 구현
- `1-I9`: local smoke test
- `1-I10`: Kaggle full single-seed run
- `1-I11`: Kaggle full paper-style 5-run reproduction
- `1-I12`: 1단계 report outputs

현재 구현 상태:
- `1-I0`부터 `1-I9`까지 완료했습니다.
- `1-I9` local smoke test는 작은 non-reproduction 설정에서 data loading,
  training, evaluation, Grad-CAM 생성까지 통과했습니다.
- `1-I10` Kaggle single-seed 실행 wrapper를 준비했습니다.
- 다음 작업은 Kaggle 안에서 `bash scripts/run_stage1_kaggle_single_seed.sh`를
  실행하고 반환된 output receipt를 확인하는 것입니다.
- readiness 판정은 public `monthly_20d` I20 full-spec reproduction 경로에
  한정됩니다. paper-wide pipeline을 주장하려면 raw OHLC image generator 작업은
  별도 gate로 추가해야 합니다.

규칙:
- 관련 planning gate를 사용자와 확인한 뒤에 code implementation을 시작합니다.

## 필수 산출물

이 폴더 아래 계획된 산출물:
- `configs/*.yaml`
- `notebooks/kaggle_stage1_runner.ipynb`
- `outputs/predictions/*`
- `outputs/metrics/*`
- `outputs/checkpoints/*`
- `outputs/figures/sample_images/*`
- `outputs/figures/gradcam/*`
- `reports/tables/*`
- `reports/figures/gradcam/*`
