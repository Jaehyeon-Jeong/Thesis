# Stage 1 Grad-CAM Detail Plan

## English

Status:
- Planning gate `1-8` completed on 2026-04-30.
- This document is a design gate only. No Grad-CAM code has been written yet.

## Purpose

Define the Figure 13-style Grad-CAM output for Stage 1 before implementation.
The goal is to reproduce the interpretation workflow for the trained Re-image
I20 CNN baseline, then extend the same output to all feasible public-data
Stage 1 horizons.

Important distinction:
- Grad-CAM is not a raw feature map.
- It is a class-discriminative heatmap built from a selected convolutional
  activation map and the gradient of a target class score.

## Sources Checked

| Source | Use in this plan |
| --- | --- |
| `../PLAN.md` | Requires Grad-CAM in Stage 1 and requires checking sources before implementation. |
| `../자료조사/Re-image 요약.md` | Maps Re-image interpretation section to pp.41-49 and confirms Figure 13-style Grad-CAM output. |
| `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | Must be visually rechecked before final code comments; Figure 13 is the target layout. |
| User-provided Figure 13 screenshot | Shows `I20R20 Grad-CAM for 20 Images from 2019`; each panel has original images followed by Grad-CAM rows for CNN layers. |
| `../자료조사/Grad-CAM요약.md` | Grad-CAM method source: pp.4-6 for logit target, gradients, channel weights, ReLU heatmap, and bilinear upsampling; pp.12-21 for layer-choice caveats. |
| `../자료조사/Grad-CAM.pdf` | Must be visually rechecked before implementation. |
| `docs/baseline_cnn_implementation_plan.md` | Defines hookable Stage 1 I20 convolution modules. |
| `docs/evaluation_prediction_plan.md` | Defines prediction files used for sample selection. |

PDF extraction note:
- The local environment does not currently have `pdftotext` or `pdfinfo`.
- The page mapping above comes from the local summaries and the user-provided
  Figure 13 image. Before adding source comments to code, visually recheck the
  PDF pages and update this source map if the exact page numbering differs.

## Reproduction Scope

Primary Figure 13-style reproduction:
- Experiment: `stage1_i20_r20`.
- Year: 2019 test samples.
- Samples: 10 images receiving `Up` classification and 10 images receiving
  `Down` classification.
- Figure layout: original image row plus one Grad-CAM row per CNN convolution
  layer.

Extended Stage 1 outputs:
- `stage1_i20_r5`
- `stage1_i20_r20`
- `stage1_i20_r60`

The extended outputs use the same Figure 13-style layout so that horizon-level
differences can be inspected.

## Target Layers

Use the three convolution modules from the GitHub-style I20 baseline:

| Grad-CAM name | Module path | Expected activation shape |
| --- | --- | --- |
| `layer1_conv` | `model.layer1[0]` | `(batch_size, 64, 13, 60)` |
| `layer2_conv` | `model.layer2[0]` | `(batch_size, 128, 5, 60)` |
| `layer3_conv` | `model.layer3[0]` | `(batch_size, 256, 3, 60)` |

Source distinction:
- Re-image Figure 13 shows Grad-CAM rows for CNN layers.
- Grad-CAM original pp.4-6 requires a selected convolutional feature map.
- The exact module names above are an implementation decision tied to
  `lich99/Stock_CNN`-style Stage 1 model naming.

## Target Class

Default target class:
- Use the predicted class for each selected sample.
- For `Up` examples, target class is `1`.
- For `Down` examples, target class is `0`.

Reason:
- Re-image Figure 13 labels the panels as images receiving `Up` or `Down`
  classification.
- Grad-CAM explains a target class score.

Implementation rule:
- Use the pre-softmax logit for the target class, not the softmax probability.

Source:
- Grad-CAM original pp.4-6, as summarized in `Grad-CAM요약.md`, defines the
  target score `y^c` before softmax.

Not used by default:
- Ground-truth class Grad-CAM.
- Counterfactual Grad-CAM.
- Guided Grad-CAM.

These can be added later, but Figure 13-style Stage 1 output uses plain
Grad-CAM for the predicted class.

## Grad-CAM Algorithm

For each selected sample, target class `c`, and target layer:

1. Run a forward pass and store the target layer activation `A`.
2. Select target logit `y^c`.
3. Backpropagate `y^c` and store gradients `d y^c / d A`.
4. Compute channel weights with spatial averaging:

```text
alpha_k^c = mean_{i,j}(d y^c / d A^k_{i,j})
```

5. Compute the class-discriminative heatmap:

```text
L_GradCAM^c = ReLU(sum_k alpha_k^c * A^k)
```

6. Min-max normalize each heatmap to `[0, 1]` for visualization.
7. Bilinearly upsample the heatmap to input image size `(64, 60)`.

Source distinction:
- Steps 1-5 and bilinear upsampling come from Grad-CAM pp.4-6.
- Per-heatmap min-max normalization is an implementation choice for stable
  visualization; the Re-image paper screenshot does not specify the exact
  normalization convention.

Edge case:
- If the heatmap maximum is zero after ReLU, save an all-zero heatmap and log a
  warning in the sample metadata.

## Sample Selection

Input:
- Use prediction output from `1-I7`.

Primary selection for `full_single_seed`:
- File: `outputs/predictions/{experiment_name}/seed_42/test_predictions.csv`.
- Filter: `year == 2019`.
- Up panel: `pred_class == 1`.
- Down panel: `pred_class == 0`.
- Select the 10 highest-confidence samples per class.
  - Up confidence: `prob_up`.
  - Down confidence: `prob_down`.

Primary selection for `full_paper_style`:
- Select samples from
  `outputs/predictions/{experiment_name}/averaged/test_predictions.csv`.
- Use averaged probabilities for confidence ranking.
- Compute Grad-CAM on each available seed checkpoint and average normalized
  heatmaps across seeds for the final ensemble-style heatmap.

Source distinction:
- Re-image Figure 13 says the examples are 20 images from 2019 and groups them
  by predicted `Up`/`Down`.
- The exact sample-selection rule is not reported in the local summary.
- Using high-confidence deterministic samples and averaging heatmaps across
  seed checkpoints are implementation choices for reproducibility.

Correctness:
- Do not require `correct == True` by default.
- The figure explains received classifications, not only correct
  classifications.
- Later diagnostic figures may separately compare correct vs incorrect
  predictions.

## Visualization Layout

For each horizon and run mode, create a Figure 13-style grid:

- Panel A: 10 `Up` predictions.
- Panel B: 10 `Down` predictions.
- Each panel:
  - Row 1: original 20-day image.
  - Row 2: Grad-CAM from `layer1_conv`.
  - Row 3: Grad-CAM from `layer2_conv`.
  - Row 4: Grad-CAM from `layer3_conv`.

Display rules:
- Original image uses the raw rendered grayscale image, not the standardized
  model tensor.
- Grad-CAM rows are standalone heatmaps, matching the Figure 13 style.
- Optional overlays can be saved later, but they are not the primary Stage 1
  reproduction output.
- Use a blue/cyan-style heatmap colormap by default. The exact colormap is an
  implementation choice because the Re-image Figure 13 note does not specify a
  matplotlib colormap.

Interpretation text:
- Brighter heatmap regions correspond to stronger positive contribution to the
  selected target class score.
- This should be described as class-discriminative activation evidence, not as
  causal proof.

## Output Paths

Planned outputs:

```text
outputs/figures/gradcam/{experiment_name}/{run_mode}/figure13_style_2019.png
outputs/figures/gradcam/{experiment_name}/{run_mode}/samples.csv
outputs/figures/gradcam/{experiment_name}/{run_mode}/heatmaps/{sample_id}_{layer_name}.npy
reports/figures/gradcam/{experiment_name}_{run_mode}_figure13_style_2019.png
```

Required `samples.csv` columns:
- `experiment_name`
- `run_mode`
- `year`
- `Date`
- `StockID`
- `local_row`
- `target_return_name`
- `target_return`
- `label`
- `pred_class`
- `target_class_for_gradcam`
- `prob_down`
- `prob_up`
- `confidence`
- `correct`
- `checkpoint_paths`
- `heatmap_aggregation`

## Config Draft

```yaml
gradcam:
  enabled: true
  primary_year: 2019
  primary_experiment: stage1_i20_r20
  extended_experiments:
    - stage1_i20_r5
    - stage1_i20_r20
    - stage1_i20_r60
  samples_per_class: 10
  target_class: predicted
  target_score: pre_softmax_logit
  target_layers:
    layer1_conv: layer1.0
    layer2_conv: layer2.0
    layer3_conv: layer3.0
  sample_selection:
    strategy: highest_confidence_by_predicted_class
    require_correct: false
  heatmap:
    apply_relu: true
    normalization: per_heatmap_minmax
    upsample: bilinear
    output_size: [64, 60]
    colormap: blue_cyan_style
  ensemble:
    selection_source: averaged_predictions
    heatmap_aggregation: mean_normalized_heatmap_across_seeds
```

## What Is Deferred to Implementation

- The hook class for collecting activations and gradients.
- The exact matplotlib colormap name after visual comparison.
- Figure-generation code.
- Local smoke test with one random trained or toy checkpoint.
- Kaggle full-run Grad-CAM after real checkpoints and prediction files exist.

## 한국어

상태:
- 2026-04-30에 planning gate `1-8`을 완료했습니다.
- 이 문서는 설계 gate입니다. 아직 Grad-CAM 코드는 작성하지 않았습니다.

## 목적

구현 전에 Figure 13 스타일 Grad-CAM 산출물을 정확히 정의합니다. 목표는
학습된 Re-image I20 CNN baseline에 대해 원논문 해석 workflow를 재현하고,
현재 public data로 가능한 1단계 horizon 전체로 같은 산출물을 확장하는 것입니다.

중요한 구분:
- Grad-CAM은 raw feature map 자체가 아닙니다.
- 선택한 convolution activation과 target class score gradient로 만든
  class-discriminative heatmap입니다.

## 확인한 근거

| 자료 | 이 계획에서 쓰는 부분 |
| --- | --- |
| `../PLAN.md` | 1단계 Grad-CAM 필수, 구현 전 근거 확인 원칙. |
| `../자료조사/Re-image 요약.md` | Re-image 해석 파트 pp.41-49, Figure 13 스타일 Grad-CAM 산출 확인. |
| `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | 최종 코드 주석 전 눈으로 다시 확인해야 하는 원문 PDF. Figure 13이 목표 layout입니다. |
| 사용자가 제공한 Figure 13 이미지 | `I20R20 Grad-CAM for 20 Images from 2019`; 원본 이미지 row 뒤에 CNN layer별 Grad-CAM row가 붙는 구조. |
| `../자료조사/Grad-CAM요약.md` | Grad-CAM pp.4-6: logit target, gradient, channel weight, ReLU heatmap, bilinear upsampling. pp.12-21: layer 선택 주의. |
| `../자료조사/Grad-CAM.pdf` | 구현 전 방법론을 다시 확인해야 하는 원문 PDF. |
| `docs/baseline_cnn_implementation_plan.md` | Stage 1 I20 CNN의 hook 가능한 convolution module 정의. |
| `docs/evaluation_prediction_plan.md` | sample selection에 사용할 prediction file 정의. |

PDF extraction 제한:
- 현재 로컬에는 `pdftotext`, `pdfinfo`가 없습니다.
- 위 page mapping은 로컬 요약과 사용자가 제공한 Figure 13 이미지를 기준으로
  둡니다. 실제 코드 주석을 넣기 전에는 PDF 페이지를 눈으로 다시 확인하고,
  page numbering이 다르면 source map을 갱신합니다.

## 재현 범위

주요 Figure 13 스타일 재현:
- 실험: `stage1_i20_r20`.
- 연도: 2019 test sample.
- sample: `Up` classification을 받은 이미지 10개, `Down` classification을
  받은 이미지 10개.
- figure layout: 원본 이미지 row + CNN convolution layer별 Grad-CAM row.

확장 1단계 산출:
- `stage1_i20_r5`
- `stage1_i20_r20`
- `stage1_i20_r60`

확장 산출도 같은 Figure 13 스타일 layout을 사용해서 horizon별 차이를 봅니다.

## Target Layer

GitHub식 I20 baseline의 세 convolution module을 사용합니다.

| Grad-CAM 이름 | Module path | 예상 activation shape |
| --- | --- | --- |
| `layer1_conv` | `model.layer1[0]` | `(batch_size, 64, 13, 60)` |
| `layer2_conv` | `model.layer2[0]` | `(batch_size, 128, 5, 60)` |
| `layer3_conv` | `model.layer3[0]` | `(batch_size, 256, 3, 60)` |

근거 구분:
- Re-image Figure 13은 CNN layer별 Grad-CAM row를 보여줍니다.
- Grad-CAM 원전 pp.4-6은 선택한 convolutional feature map을 대상으로 합니다.
- 위 module 이름은 `lich99/Stock_CNN`식 1단계 model naming에 따른 구현상 선택입니다.

## Target Class

기본 target class:
- 선택된 sample의 predicted class를 사용합니다.
- `Up` example은 target class `1`.
- `Down` example은 target class `0`.

이유:
- Re-image Figure 13의 panel 제목이 `Up` 또는 `Down` classification을 받은
  이미지입니다.
- Grad-CAM은 특정 target class score를 설명합니다.

구현 규칙:
- target class의 softmax 이전 logit을 사용합니다. softmax probability를
  gradient target으로 쓰지 않습니다.

근거:
- `Grad-CAM요약.md`가 정리한 Grad-CAM 원전 pp.4-6은 target score `y^c`를
  softmax 이전 score로 둡니다.

기본값에서 사용하지 않는 것:
- ground-truth class Grad-CAM.
- counterfactual Grad-CAM.
- Guided Grad-CAM.

나중에 추가할 수는 있지만, Figure 13 스타일 1단계 산출은 predicted class에
대한 plain Grad-CAM입니다.

## Grad-CAM 알고리즘

선택한 sample, target class `c`, target layer마다:

1. forward pass를 실행하고 target layer activation `A`를 저장합니다.
2. target logit `y^c`를 선택합니다.
3. `y^c`를 backpropagate해서 gradient `d y^c / d A`를 저장합니다.
4. gradient를 spatial averaging해서 channel weight를 계산합니다.

```text
alpha_k^c = mean_{i,j}(d y^c / d A^k_{i,j})
```

5. class-discriminative heatmap을 계산합니다.

```text
L_GradCAM^c = ReLU(sum_k alpha_k^c * A^k)
```

6. 시각화를 위해 heatmap별 min-max normalization을 `[0, 1]`로 적용합니다.
7. bilinear interpolation으로 heatmap을 input image size `(64, 60)`에 맞춥니다.

근거 구분:
- 1-5단계와 bilinear upsampling은 Grad-CAM 원전 pp.4-6 근거입니다.
- heatmap별 min-max normalization은 안정적인 시각화를 위한 구현상 선택입니다.
  Re-image Figure 13 screenshot은 정확한 normalization 방식을 명시하지 않습니다.

예외:
- ReLU 이후 heatmap maximum이 0이면 all-zero heatmap을 저장하고 sample
  metadata에 warning을 남깁니다.

## Sample Selection

입력:
- `1-I7`에서 만든 prediction output을 사용합니다.

`full_single_seed` 기본 선택:
- 파일: `outputs/predictions/{experiment_name}/seed_42/test_predictions.csv`.
- 필터: `year == 2019`.
- Up panel: `pred_class == 1`.
- Down panel: `pred_class == 0`.
- class별 confidence가 높은 sample 10개를 선택합니다.
  - Up confidence: `prob_up`.
  - Down confidence: `prob_down`.

`full_paper_style` 기본 선택:
- `outputs/predictions/{experiment_name}/averaged/test_predictions.csv`에서
  sample을 선택합니다.
- confidence ranking은 averaged probability 기준입니다.
- 최종 ensemble-style heatmap은 각 seed checkpoint에서 Grad-CAM을 계산한 뒤,
  normalized heatmap을 seed 방향으로 평균합니다.

근거 구분:
- Re-image Figure 13은 2019년 이미지 20개를 쓰고, predicted `Up`/`Down`으로
  나누어 보여줍니다.
- 정확한 sample-selection rule은 로컬 요약에 보고되지 않았습니다.
- high-confidence deterministic sample 선택과 seed별 heatmap 평균은
  재현 가능한 산출물을 위한 구현상 선택입니다.

정답 여부:
- 기본값에서는 `correct == True`를 요구하지 않습니다.
- Figure 13은 받은 classification을 설명하는 그림이지, 정답 sample만
  설명하는 그림이라고 보지 않습니다.
- 이후 진단용 figure에서 correct/incorrect를 따로 비교할 수 있습니다.

## Visualization Layout

horizon과 run mode마다 Figure 13 스타일 grid를 만듭니다.

- Panel A: `Up` prediction 10개.
- Panel B: `Down` prediction 10개.
- 각 panel:
  - 1행: 원본 20-day image.
  - 2행: `layer1_conv` Grad-CAM.
  - 3행: `layer2_conv` Grad-CAM.
  - 4행: `layer3_conv` Grad-CAM.

표시 규칙:
- 원본 이미지는 standardized model tensor가 아니라 raw rendered grayscale image를 사용합니다.
- Grad-CAM row는 Figure 13처럼 standalone heatmap으로 저장합니다.
- overlay image는 나중에 추가할 수 있지만 1단계 primary reproduction output은 아닙니다.
- 기본 colormap은 blue/cyan 스타일 heatmap으로 둡니다. Re-image Figure 13
  note가 matplotlib colormap 이름을 명시하지 않으므로 정확한 colormap은
  구현상 선택입니다.

해석 문구:
- 밝은 heatmap 영역은 선택한 target class score에 더 큰 positive contribution을
  한 영역입니다.
- 이를 causal proof가 아니라 class-discriminative activation evidence로
  설명해야 합니다.

## Output Path

계획한 산출물:

```text
outputs/figures/gradcam/{experiment_name}/{run_mode}/figure13_style_2019.png
outputs/figures/gradcam/{experiment_name}/{run_mode}/samples.csv
outputs/figures/gradcam/{experiment_name}/{run_mode}/heatmaps/{sample_id}_{layer_name}.npy
reports/figures/gradcam/{experiment_name}_{run_mode}_figure13_style_2019.png
```

필수 `samples.csv` columns:
- `experiment_name`
- `run_mode`
- `year`
- `Date`
- `StockID`
- `local_row`
- `target_return_name`
- `target_return`
- `label`
- `pred_class`
- `target_class_for_gradcam`
- `prob_down`
- `prob_up`
- `confidence`
- `correct`
- `checkpoint_paths`
- `heatmap_aggregation`

## Config Draft

```yaml
gradcam:
  enabled: true
  primary_year: 2019
  primary_experiment: stage1_i20_r20
  extended_experiments:
    - stage1_i20_r5
    - stage1_i20_r20
    - stage1_i20_r60
  samples_per_class: 10
  target_class: predicted
  target_score: pre_softmax_logit
  target_layers:
    layer1_conv: layer1.0
    layer2_conv: layer2.0
    layer3_conv: layer3.0
  sample_selection:
    strategy: highest_confidence_by_predicted_class
    require_correct: false
  heatmap:
    apply_relu: true
    normalization: per_heatmap_minmax
    upsample: bilinear
    output_size: [64, 60]
    colormap: blue_cyan_style
  ensemble:
    selection_source: averaged_predictions
    heatmap_aggregation: mean_normalized_heatmap_across_seeds
```

## 구현 단계로 넘길 것

- activation/gradient hook class.
- 시각 비교 후 정확한 matplotlib colormap 이름 결정.
- figure generation code.
- 임시 또는 실제 checkpoint 기반 local smoke test.
- 실제 checkpoint와 prediction file 생성 후 Kaggle full-run Grad-CAM.
