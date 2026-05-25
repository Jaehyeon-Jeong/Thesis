# 4-7 Grad-CAM plus Context/Gate/Gamma/Beta Export Plan

## English

Status: complete for planning.

Purpose:
- define how Stage 4 explanations are generated after model training;
- keep Grad-CAM consistent with Stage 1/2/3 while adding Stage 4 context
  interpretation;
- save context values and gate/gamma/beta values beside the Grad-CAM samples;
- avoid overclaiming: Grad-CAM and FiLM parameters are post-hoc explanations,
  not causal proof by themselves.

This is an implementation/export plan. It follows the local Grad-CAM summary:
use the pre-softmax target score, selected feature-map activation, gradients
with respect to that activation, spatial-average gradient weights, ReLU, and
bilinear upsampling to the input image size.

## Fixed Explanation Target

Primary target:

```text
target_class = predicted_class
target_score = logits[:, target_class]
```

Reason:
- The core question is why the trained model made its actual decision.
- The predicted-class logit explains the actual decision path.
- The true label is still stored beside the sample so wrong predictions can be
  inspected.

Optional later diagnostic:
- For wrong predictions, generate an extra true-label Grad-CAM. This is not the
  primary Figure-13-style Stage 4 output.

## Sample Selection

Primary final figure:

```text
split = test
10 Predicted Up samples
10 Predicted Down samples
total = 20 samples
```

Selection policy:
1. Read the test prediction CSV for the experiment and seed.
2. Split samples by `predicted_class`.
3. Prefer correct predictions first so the main figure shows successful learned
   decision patterns.
4. If fewer than 10 correct predictions exist for one predicted class, fill the
   remaining slots with highest-confidence predictions from that class and mark
   `correct = false`.
5. Sort selected samples by confidence within each predicted class.
6. Save the exact selected sample list.

Required sample metadata:
- sample index;
- image end date;
- true label;
- predicted class;
- target class used for Grad-CAM;
- `prob_down`, `prob_up`;
- correctness;
- future return;
- split;
- experiment name;
- run seed.

For smoke/debug runs, `samples_per_predicted_class=2` is allowed. Final Stage 4
report figures should use 10 per predicted class.

## Target Feature Maps by Model Track

The explanation layer must reflect the model state that actually reaches the
classifier.

| Track | Main Grad-CAM target layers | Context-conditioned? | Notes |
| --- | --- | --- | --- |
| 4-A concat | standard CNN block outputs/layer conv maps | No | Context only enters after flatten; save context metadata beside the image explanation |
| 4-B gating | layer1-3 standard outputs, layer4 post-gate feature map | Yes, layer4 only | Save final-layer gate vector |
| 4-C gamma-only FiLM | post-gamma feature map after BatchNorm/gamma in every block | Yes | Save gamma vectors for all blocks |
| 4-D full FiLM | post-gamma/beta feature map after BatchNorm/gamma/beta in every block | Yes | Save gamma and beta vectors for all blocks |

For 4-C/4-D, the primary heatmap uses the conditioned feature:

```text
F_i_conditioned = gamma_i * BN_i_output              # gamma-only
F_i_conditioned = gamma_i * BN_i_output + beta_i     # full FiLM
```

This is the correct Stage 4 explanation point because it shows the feature map
after market context has modulated it. Optional pre-FiLM maps may be saved as
diagnostics, but they should not replace the primary figure.

## Figure Layout

Primary Figure-13-style Stage 4 figure:

```text
columns = selected samples
rows    = original image + layer1 + layer2 + layer3 + layer4 Grad-CAM
panels  = Predicted Down and Predicted Up
```

Each column title should include:
- date;
- predicted class: `Down=0` or `Up=1`;
- true label;
- `prob_up`;
- correctness marker.

The figure itself should stay readable. Detailed context and gamma/beta values
belong in companion CSV/JSON files, not as dense text inside the image.

Required copies:
- one figure under `outputs/stage4/...`;
- one report-copy figure under `reports/figures/stage4_gradcam/...`.

## Context Export

For every selected Grad-CAM sample, write one row with:
- raw context values;
- transformed context values;
- normalized context values;
- missing flags;
- F&G source date and source age;
- scaler file reference;
- context embedding summary:
  `mean`, `std`, `min`, `max`, `q05`, `q50`, `q95`.

This links Grad-CAM to the market state:

```text
sample/date -> chart image -> prediction -> context vector -> heatmap
```

## Gate/Gamma/Beta Export

Write two levels of modulation export.

### 1. Selected-Sample Full Vectors

For selected Grad-CAM samples only, write channel-level long-format tables.

4-B gating:

```text
sample_id, date, layer, channel, gate
```

4-C gamma-only:

```text
sample_id, date, layer, channel, gamma, delta_gamma
```

4-D full FiLM:

```text
sample_id, date, layer, channel, gamma, delta_gamma, beta
```

Layer names:

```text
layer1, layer2, layer3, layer4
```

Channel counts:

```text
layer1 = 64
layer2 = 128
layer3 = 256
layer4 = 512
```

### 2. Full-Split Summary Statistics

For the whole evaluated split, write compact per-sample and per-channel summary
tables. Do not default to a huge all-sample, all-channel CSV unless explicitly
needed.

Per-sample summary:

```text
sample_id, date, predicted_class, true_label, correct,
layer, value_type, mean, std, min, q05, q50, q95, max,
top_positive_channels, top_negative_channels
```

Per-channel summary:

```text
layer, channel, value_type,
mean, std, min, q05, q50, q95, max,
mean_when_pred_up, mean_when_pred_down,
mean_when_correct, mean_when_wrong
```

This gives interpretable aggregate behavior without making the default output
unnecessarily large.

## Recommended Output Paths

```text
outputs/stage4/figures/gradcam/<experiment>/seed_<seed>/<split>/
    figure13_style_context_gradcam_<split>.png
    samples.csv
    heatmaps/
    gradcam_summary.json

outputs/stage4/explanations/<experiment>/seed_<seed>/<split>/
    selected_context_values.csv
    selected_modulation_values_long.csv
    selected_modulation_summary.csv
    split_modulation_sample_summary.csv
    split_modulation_channel_summary.csv
    context_modulation_correlations.csv
    explanation_manifest.json

reports/figures/stage4_gradcam/
    <experiment>_seed_<seed>_<split>_figure13_style_context_gradcam.png
```

For 4-A concat:
- `selected_modulation_values_long.csv` is not applicable;
- still write `selected_context_values.csv` and `explanation_manifest.json`.

## Context-Modulation Analysis

For 4-B/4-C/4-D, compute lightweight diagnostic correlations on the evaluated
split:

```text
context feature vs per-layer modulation mean/std
context feature vs predicted probability
context feature vs correctness
```

Use these as descriptive diagnostics only:
- correlation does not prove causality;
- channel identities can shift across seeds;
- final claims should use seed-stable patterns where possible.

## Guardrails

Implementation must fail or warn if:
- selected sample count is below the requested count per predicted class;
- Grad-CAM target class is not recorded;
- heatmap is all zeros or non-finite;
- selected Grad-CAM samples do not have matching context rows;
- 4-B/4-C/4-D explanations are missing gate/gamma/beta exports;
- a FiLM model exports pre-FiLM heatmaps as the primary heatmap without
  explicitly marking them as pre-FiLM diagnostics.

## 4-7 Decision

Proceed to 4-8 and implementation planning with:

```text
primary_target = predicted_class_logit
final_samples = 10 predicted Up + 10 predicted Down from test
smoke_samples = 2 predicted Up + 2 predicted Down

4-A export:
    Grad-CAM + context values

4-B export:
    Grad-CAM + context values + final-layer gate values

4-C export:
    Grad-CAM on post-gamma feature maps + context values + gamma values

4-D export:
    Grad-CAM on post-gamma/beta feature maps + context values + gamma/beta values
```

## 한국어

상태: 계획 단계 완료.

목적:
- Stage 4 모델 학습 후 explanation을 어떻게 생성할지 정의합니다.
- Stage 1/2/3의 Grad-CAM 원칙과 일관성을 유지하면서 Stage 4 context 해석력을
  추가합니다.
- Grad-CAM sample 옆에 context 값과 gate/gamma/beta 값을 같이 저장합니다.
- 과잉 해석을 피합니다. Grad-CAM과 FiLM parameter는 post-hoc explanation이지,
  그 자체로 인과 증명은 아닙니다.

이 문서는 구현/export 계획입니다. 로컬 Grad-CAM 요약을 따릅니다: softmax 이전
target score, 선택 feature-map activation, 해당 activation에 대한 gradient,
gradient의 spatial average, ReLU, input image size로 bilinear upsampling.

## 고정 explanation target

Primary target:

```text
target_class = predicted_class
target_score = logits[:, target_class]
```

이유:
- 핵심 질문은 학습된 모델이 실제로 왜 이 결정을 했는가입니다.
- predicted-class logit을 사용해야 실제 decision path를 설명합니다.
- true label은 sample metadata에 같이 저장해서 오답을 따로 볼 수 있게 합니다.

Optional later diagnostic:
- 오답 sample에 대해 true-label Grad-CAM을 추가로 만들 수 있지만, main
  Figure-13-style Stage 4 output은 아닙니다.

## Sample 선택

Primary final figure:

```text
split = test
Predicted Up 10개
Predicted Down 10개
총 20개 sample
```

선택 정책:
1. 해당 experiment/seed의 test prediction CSV를 읽습니다.
2. `predicted_class` 기준으로 sample을 나눕니다.
3. main figure는 학습된 성공 pattern을 보기 위해 correct prediction을 우선합니다.
4. 특정 predicted class에서 correct prediction이 10개보다 적으면 confidence가 높은
   prediction으로 채우고 `correct = false`를 표시합니다.
5. 각 predicted class 안에서 confidence 기준으로 정렬합니다.
6. 선택된 sample list를 저장합니다.

필수 sample metadata:
- sample index;
- image end date;
- true label;
- predicted class;
- Grad-CAM에 사용한 target class;
- `prob_down`, `prob_up`;
- correct 여부;
- future return;
- split;
- experiment name;
- run seed.

Smoke/debug run에서는 `samples_per_predicted_class=2`를 허용합니다. 최종 Stage 4
보고 figure는 predicted class별 10개를 사용합니다.

## 모델 track별 target feature map

Explanation layer는 classifier로 실제 전달되는 model state를 반영해야 합니다.

| Track | Main Grad-CAM target layers | Context-conditioned? | Notes |
| --- | --- | --- | --- |
| 4-A concat | standard CNN block outputs/layer conv maps | No | context는 flatten 뒤에만 들어가므로 image branch explanation + context metadata |
| 4-B gating | layer1-3 standard outputs, layer4 post-gate feature map | Yes, layer4 only | final-layer gate vector 저장 |
| 4-C gamma-only FiLM | 모든 block의 BatchNorm/gamma 이후 post-gamma feature map | Yes | 모든 block의 gamma vector 저장 |
| 4-D full FiLM | 모든 block의 BatchNorm/gamma/beta 이후 post-gamma/beta feature map | Yes | 모든 block의 gamma/beta vector 저장 |

4-C/4-D에서 primary heatmap은 context가 적용된 feature를 사용합니다.

```text
F_i_conditioned = gamma_i * BN_i_output              # gamma-only
F_i_conditioned = gamma_i * BN_i_output + beta_i     # full FiLM
```

이 지점이 Stage 4의 올바른 explanation point입니다. 시장 context가 시각 feature를
조절한 뒤의 feature map을 보기 때문입니다. Pre-FiLM map은 diagnostic으로 저장할 수
있지만 primary figure를 대체하지 않습니다.

## Figure layout

Primary Figure-13-style Stage 4 figure:

```text
columns = selected samples
rows    = original image + layer1 + layer2 + layer3 + layer4 Grad-CAM
panels  = Predicted Down and Predicted Up
```

각 column title에는 다음을 포함합니다.
- date;
- predicted class: `Down=0` 또는 `Up=1`;
- true label;
- `prob_up`;
- correct marker.

Figure 자체는 읽기 쉽게 유지합니다. 자세한 context와 gamma/beta 값은 이미지 안에
빽빽하게 쓰지 않고 companion CSV/JSON에 저장합니다.

필수 copy:
- `outputs/stage4/...` 아래 figure 1개;
- `reports/figures/stage4_gradcam/...` 아래 report-copy figure 1개.

## Context export

선택된 모든 Grad-CAM sample에 대해 row 하나를 저장합니다.
- raw context values;
- transformed context values;
- normalized context values;
- missing flags;
- F&G source date와 source age;
- scaler file reference;
- context embedding summary:
  `mean`, `std`, `min`, `max`, `q05`, `q50`, `q95`.

이 저장물은 Grad-CAM을 시장 상태와 연결합니다.

```text
sample/date -> chart image -> prediction -> context vector -> heatmap
```

## Gate/Gamma/Beta export

Modulation export는 두 수준으로 저장합니다.

### 1. 선택 sample full vector

Grad-CAM에 선택된 sample에 대해서만 channel-level long-format table을 저장합니다.

4-B gating:

```text
sample_id, date, layer, channel, gate
```

4-C gamma-only:

```text
sample_id, date, layer, channel, gamma, delta_gamma
```

4-D full FiLM:

```text
sample_id, date, layer, channel, gamma, delta_gamma, beta
```

Layer names:

```text
layer1, layer2, layer3, layer4
```

Channel counts:

```text
layer1 = 64
layer2 = 128
layer3 = 256
layer4 = 512
```

### 2. 전체 split summary statistics

평가 split 전체에 대해서는 compact per-sample/per-channel summary table을 저장합니다.
필요하다고 명시하지 않는 한 all-sample, all-channel 거대 CSV를 기본으로 만들지
않습니다.

Per-sample summary:

```text
sample_id, date, predicted_class, true_label, correct,
layer, value_type, mean, std, min, q05, q50, q95, max,
top_positive_channels, top_negative_channels
```

Per-channel summary:

```text
layer, channel, value_type,
mean, std, min, q05, q50, q95, max,
mean_when_pred_up, mean_when_pred_down,
mean_when_correct, mean_when_wrong
```

이렇게 하면 기본 output을 과하게 키우지 않으면서 aggregate behavior를 해석할 수 있습니다.

## 권장 output path

```text
outputs/stage4/figures/gradcam/<experiment>/seed_<seed>/<split>/
    figure13_style_context_gradcam_<split>.png
    samples.csv
    heatmaps/
    gradcam_summary.json

outputs/stage4/explanations/<experiment>/seed_<seed>/<split>/
    selected_context_values.csv
    selected_modulation_values_long.csv
    selected_modulation_summary.csv
    split_modulation_sample_summary.csv
    split_modulation_channel_summary.csv
    context_modulation_correlations.csv
    explanation_manifest.json

reports/figures/stage4_gradcam/
    <experiment>_seed_<seed>_<split>_figure13_style_context_gradcam.png
```

4-A concat:
- `selected_modulation_values_long.csv`는 해당 없음;
- 그래도 `selected_context_values.csv`와 `explanation_manifest.json`은 저장합니다.

## Context-modulation 분석

4-B/4-C/4-D에서는 evaluated split에 대해 가벼운 diagnostic correlation을 계산합니다.

```text
context feature vs per-layer modulation mean/std
context feature vs predicted probability
context feature vs correctness
```

이 값은 descriptive diagnostic으로만 사용합니다.
- correlation은 causality를 증명하지 않습니다.
- channel identity는 seed마다 달라질 수 있습니다.
- 최종 주장은 가능하면 seed-stable pattern을 기준으로 합니다.

## Guardrails

구현은 다음 경우 실패 또는 warning을 내야 합니다.
- selected sample count가 requested count per predicted class보다 적음;
- Grad-CAM target class가 기록되지 않음;
- heatmap이 all-zero 또는 non-finite;
- 선택된 Grad-CAM sample과 matching context row가 없음;
- 4-B/4-C/4-D explanations에 gate/gamma/beta export가 없음;
- FiLM model이 pre-FiLM heatmap을 primary heatmap으로 export하면서 pre-FiLM
  diagnostic이라고 표시하지 않음.

## 4-7 결정

다음 결정으로 4-8과 구현 계획으로 넘어갑니다.

```text
primary_target = predicted_class_logit
final_samples = test에서 predicted Up 10개 + predicted Down 10개
smoke_samples = predicted Up 2개 + predicted Down 2개

4-A export:
    Grad-CAM + context values

4-B export:
    Grad-CAM + context values + final-layer gate values

4-C export:
    post-gamma feature map 기준 Grad-CAM + context values + gamma values

4-D export:
    post-gamma/beta feature map 기준 Grad-CAM + context values + gamma/beta values
```
