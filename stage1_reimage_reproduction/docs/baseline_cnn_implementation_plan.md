# Stage 1 Baseline CNN Implementation Detail Plan

## English

Status:
- Stage 1-5 completed as a detail plan.
- No model code has been implemented yet.

## Purpose

Define the exact I20 baseline CNN implementation before writing code. This
stage fixes the model architecture, tensor shapes, class names, config fields,
and Grad-CAM hook points.

## Sources Checked

Process source:
- `../PLAN.md`
- `docs/stage1_checklist.md`
- `docs/source_map.md`

Paper and method source:
- `../자료조사/Re-image 요약.md`
- `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
- `../자료조사/Grad-CAM요약.md`
- `../자료조사/Grad-CAM.pdf`

Reference implementation:
- Repository: `https://github.com/lich99/Stock_CNN`
- Commit: `415e2acf2a5013afca67e383acd3edc61fced840`
- File: `models/baseline.py`
- Local copy: `../stage0_data_check/references/stock_cnn_baseline_415e2ac.py.txt`

## Scope

Stage 1 direct implementation target:
- `I20/R5`
- `I20/R20`
- `I20/R60`

Image input:
- Public `monthly_20d` rendered `.dat` images.
- Full specification only: `OHLC + 20-day MA + Volume`.
- Tensor convention: `(batch_size, 1, height=64, width=60)`.

Out of scope for this implementation item:
- `I5` and `I60` stock models.
- A/B/C/D image-spec ablation.
- BTC, Linear, FiLM, and News/LLM conditioning.
- Training loop behavior, which remains deferred to `1-6`.

## Core Decision

The Stage 1 I20 baseline will follow `Stock_CNN/models/baseline.py` as closely
as possible.

Important mismatch to document in code:
- The local Re-image summary says the paper emphasizes vertical dilation in the
  first convolution layer.
- The GitHub I20 reference applies `dilation=(2, 1)` and `padding=(12, 1)` to
  all three convolution layers.
- Because the user fixed the rule that model core implementation should follow
  GitHub, Stage 1 will use the GitHub I20 layer values.

## Model Class Plan

Planned file:

```text
src/models/stock_cnn.py
```

Planned class:

```text
StockCNNI20
```

Reason:
- The direct Stage 1 reproduction data is I20 only.
- Naming the class `StockCNNI20` makes the data limitation explicit.
- Later `I5` and `I60` classes can be added only after source/data support is
  confirmed.

## Architecture

The model keeps the reference names `layer1`, `layer2`, and `layer3` so that the
implementation remains close to GitHub and Grad-CAM target layers are easy to
address.

| Component | Exact planned implementation |
| --- | --- |
| Input reshape | `x.reshape(-1, 1, 64, 60)` |
| Layer 1 | `Conv2d(1, 64, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`, `BatchNorm2d(64)`, `LeakyReLU(0.01, inplace=True)`, `MaxPool2d((2,1), stride=(2,1))` |
| Layer 2 | `Conv2d(64, 128, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`, `BatchNorm2d(128)`, `LeakyReLU(0.01, inplace=True)`, `MaxPool2d((2,1), stride=(2,1))` |
| Layer 3 | `Conv2d(128, 256, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`, `BatchNorm2d(256)`, `LeakyReLU(0.01, inplace=True)`, `MaxPool2d((2,1), stride=(2,1))` |
| Classifier | `Dropout(p=0.5)`, `Linear(46080, 2)` |
| Forward output | logits, shape `(batch_size, 2)` |

Softmax:
- `forward` will return logits, matching GitHub.
- Softmax will be applied only in evaluation/prediction-output code.

## Shape Contract

Verified with the GitHub layer values:

| Step | Shape |
| --- | --- |
| Input | `(batch_size, 1, 64, 60)` |
| After `layer1` | `(batch_size, 64, 13, 60)` |
| After `layer2` | `(batch_size, 128, 5, 60)` |
| After `layer3` | `(batch_size, 256, 3, 60)` |
| Flatten | `(batch_size, 46080)` |
| Logits | `(batch_size, 2)` |

Parameter count target:
- `708,866` parameters.
- This matches the Re-image summary's reported I20 parameter count and the
  GitHub I20 architecture.

## Grad-CAM Readiness

The model should not compute Grad-CAM inside `forward`.

Instead:
- Keep named modules `layer1`, `layer2`, and `layer3`.
- Expose target-layer lookup for Grad-CAM without changing the forward
  computation.
- Planned Grad-CAM target names:
  - `layer1_conv`: `model.layer1[0]`
  - `layer2_conv`: `model.layer2[0]`
  - `layer3_conv`: `model.layer3[0]`

Reason:
- Re-image Figure 13 shows Grad-CAM for each CNN layer.
- Grad-CAM should hook convolutional activations and gradients, not raw input
  pixels or fully connected outputs.
- Full Grad-CAM implementation details remain deferred to `1-8`.

## Config Fields

Planned config fields:

```yaml
model:
  name: stock_cnn_i20
  reference_repo: lich99/Stock_CNN
  reference_commit: 415e2acf2a5013afca67e383acd3edc61fced840
  input_channels: 1
  input_height: 64
  input_width: 60
  num_classes: 2
  reshape_in_forward: true
  conv_kernel_size: [5, 3]
  conv_stride: [3, 1]
  conv_dilation: [2, 1]
  conv_padding: [12, 1]
  channels: [64, 128, 256]
  pool_kernel_size: [2, 1]
  pool_stride: [2, 1]
  leaky_relu_negative_slope: 0.01
  dropout: 0.5
  flatten_dim: 46080
  output_type: logits
  gradcam_target_layers:
    - layer1_conv
    - layer2_conv
    - layer3_conv
```

## Source Comment Requirement

The model file must include a compact source note near the class definition:

```python
# Reference implementation:
#   lich99/Stock_CNN/models/baseline.py
#   commit: 415e2acf2a5013afca67e383acd3edc61fced840
#
# Paper source:
#   Jiang, Kelly, and Xiu, Re-Imagining Price Trends,
#   Figure 7, p.18; CNN/training details around pp.12-22.
#
# Tensor convention:
#   images: (batch_size, 1, height=64, width=60)
#
# Note:
#   The paper summary emphasizes first-layer vertical dilation, while the
#   checked GitHub I20 baseline applies dilation=(2, 1) to all three conv
#   layers. Stage 1 follows the GitHub model core by design.
```

## Smoke Checks for Later Implementation

When the code is written, local smoke tests should verify:
- Random input `(2, 1, 64, 60)` returns logits `(2, 2)`.
- Intermediate shapes are exactly:
  - `layer1`: `(2, 64, 13, 60)`
  - `layer2`: `(2, 128, 5, 60)`
  - `layer3`: `(2, 256, 3, 60)`
- Flatten dim is `46080`.
- Parameter count is `708,866`.
- Softmax is not applied inside `forward`.
- Grad-CAM target layer names resolve to convolution modules.

## Deferred Items

Deferred to `1-6`:
- Loss, optimizer, epoch cap, early stopping, device behavior, checkpointing.

Deferred to `1-6K`:
- Kaggle runner and environment path mapping.

Deferred to `1-7`:
- Metrics and prediction CSV schema.

Deferred to `1-8`:
- Grad-CAM algorithm, sample selection, heatmap normalization, and Figure
  13-style grid generation.

## 한국어

상태:
- 1-5를 Baseline CNN 구현 세부계획으로 완료했습니다.
- 모델 코드는 아직 구현하지 않았습니다.

## 목적

코드를 작성하기 전에 I20 baseline CNN의 구조, tensor shape, class 이름,
config field, Grad-CAM hook point를 정확히 고정합니다.

## 확인한 근거

진행 기준:
- `../PLAN.md`
- `docs/stage1_checklist.md`
- `docs/source_map.md`

논문/방법론 근거:
- `../자료조사/Re-image 요약.md`
- `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
- `../자료조사/Grad-CAM요약.md`
- `../자료조사/Grad-CAM.pdf`

기준 구현:
- Repository: `https://github.com/lich99/Stock_CNN`
- Commit: `415e2acf2a5013afca67e383acd3edc61fced840`
- File: `models/baseline.py`
- Local copy: `../stage0_data_check/references/stock_cnn_baseline_415e2ac.py.txt`

## 범위

1단계 직접 구현 대상:
- `I20/R5`
- `I20/R20`
- `I20/R60`

이미지 입력:
- public `monthly_20d` 렌더링 `.dat` 이미지.
- full specification only: `OHLC + 20-day MA + Volume`.
- tensor convention: `(batch_size, 1, height=64, width=60)`.

이번 항목 범위 밖:
- `I5`, `I60` stock model.
- A/B/C/D image-spec ablation.
- BTC, Linear, FiLM, News/LLM conditioning.
- training loop 동작. 이 부분은 `1-6`에서 결정합니다.

## 핵심 결정

1단계 I20 baseline은 `Stock_CNN/models/baseline.py`를 최대한 그대로 따릅니다.

코드에 반드시 남길 mismatch:
- 로컬 Re-image 요약은 paper가 첫 번째 convolution layer의 vertical dilation을
  강조한다고 정리합니다.
- GitHub I20 기준 구현은 세 convolution layer 모두에 `dilation=(2, 1)`과
  `padding=(12, 1)`을 적용합니다.
- 사용자가 모델 핵심 구현은 GitHub를 따르라고 고정했으므로, 1단계 I20은
  GitHub layer 값을 사용합니다.

## Model Class 계획

예정 파일:

```text
src/models/stock_cnn.py
```

예정 class:

```text
StockCNNI20
```

이유:
- 현재 1단계 직접 재현 데이터가 I20뿐입니다.
- `StockCNNI20`로 이름을 고정하면 데이터 제한이 코드에서 명확해집니다.
- `I5`, `I60` class는 source/data support를 다시 확인한 뒤 나중에 추가합니다.

## Architecture

GitHub와 가깝게 유지하고 Grad-CAM target layer를 지정하기 쉽도록
`layer1`, `layer2`, `layer3` 이름을 유지합니다.

| Component | 정확한 예정 구현 |
| --- | --- |
| Input reshape | `x.reshape(-1, 1, 64, 60)` |
| Layer 1 | `Conv2d(1, 64, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`, `BatchNorm2d(64)`, `LeakyReLU(0.01, inplace=True)`, `MaxPool2d((2,1), stride=(2,1))` |
| Layer 2 | `Conv2d(64, 128, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`, `BatchNorm2d(128)`, `LeakyReLU(0.01, inplace=True)`, `MaxPool2d((2,1), stride=(2,1))` |
| Layer 3 | `Conv2d(128, 256, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`, `BatchNorm2d(256)`, `LeakyReLU(0.01, inplace=True)`, `MaxPool2d((2,1), stride=(2,1))` |
| Classifier | `Dropout(p=0.5)`, `Linear(46080, 2)` |
| Forward output | logits, shape `(batch_size, 2)` |

Softmax:
- `forward`는 GitHub처럼 logits를 반환합니다.
- softmax는 evaluation/prediction-output code에서만 적용합니다.

## Shape Contract

GitHub layer 값으로 확인한 shape:

| Step | Shape |
| --- | --- |
| Input | `(batch_size, 1, 64, 60)` |
| After `layer1` | `(batch_size, 64, 13, 60)` |
| After `layer2` | `(batch_size, 128, 5, 60)` |
| After `layer3` | `(batch_size, 256, 3, 60)` |
| Flatten | `(batch_size, 46080)` |
| Logits | `(batch_size, 2)` |

목표 parameter count:
- `708,866`.
- Re-image 요약의 I20 parameter count와 GitHub I20 구조가 일치합니다.

## Grad-CAM 준비

모델 `forward` 안에서 Grad-CAM을 계산하지 않습니다.

대신:
- `layer1`, `layer2`, `layer3` named module을 유지합니다.
- forward 계산을 바꾸지 않는 방식으로 Grad-CAM target lookup을 노출합니다.
- 예정 target 이름:
  - `layer1_conv`: `model.layer1[0]`
  - `layer2_conv`: `model.layer2[0]`
  - `layer3_conv`: `model.layer3[0]`

이유:
- Re-image Figure 13은 CNN 각 layer의 Grad-CAM을 보여줍니다.
- Grad-CAM은 input pixel이나 fully connected output이 아니라 convolutional
  activation과 gradient를 hook해야 합니다.
- 실제 Grad-CAM 구현 세부사항은 `1-8`에서 다룹니다.

## Config Fields

예정 config fields:

```yaml
model:
  name: stock_cnn_i20
  reference_repo: lich99/Stock_CNN
  reference_commit: 415e2acf2a5013afca67e383acd3edc61fced840
  input_channels: 1
  input_height: 64
  input_width: 60
  num_classes: 2
  reshape_in_forward: true
  conv_kernel_size: [5, 3]
  conv_stride: [3, 1]
  conv_dilation: [2, 1]
  conv_padding: [12, 1]
  channels: [64, 128, 256]
  pool_kernel_size: [2, 1]
  pool_stride: [2, 1]
  leaky_relu_negative_slope: 0.01
  dropout: 0.5
  flatten_dim: 46080
  output_type: logits
  gradcam_target_layers:
    - layer1_conv
    - layer2_conv
    - layer3_conv
```

## Source Comment Requirement

모델 파일의 class 정의 근처에는 아래처럼 짧은 source note를 남깁니다.

```python
# Reference implementation:
#   lich99/Stock_CNN/models/baseline.py
#   commit: 415e2acf2a5013afca67e383acd3edc61fced840
#
# Paper source:
#   Jiang, Kelly, and Xiu, Re-Imagining Price Trends,
#   Figure 7, p.18; CNN/training details around pp.12-22.
#
# Tensor convention:
#   images: (batch_size, 1, height=64, width=60)
#
# Note:
#   The paper summary emphasizes first-layer vertical dilation, while the
#   checked GitHub I20 baseline applies dilation=(2, 1) to all three conv
#   layers. Stage 1 follows the GitHub model core by design.
```

## 이후 구현 시 Smoke Check

코드를 작성하면 local smoke test에서 아래를 확인합니다.
- random input `(2, 1, 64, 60)`이 logits `(2, 2)`를 반환.
- intermediate shape:
  - `layer1`: `(2, 64, 13, 60)`
  - `layer2`: `(2, 128, 5, 60)`
  - `layer3`: `(2, 256, 3, 60)`
- flatten dim은 `46080`.
- parameter count는 `708,866`.
- `forward` 안에서 softmax를 적용하지 않음.
- Grad-CAM target layer 이름이 convolution module로 resolve됨.

## 이후 단계로 넘길 항목

`1-6`으로 넘김:
- loss, optimizer, epoch cap, early stopping, device behavior, checkpointing.

`1-6K`로 넘김:
- Kaggle runner와 environment path mapping.

`1-7`로 넘김:
- metrics와 prediction CSV schema.

`1-8`로 넘김:
- Grad-CAM algorithm, sample selection, heatmap normalization, Figure 13-style
  grid generation.
