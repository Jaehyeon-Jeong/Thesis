# Stage 2 BTC Baseline CNN Adaptation Plan

## English

Status: planning complete for checklist 2-5. Implementation happens later in
`2-I5`.

Purpose:
- Apply the Stage 1 Re-image/Stock_CNN-style CNN core to BTC images.
- Keep the research design fixed: Stage 2 is BTC asset-class extension, not a
  new model design.
- Avoid silently forcing all BTC windows through the I20-only Stage 1 model.

Source basis:
- Re-image summary: `자료조사/Re-image 요약.md`
  - line 36: baseline image sizes `32x15`, `64x60`, `96x180`.
  - line 47: I5/I20/I60 CNN depth, channels, flatten dimensions, and parameter
    counts.
  - line 49: cross-entropy, Adam, batch size `128`, dropout `0.5`, Xavier
    initialization, validation-loss early stopping.
- Stage 1 model implementation:
  - `stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py`
- GitHub reference:
  - `lich99/Stock_CNN`, `models/baseline.py`, commit
    `415e2acf2a5013afca67e383acd3edc61fced840`.

Important source limitation:
- The checked GitHub implementation is I20-specific. It reshapes every input to
  `(batch, 1, 64, 60)` and uses `Linear(46080, 2)`.
- Therefore I20 can reuse the exact Stage 1/GitHub-style core.
- I5 and I60 must be implemented as separate Stage-1/Stock_CNN-style variants
  using the paper architecture targets. They must not reuse the I20 reshape or
  I20 classifier.

## Model Variants

| Image window | Model name | Input shape | Blocks | Channels | Flatten dim | Expected params |
| --- | --- | --- | ---: | --- | ---: | ---: |
| I5 | `stock_cnn_i5` | `(batch, 1, 32, 15)` | 2 | `64 -> 128` | 15,360 | 155,138 |
| I20 | `stock_cnn_i20` | `(batch, 1, 64, 60)` | 3 | `64 -> 128 -> 256` | 46,080 | 708,866 |
| I60 | `stock_cnn_i60` | `(batch, 1, 96, 180)` | 4 | `64 -> 128 -> 256 -> 512` | 184,320 | 2,952,962 |

CSV artifact:
- `stage2_btc_extension/reports/tables/stage2_baseline_cnn_architecture_plan.csv`

## Fixed CNN Block

Each block keeps the Stage 1/Re-image order:

```text
Conv2d -> BatchNorm2d -> LeakyReLU(0.01) -> MaxPool2d(2, 1)
```

Shared fixed values:
- input channels: `1`
- output classes: `2`
- convolution kernel: `(5, 3)`
- pooling: `(2, 1)`
- dropout before classifier: `0.5`
- output: raw logits `(batch, 2)`
- Up class index: `1`
- softmax is applied only in evaluation, not inside `forward()`.

I20 exact implementation:
- Use the checked GitHub-style Stage 1 core:
  - three conv blocks;
  - `stride=(3, 1)`;
  - `dilation=(2, 1)` in all three conv layers;
  - `padding=(12, 1)` in all three conv layers;
  - `Linear(46080, 2)`.

I5/I60 implementation rule:
- Implement window-specific variants that match the paper-reported block count,
  channel sequence, flatten dimension, and parameter count.
- Before implementation, re-open the local paper/PDF page mapping for Figure 7
  and the CNN section.
- If stride/padding/dilation details needed to hit the reported shapes conflict
  with the Stage 1/GitHub I20 pattern, record the mismatch in `source_map.md`
  and keep the paper-reported shape/parameter target as the guardrail.

## Experiment Mapping

The model is selected by image window, not by return horizon:

| Experiment group | Model |
| --- | --- |
| I5/R5, I5/R20, I5/R60 | `stock_cnn_i5` |
| I20/R5, I20/R20, I20/R60 | `stock_cnn_i20` |
| I60/R5, I60/R20, I60/R60 | `stock_cnn_i60` |

The four image specifications do not change the model architecture because MA
and volume are drawn into the same one-channel grayscale image:
- `ohlc`
- `ohlc_vb`
- `ohlc_ma`
- `ohlc_ma_vb`

Therefore Stage 2 baseline has:

```text
3 image windows x 3 return horizons x 4 image specs = 36 baseline runs
```

## Implementation Interface

Implementation target:
- `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py`

Required API:
- `build_stock_cnn_from_config(config) -> nn.Module`
- `StockCNNConfig`
- `StockCNNI5`
- `StockCNNI20`
- `StockCNNI60`
- `count_parameters(model)`
- `forward_with_shapes(x)`
- `gradcam_target_layers()`

Reason for this interface:
- `forward()` stays close to the GitHub model and returns logits only.
- `forward_with_shapes()` lets us verify tensor shapes in local smoke tests.
- `gradcam_target_layers()` is needed by Stage 2 Grad-CAM.
- Later Stage 3 Linear and Stage 4 FiLM need clean access to the CNN feature
  extractor and classifier boundary.

Design constraint:
- Add helper methods only for inspection and later extension.
- Do not change the baseline logits produced by the CNN forward path.

## Training Defaults

Stage 2 baseline training keeps Stage 1 defaults unless a later checklist item
records a reason to change them:
- loss: cross-entropy
- optimizer: Adam
- learning rate: `1e-5`
- batch size: `128`
- initialization: Xavier for Conv/Linear, BatchNorm weight `1`, bias `0`
- early stopping: validation loss, patience `2`
- max epochs: `100` safety cap
- dropout: `0.5`
- default seeds:
  - first debug run: `42`
  - full paper-style run: five independent seeds, matching Stage 1 policy

No pretrained stock checkpoint transfer is used in the default Stage 2 BTC
baseline. Stage 2 trains BTC models from scratch after BTC images and labels are
constructed.

## Validation Checks

Each implementation must pass:
- parameter count check for I5/I20/I60;
- shape check for every block and final logits;
- batch input check for all four image specs;
- `CrossEntropyLoss` compatibility check with labels `(batch,)`;
- Grad-CAM target layer availability check.

Expected local smoke output:

```text
stock_cnn_i5  input=(2,1,32,15)  flatten=15360   params=155138
stock_cnn_i20 input=(2,1,64,60)  flatten=46080   params=708866
stock_cnn_i60 input=(2,1,96,180) flatten=184320  params=2952962
```

## 한국어

상태: checklist 2-5 계획 완료. 실제 구현은 이후 `2-I5`에서 합니다.

목적:
- Stage 1의 Re-image/Stock_CNN식 CNN core를 BTC image에 적용합니다.
- 연구 설계는 고정합니다. Stage 2는 BTC 자산군 확장이지 새 모델 설계가 아닙니다.
- 모든 BTC window를 I20 전용 Stage 1 모델에 억지로 넣지 않습니다.

근거:
- Re-image 요약: `자료조사/Re-image 요약.md`
  - line 36: baseline image size `32x15`, `64x60`, `96x180`.
  - line 47: I5/I20/I60 CNN depth, channel, flatten dimension, parameter count.
  - line 49: cross-entropy, Adam, batch size `128`, dropout `0.5`, Xavier
    initialization, validation-loss early stopping.
- Stage 1 model implementation:
  - `stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py`
- GitHub reference:
  - `lich99/Stock_CNN`, `models/baseline.py`, commit
    `415e2acf2a5013afca67e383acd3edc61fced840`.

중요한 source 제한:
- 확인한 GitHub 구현은 I20 전용입니다. 모든 input을 `(batch, 1, 64, 60)`으로
  reshape하고 `Linear(46080, 2)`를 사용합니다.
- 따라서 I20은 Stage 1/GitHub식 core를 그대로 재사용할 수 있습니다.
- I5와 I60은 논문 architecture target을 기준으로 별도 Stage-1/Stock_CNN식 variant를
  구현해야 합니다. I20 reshape나 I20 classifier를 재사용하면 안 됩니다.

## Model Variants

| Image window | Model name | Input shape | Blocks | Channels | Flatten dim | Expected params |
| --- | --- | --- | ---: | --- | ---: | ---: |
| I5 | `stock_cnn_i5` | `(batch, 1, 32, 15)` | 2 | `64 -> 128` | 15,360 | 155,138 |
| I20 | `stock_cnn_i20` | `(batch, 1, 64, 60)` | 3 | `64 -> 128 -> 256` | 46,080 | 708,866 |
| I60 | `stock_cnn_i60` | `(batch, 1, 96, 180)` | 4 | `64 -> 128 -> 256 -> 512` | 184,320 | 2,952,962 |

CSV artifact:
- `stage2_btc_extension/reports/tables/stage2_baseline_cnn_architecture_plan.csv`

## Fixed CNN Block

각 block은 Stage 1/Re-image 순서를 유지합니다:

```text
Conv2d -> BatchNorm2d -> LeakyReLU(0.01) -> MaxPool2d(2, 1)
```

공통 고정값:
- input channels: `1`
- output classes: `2`
- convolution kernel: `(5, 3)`
- pooling: `(2, 1)`
- classifier 전 dropout: `0.5`
- output: raw logits `(batch, 2)`
- Up class index: `1`
- softmax는 `forward()` 안이 아니라 evaluation에서만 적용합니다.

I20 exact implementation:
- 확인된 GitHub식 Stage 1 core를 사용합니다:
  - conv block 3개;
  - `stride=(3, 1)`;
  - 세 conv layer 모두 `dilation=(2, 1)`;
  - 세 conv layer 모두 `padding=(12, 1)`;
  - `Linear(46080, 2)`.

I5/I60 구현 규칙:
- window별 variant를 따로 만들고, 논문이 보고한 block 수, channel sequence,
  flatten dimension, parameter count를 맞춥니다.
- 구현 전에 Figure 7과 CNN section의 로컬 PDF/page mapping을 다시 확인합니다.
- 보고된 shape를 맞추기 위한 stride/padding/dilation detail이 Stage 1/GitHub I20
  pattern과 충돌하면 `source_map.md`에 mismatch를 기록하고, paper-reported
  shape/parameter target을 guardrail로 둡니다.

## Experiment Mapping

모델은 return horizon이 아니라 image window로 선택합니다:

| Experiment group | Model |
| --- | --- |
| I5/R5, I5/R20, I5/R60 | `stock_cnn_i5` |
| I20/R5, I20/R20, I20/R60 | `stock_cnn_i20` |
| I60/R5, I60/R20, I60/R60 | `stock_cnn_i60` |

네 가지 image specification은 model architecture를 바꾸지 않습니다. MA와 volume이
별도 channel이 아니라 같은 1-channel grayscale image 안에 그려지기 때문입니다:
- `ohlc`
- `ohlc_vb`
- `ohlc_ma`
- `ohlc_ma_vb`

따라서 Stage 2 baseline은 다음 구조입니다:

```text
3 image windows x 3 return horizons x 4 image specs = 36 baseline runs
```

## Implementation Interface

구현 대상:
- `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py`

필수 API:
- `build_stock_cnn_from_config(config) -> nn.Module`
- `StockCNNConfig`
- `StockCNNI5`
- `StockCNNI20`
- `StockCNNI60`
- `count_parameters(model)`
- `forward_with_shapes(x)`
- `gradcam_target_layers()`

이 interface를 두는 이유:
- `forward()`는 GitHub model과 가깝게 유지하고 logits만 반환합니다.
- `forward_with_shapes()`로 local smoke test에서 tensor shape를 확인합니다.
- `gradcam_target_layers()`는 Stage 2 Grad-CAM에 필요합니다.
- 이후 Stage 3 Linear와 Stage 4 FiLM에서는 CNN feature extractor와 classifier
  경계에 깔끔하게 접근해야 합니다.

설계 제한:
- helper method는 inspection과 이후 확장을 위해서만 추가합니다.
- baseline CNN forward path가 만드는 logits는 바꾸지 않습니다.

## Training Defaults

Stage 2 baseline training은 나중 checklist에서 바꿀 이유를 기록하지 않는 한 Stage 1
기본값을 유지합니다:
- loss: cross-entropy
- optimizer: Adam
- learning rate: `1e-5`
- batch size: `128`
- initialization: Conv/Linear는 Xavier, BatchNorm weight `1`, bias `0`
- early stopping: validation loss, patience `2`
- max epochs: `100` safety cap
- dropout: `0.5`
- default seeds:
  - first debug run: `42`
  - full paper-style run: Stage 1 policy와 맞춰 5회 independent seed

기본 Stage 2 BTC baseline에서는 pretrained stock checkpoint transfer를 사용하지
않습니다. BTC image와 label을 만든 뒤 BTC model을 from scratch로 학습합니다.

## Validation Checks

구현은 반드시 다음을 통과해야 합니다:
- I5/I20/I60 parameter count check;
- block별 shape와 final logits shape check;
- 네 image spec에 대한 batch input check;
- label `(batch,)`와 `CrossEntropyLoss` compatibility check;
- Grad-CAM target layer availability check.

예상 local smoke output:

```text
stock_cnn_i5  input=(2,1,32,15)  flatten=15360   params=155138
stock_cnn_i20 input=(2,1,64,60)  flatten=46080   params=708866
stock_cnn_i60 input=(2,1,96,180) flatten=184320  params=2952962
```
