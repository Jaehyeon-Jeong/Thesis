# 4-6 Concat/Gating/FiLM Insertion Design

## English

Status: complete for planning.

Purpose:
- lock where the 4-A/4-B/4-C/4-D context paths attach to the fixed Stage 2
  `I60/R20/ohlc_ma_vb` Stock_CNN;
- define tensor shapes, modulation heads, and identity-safe initialization;
- keep the four ablations comparable and small enough for the BTC sample size;
- separate the shared context encoder from fusion/modulation heads.

This is an implementation choice for this thesis. It is not reported by the
Re-image paper. The FiLM placement follows the local FiLM reference review:
apply channel-wise modulation after BatchNorm and before activation.

## Fixed Inputs From 4-4 and 4-5

Primary Stage 4 model family:

```text
image/model = Stage 2 I60/R20/ohlc_ma_vb
input image = (batch_size, 1, 96, 180)
context     = normalized 8-feature vector
embedding   = context encoder output, (batch_size, 32)
```

The shared context encoder architecture is fixed by 4-5:

```text
Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU
```

Each ablation trains its own copy of this encoder. "Shared" means the same
architecture and preprocessing, not shared weights across separately trained
models.

## I60 Stock_CNN Shape Contract

The selected I60 Stage 2 model uses four blocks with channels
`64 -> 128 -> 256 -> 512`.

Each block is:

```text
Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d
```

For Stage 4 FiLM models, the block becomes:

```text
Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d
```

Expected I60 shapes:

| Location | Shape |
| --- | --- |
| input | `(B, 1, 96, 180)` |
| block1 BatchNorm output | `(B, 64, 36, 180)` |
| block1 output after pool | `(B, 64, 18, 180)` |
| block2 BatchNorm output | `(B, 128, 10, 180)` |
| block2 output after pool | `(B, 128, 5, 180)` |
| block3 BatchNorm output | `(B, 256, 6, 180)` |
| block3 output after pool | `(B, 256, 3, 180)` |
| block4 BatchNorm output | `(B, 512, 5, 180)` |
| block4 output after pool | `(B, 512, 2, 180)` |
| flatten | `(B, 184320)` |
| logits | `(B, 2)` |

The BatchNorm-output rows are the insertion points for 4-C and 4-D.

## 4-A. CNN + Context Concat

Design:

```text
image -> CNN layers -> flatten image_feature: (B, 184320)
context -> shared context encoder -> context_embedding: (B, 32)

concat_feature = [image_feature, context_embedding]: (B, 184352)
Dropout(0.5) -> Linear(184352, 2) -> logits
```

Rules:
- Do not change the convolution blocks.
- Do not let context affect visual feature extraction.
- Only the final classifier sees context.

Interpretation:
- This is the naive side-information baseline.
- If concat matches FiLM, the thesis cannot claim FiLM is necessary.
- If FiLM beats concat, that supports the claim that conditional feature
  modulation matters.

Parameter change versus Stage 2 I60:
- context encoder: `1,344` parameters;
- classifier extra parameters from adding 32 features: `64`;
- total extra trainable parameters: about `1,408`.

## 4-B. CNN + Context Gating

Design:

```text
image -> CNN layers -> final feature map F4: (B, 512, 2, 180)
context -> shared context encoder -> context_embedding: (B, 32)
context_embedding -> Linear(32, 512) -> raw_gate: (B, 512)
gate = 2 * sigmoid(raw_gate): (B, 512)
F4_gated = F4 * gate[:, :, None, None]
flatten -> Dropout(0.5) -> Linear(184320, 2) -> logits
```

Rules:
- Gate only the final CNN feature map for the first run.
- Use channel-wise gates, not a full `184320`-dimensional feature gate.
- Initialize the gate head to zero so `raw_gate = 0`, `gate = 1`, and the model
  starts at identity modulation.
- Do not add beta/shift in 4-B.

Why final-channel gating:
- It is a meaningful multiplicative alternative to concat.
- It is simpler than block-wise FiLM.
- It avoids a very large feature-level gate that would overfit the small BTC
  train split.

Parameter change versus Stage 2 I60:
- context encoder: `1,344`;
- gate head: `32 * 512 + 512 = 16,896`;
- total extra trainable parameters: about `18,240`.

## 4-C. CNN + Context FiLM Gamma-Only

Design:

For each convolution block `i`:

```text
context_embedding -> Linear(32, C_i) -> delta_gamma_i: (B, C_i)
gamma_i = 1 + delta_gamma_i

Conv2d -> BatchNorm2d -> gamma_i * feature -> LeakyReLU -> MaxPool2d
```

where:

```text
C = [64, 128, 256, 512]
sum(C) = 960
```

Rules:
- Generate one channel-wise `gamma` vector per block.
- Broadcast `gamma_i` from `(B, C_i)` to `(B, C_i, 1, 1)`.
- Initialize every gamma head to zero so `delta_gamma = 0` and `gamma = 1`.
- Do not generate beta in 4-C.

Interpretation:
- This isolates FiLM scaling.
- It is the closest ablation to the FiLM paper's gamma-centered analysis.
- It lets us inspect which market regimes scale which CNN channels.

Parameter change versus Stage 2 I60:
- context encoder: `1,344`;
- gamma heads: `32 * 960 + 960 = 31,680`;
- total extra trainable parameters: about `33,024`.

## 4-D. CNN + Context FiLM Full

Design:

For each convolution block `i`:

```text
context_embedding -> Linear(32, 2 * C_i) -> [delta_gamma_i, beta_i]
gamma_i = 1 + delta_gamma_i

Conv2d -> BatchNorm2d -> gamma_i * feature + beta_i -> LeakyReLU -> MaxPool2d
```

Rules:
- Generate channel-wise `gamma` and `beta` for every block.
- Broadcast both vectors from `(B, C_i)` to `(B, C_i, 1, 1)`.
- Initialize the final FiLM heads to zero so the model starts with
  `gamma = 1` and `beta = 0`.
- Keep the original Stock_CNN classifier dimension unchanged:
  `Dropout(0.5) -> Linear(184320, 2)`.

Interpretation:
- This is the primary Stage 4 FiLM model.
- Gamma shows context-dependent feature scaling.
- Beta shows context-dependent feature shifting.
- Saved gamma/beta values become the main interpretability object for Stage 4.

Parameter change versus Stage 2 I60:
- context encoder: `1,344`;
- gamma/beta heads: `2 * (32 * 960 + 960) = 63,360`;
- total extra trainable parameters: about `64,704`.

## Initialization and Training Fairness

Primary Stage 4 comparison rule:
- train all four context models from scratch under the same split, seed,
  optimizer, early stopping, and evaluation setup as the Stage 2 BTC pipeline;
- compare against the Stage 2 selected five-seed baseline;
- do not warm-start from a trained Stage 2 checkpoint in the primary result.

Reason:
- Warm-starting would test whether context can fine-tune a trained baseline,
  not whether the architecture is better under the same training protocol.
- Warm-start can be a later diagnostic, but it must be labeled separately.

Initialization:
- Keep Stage 2 Conv/Linear/BatchNorm initialization for the image branch.
- Initialize context encoder Linear layers with the same project default
  Linear initialization.
- Initialize gating and FiLM output heads to zero:
  - gate starts as `1`;
  - gamma starts as `1`;
  - beta starts as `0`.

This preserves the unconditioned feature scale at the first optimization step.

## Forward API Design

All Stage 4 models should support:

```text
logits = model(image, context)
```

and a diagnostic mode:

```text
output = model(image, context, return_context_outputs=True)
```

The diagnostic output should include:
- `logits`;
- `context_embedding`;
- for 4-B: `gate_layer4`;
- for 4-C: `gamma_layer1`, `gamma_layer2`, `gamma_layer3`, `gamma_layer4`;
- for 4-D: all gamma and beta tensors.

4-7 will define which of these tensors are written to disk for Grad-CAM and
interpretability reports.

## Guardrails

Implementation must fail early if:
- `image_window != 60` for the first Stage 4 main run;
- context vector dimension is not `8`;
- context embedding dimension is not `32`;
- flatten dimension is not `184320`;
- FiLM/gate tensors are not broadcastable to their target feature maps;
- a model silently falls back to the Stage 2 classifier while expecting concat
  input dimension `184352`.

## 4-6 Decision

Proceed to 4-7 and implementation planning with:

```text
4-A concat:
    [flattened image feature 184320, context embedding 32]
    -> Dropout(0.5) -> Linear(184352, 2)

4-B gating:
    final layer4 feature map (B, 512, 2, 180)
    context embedding -> Linear(32, 512)
    gate = 2 * sigmoid(raw_gate)
    F4' = gate * F4

4-C gamma-only FiLM:
    after BatchNorm in every block
    gamma_i = 1 + Linear(32, C_i)(context_embedding)
    F_i' = gamma_i * F_i

4-D full FiLM:
    after BatchNorm in every block
    gamma_i, beta_i = Linear(32, 2*C_i)(context_embedding)
    gamma_i = 1 + delta_gamma_i
    F_i' = gamma_i * F_i + beta_i
```

## 한국어

상태: 계획 단계 완료.

목적:
- 4-A/4-B/4-C/4-D context path를 고정된 Stage 2
  `I60/R20/ohlc_ma_vb` Stock_CNN의 어디에 붙일지 고정합니다.
- tensor shape, modulation head, identity-safe initialization을 정의합니다.
- 네 ablation을 비교 가능하게 유지하고 BTC sample size에 비해 과한 모델이 되지
  않게 합니다.
- shared context encoder와 fusion/modulation head를 분리합니다.

이 내용은 이 논문의 구현 선택입니다. Re-image 논문에 보고된 내용은 아닙니다.
FiLM 위치는 local FiLM reference review를 따릅니다: BatchNorm 뒤, activation 전
channel-wise modulation을 적용합니다.

## 4-4와 4-5에서 고정된 입력

Primary Stage 4 model family:

```text
image/model = Stage 2 I60/R20/ohlc_ma_vb
input image = (batch_size, 1, 96, 180)
context     = normalized 8-feature vector
embedding   = context encoder output, (batch_size, 32)
```

Shared context encoder는 4-5에서 다음처럼 고정했습니다.

```text
Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU
```

각 ablation은 이 encoder의 자기 복사본을 학습합니다. 여기서 "shared"는 네 모델이
같은 weight를 공유한다는 뜻이 아니라, 같은 구조와 같은 preprocessing을 쓴다는
뜻입니다.

## I60 Stock_CNN shape 규칙

선택된 I60 Stage 2 모델은 네 block을 사용합니다.

```text
64 -> 128 -> 256 -> 512
```

각 block은 다음 구조입니다.

```text
Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d
```

Stage 4 FiLM model에서는 다음처럼 바뀝니다.

```text
Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d
```

I60 예상 shape:

| 위치 | Shape |
| --- | --- |
| input | `(B, 1, 96, 180)` |
| block1 BatchNorm output | `(B, 64, 36, 180)` |
| block1 output after pool | `(B, 64, 18, 180)` |
| block2 BatchNorm output | `(B, 128, 10, 180)` |
| block2 output after pool | `(B, 128, 5, 180)` |
| block3 BatchNorm output | `(B, 256, 6, 180)` |
| block3 output after pool | `(B, 256, 3, 180)` |
| block4 BatchNorm output | `(B, 512, 5, 180)` |
| block4 output after pool | `(B, 512, 2, 180)` |
| flatten | `(B, 184320)` |
| logits | `(B, 2)` |

BatchNorm output 위치가 4-C와 4-D의 삽입 지점입니다.

## 4-A. CNN + Context Concat

설계:

```text
image -> CNN layers -> flatten image_feature: (B, 184320)
context -> shared context encoder -> context_embedding: (B, 32)

concat_feature = [image_feature, context_embedding]: (B, 184352)
Dropout(0.5) -> Linear(184352, 2) -> logits
```

규칙:
- convolution block은 바꾸지 않습니다.
- context가 visual feature extraction을 바꾸지 않습니다.
- context는 마지막 classifier에서만 보입니다.

해석:
- 가장 단순한 side-information baseline입니다.
- concat이 FiLM과 비슷하면 FiLM의 필요성을 강하게 주장하기 어렵습니다.
- FiLM이 concat보다 좋으면 conditional feature modulation이 중요하다는 주장에
  힘이 생깁니다.

Stage 2 I60 대비 parameter 증가:
- context encoder: `1,344`;
- 32개 feature 추가로 늘어나는 classifier parameter: `64`;
- 총 추가 trainable parameter: 약 `1,408`.

## 4-B. CNN + Context Gating

설계:

```text
image -> CNN layers -> final feature map F4: (B, 512, 2, 180)
context -> shared context encoder -> context_embedding: (B, 32)
context_embedding -> Linear(32, 512) -> raw_gate: (B, 512)
gate = 2 * sigmoid(raw_gate): (B, 512)
F4_gated = F4 * gate[:, :, None, None]
flatten -> Dropout(0.5) -> Linear(184320, 2) -> logits
```

규칙:
- 첫 run에서는 final CNN feature map만 gate합니다.
- `184320`차원 전체 feature gate가 아니라 channel-wise gate를 사용합니다.
- gate head를 zero initialize해서 `raw_gate = 0`, `gate = 1`에서 시작합니다.
- 4-B에는 beta/shift를 추가하지 않습니다.

final-channel gating을 쓰는 이유:
- concat보다 강한 multiplicative modulation 대안입니다.
- block-wise FiLM보다 단순합니다.
- 작은 BTC train split에서 거대한 feature-level gate를 쓰는 overfitting 위험을 피합니다.

Stage 2 I60 대비 parameter 증가:
- context encoder: `1,344`;
- gate head: `32 * 512 + 512 = 16,896`;
- 총 추가 trainable parameter: 약 `18,240`.

## 4-C. CNN + Context FiLM Gamma-Only

설계:

각 convolution block `i`마다:

```text
context_embedding -> Linear(32, C_i) -> delta_gamma_i: (B, C_i)
gamma_i = 1 + delta_gamma_i

Conv2d -> BatchNorm2d -> gamma_i * feature -> LeakyReLU -> MaxPool2d
```

여기서:

```text
C = [64, 128, 256, 512]
sum(C) = 960
```

규칙:
- block마다 channel-wise `gamma` vector 하나를 만듭니다.
- `gamma_i`는 `(B, C_i)`에서 `(B, C_i, 1, 1)`로 broadcast합니다.
- 모든 gamma head를 zero initialize해서 `delta_gamma = 0`, `gamma = 1`에서
  시작합니다.
- 4-C에서는 beta를 만들지 않습니다.

해석:
- FiLM scaling만 분리해서 보는 ablation입니다.
- FiLM 논문의 gamma 중심 분석과 가장 직접적으로 연결됩니다.
- 어떤 market regime에서 어떤 CNN channel이 scale되는지 확인할 수 있습니다.

Stage 2 I60 대비 parameter 증가:
- context encoder: `1,344`;
- gamma heads: `32 * 960 + 960 = 31,680`;
- 총 추가 trainable parameter: 약 `33,024`.

## 4-D. CNN + Context FiLM Full

설계:

각 convolution block `i`마다:

```text
context_embedding -> Linear(32, 2 * C_i) -> [delta_gamma_i, beta_i]
gamma_i = 1 + delta_gamma_i

Conv2d -> BatchNorm2d -> gamma_i * feature + beta_i -> LeakyReLU -> MaxPool2d
```

규칙:
- 모든 block에 channel-wise `gamma`와 `beta`를 만듭니다.
- 두 vector 모두 `(B, C_i)`에서 `(B, C_i, 1, 1)`로 broadcast합니다.
- 최종 FiLM head를 zero initialize해서 `gamma = 1`, `beta = 0`에서 시작합니다.
- 기존 Stock_CNN classifier dimension은 그대로 둡니다:
  `Dropout(0.5) -> Linear(184320, 2)`.

해석:
- Stage 4의 main FiLM model입니다.
- gamma는 context-dependent feature scaling입니다.
- beta는 context-dependent feature shifting입니다.
- 저장된 gamma/beta 값이 Stage 4의 핵심 해석 대상입니다.

Stage 2 I60 대비 parameter 증가:
- context encoder: `1,344`;
- gamma/beta heads: `2 * (32 * 960 + 960) = 63,360`;
- 총 추가 trainable parameter: 약 `64,704`.

## 초기화와 학습 공정성

Primary Stage 4 비교 규칙:
- 네 context model은 Stage 2 BTC pipeline과 같은 split, seed, optimizer,
  early stopping, evaluation setup으로 scratch training합니다.
- 비교 기준은 Stage 2 selected five-seed baseline입니다.
- Primary result에서는 학습된 Stage 2 checkpoint로 warm-start하지 않습니다.

이유:
- Warm-start를 쓰면 "같은 학습 protocol에서 architecture가 나은가"가 아니라
  "이미 학습된 baseline에 context를 붙여 fine-tune하면 좋아지는가"를 테스트하게 됩니다.
- Warm-start는 나중의 diagnostic으로는 가능하지만 primary result와 분리해서 표시해야 합니다.

초기화:
- image branch는 Stage 2 Conv/Linear/BatchNorm initialization을 유지합니다.
- context encoder Linear layer는 프로젝트 기본 Linear initialization을 사용합니다.
- gating/FiLM output head는 zero initialize합니다.
  - gate는 `1`에서 시작합니다.
  - gamma는 `1`에서 시작합니다.
  - beta는 `0`에서 시작합니다.

이렇게 해야 첫 optimization step에서 unconditioned feature scale을 망가뜨리지 않습니다.

## Forward API 설계

모든 Stage 4 model은 다음을 지원해야 합니다.

```text
logits = model(image, context)
```

그리고 diagnostic mode도 지원해야 합니다.

```text
output = model(image, context, return_context_outputs=True)
```

Diagnostic output에는 다음을 포함합니다.
- `logits`;
- `context_embedding`;
- 4-B: `gate_layer4`;
- 4-C: `gamma_layer1`, `gamma_layer2`, `gamma_layer3`, `gamma_layer4`;
- 4-D: 모든 gamma와 beta tensor.

4-7에서 이 tensor 중 어떤 것을 disk에 저장할지 Grad-CAM/해석 보고 기준으로
정의합니다.

## Guardrails

구현은 다음 경우 명확히 실패해야 합니다.
- 첫 Stage 4 main run에서 `image_window != 60`;
- context vector dimension이 `8`이 아님;
- context embedding dimension이 `32`가 아님;
- flatten dimension이 `184320`이 아님;
- FiLM/gate tensor가 대상 feature map에 broadcast되지 않음;
- concat input dimension `184352`를 기대하면서 Stage 2 classifier로 조용히 fallback함.

## 4-6 결정

다음 결정으로 4-7과 구현 계획으로 넘어갑니다.

```text
4-A concat:
    [flattened image feature 184320, context embedding 32]
    -> Dropout(0.5) -> Linear(184352, 2)

4-B gating:
    final layer4 feature map (B, 512, 2, 180)
    context embedding -> Linear(32, 512)
    gate = 2 * sigmoid(raw_gate)
    F4' = gate * F4

4-C gamma-only FiLM:
    after BatchNorm in every block
    gamma_i = 1 + Linear(32, C_i)(context_embedding)
    F_i' = gamma_i * F_i

4-D full FiLM:
    after BatchNorm in every block
    gamma_i, beta_i = Linear(32, 2*C_i)(context_embedding)
    gamma_i = 1 + delta_gamma_i
    F_i' = gamma_i * F_i + beta_i
```
