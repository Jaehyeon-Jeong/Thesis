# 3-2 Linear Adapter Design and Insertion-Point Review

## English

Purpose:
- Decide where the Linear component is inserted.
- Keep the Stage 2 CNN feature extractor unchanged.
- Avoid an infeasible dense `feature_dim -> feature_dim` layer.

Fixed rule from root `PLAN.md`:
- Stage 3 compares BTC CNN baseline against BTC CNN + `torch.nn.Linear(...,
  bias=False)`.
- Dense Linear and channel-wise Gamma-only FiLM are not the same. Stage 3 is a
  bridge comparison before Stage 4 FiLM.

Stage 2 feature dimensions:

| Image window | CNN variant | Flatten feature dim | Naive `feature_dim x feature_dim` params |
|---:|:---|---:|---:|
| `I5` | `stock_cnn_i5` | 15,360 | 235,929,600 |
| `I20` | `stock_cnn_i20` | 46,080 | 2,123,366,400 |
| `I60` | `stock_cnn_i60` | 184,320 | 33,973,862,400 |

Decision:
- Do not use `Linear(feature_dim, feature_dim)` as the default Stage 3 adapter.
  It is too large, especially for `I60`.
- Use a config-driven bias-free Linear adapter/head.
- Default fast comparison:

```text
image
  -> fixed Stage 2 Stock_CNN-style convolution blocks
  -> flattened feature vector: (batch_size, feature_dim)
  -> Linear(feature_dim, adapter_dim, bias=False)
  -> Linear(adapter_dim, 2, bias=False)
  -> logits: (batch_size, 2)
```

Default `adapter_dim`:
- `128` for the first Stage 3 grid.
- This keeps the adapter tractable:
  - `I5`: about `1.97M` adapter weights.
  - `I20`: about `5.90M` adapter weights.
  - `I60`: about `23.59M` adapter weights.

Why both Linear layers use `bias=False`:
- The Stage 3 comparison is meant to remove additive shift from the new adapter
  path.
- If the final classifier kept `bias=True`, the model would reintroduce an
  additive shift immediately after the bias-free adapter.

Implementation boundary:
- The convolution blocks, image shapes, label logic, split, normalization, and
  metrics remain unchanged.
- This is an implementation choice, not a reported detail from the Re-image
  paper.
- The exact adapter dimension must be stored in config and run metadata.

## 한국어

목적:
- Linear component를 어디에 넣을지 결정합니다.
- Stage 2 CNN feature extractor는 바꾸지 않습니다.
- 계산이 불가능한 dense `feature_dim -> feature_dim` layer를 피합니다.

Root `PLAN.md`에서 가져온 고정 규칙:
- Stage 3는 BTC CNN baseline과 BTC CNN + `torch.nn.Linear(..., bias=False)`를
  비교합니다.
- Dense Linear와 channel-wise Gamma-only FiLM은 같은 것이 아닙니다. Stage 3는
  Stage 4 FiLM 전의 bridge comparison입니다.

Stage 2 feature dimension:

| Image window | CNN variant | Flatten feature dim | 단순 `feature_dim x feature_dim` params |
|---:|:---|---:|---:|
| `I5` | `stock_cnn_i5` | 15,360 | 235,929,600 |
| `I20` | `stock_cnn_i20` | 46,080 | 2,123,366,400 |
| `I60` | `stock_cnn_i60` | 184,320 | 33,973,862,400 |

결정:
- 기본 Stage 3 adapter로 `Linear(feature_dim, feature_dim)`는 쓰지 않습니다.
  특히 `I60`에서 너무 큽니다.
- config로 제어되는 bias-free Linear adapter/head를 사용합니다.
- 기본 빠른 비교 구조:

```text
image
  -> 고정된 Stage 2 Stock_CNN-style convolution blocks
  -> flattened feature vector: (batch_size, feature_dim)
  -> Linear(feature_dim, adapter_dim, bias=False)
  -> Linear(adapter_dim, 2, bias=False)
  -> logits: (batch_size, 2)
```

기본 `adapter_dim`:
- 첫 Stage 3 grid에서는 `128`로 둡니다.
- 이 설정은 adapter를 계산 가능한 크기로 유지합니다:
  - `I5`: adapter weight 약 `1.97M`.
  - `I20`: adapter weight 약 `5.90M`.
  - `I60`: adapter weight 약 `23.59M`.

두 Linear layer 모두 `bias=False`로 두는 이유:
- Stage 3 비교는 새 adapter path에서 additive shift를 제거하는 것이 목적입니다.
- final classifier가 `bias=True`이면 bias-free adapter 바로 뒤에서 additive shift가
  다시 들어옵니다.

구현 경계:
- convolution block, image shape, label logic, split, normalization, metric은
  바꾸지 않습니다.
- 이 결정은 Re-image 논문에 보고된 값이 아니라 구현상 선택입니다.
- 정확한 adapter dimension은 config와 run metadata에 반드시 저장합니다.
