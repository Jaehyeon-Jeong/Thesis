# Stage 3 Pipeline

## English

Stage 3 adds a Linear adapter comparison to the BTC image-CNN pipeline.

Fixed inherited components from Stage 2:
- BTC OHLCV loader
- chart image generation for `I5`, `I20`, `I60`
- image specifications: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- future-return label rule: `return > 0 -> Up=1`
- train/validation/test split policy
- train-only pixel normalization
- classification metrics
- BTC long/flat and long/short trading metrics
- Grad-CAM output requirement

New Stage 3 component:
- A Linear adapter inserted after CNN feature extraction.
- First version uses `bias=False`.

Planned model flow:

```text
image
  -> fixed CNN feature extractor
  -> flatten feature
  -> Linear(feature_dim, feature_dim or adapter_dim, bias=False)
  -> classifier logits
  -> probability / prediction / metrics
```

The exact adapter dimension is not finalized in this scaffold step. It must be
decided in `3-2` after checking the Stage 2 model output shapes and the intended
comparison design.

## 한국어

Stage 3는 BTC image-CNN pipeline에 Linear adapter 비교를 추가합니다.

Stage 2에서 고정해서 가져오는 것:
- BTC OHLCV loader
- `I5`, `I20`, `I60` chart image generation
- image specification: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- 미래 수익률 label 규칙: `return > 0 -> Up=1`
- train/validation/test split 정책
- train-only pixel normalization
- classification metric
- BTC long/flat, long/short trading metric
- Grad-CAM output requirement

Stage 3에서 새로 추가하는 것:
- CNN feature extraction 뒤에 Linear adapter를 삽입합니다.
- 첫 버전은 `bias=False`를 사용합니다.

예정 model flow:

```text
image
  -> fixed CNN feature extractor
  -> flatten feature
  -> Linear(feature_dim, feature_dim 또는 adapter_dim, bias=False)
  -> classifier logits
  -> probability / prediction / metrics
```

정확한 adapter dimension은 이 scaffold 단계에서 확정하지 않습니다. Stage 2 model
output shape와 비교 설계를 확인한 뒤 `3-2`에서 결정해야 합니다.
