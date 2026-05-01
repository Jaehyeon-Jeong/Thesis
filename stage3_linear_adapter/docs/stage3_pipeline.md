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
- The default first comparison uses `adapter_dim=128`.
- A naive `Linear(feature_dim, feature_dim)` is not used because `I60` would
  require about `33.97B` weights.

Planned model flow:

```text
image
  -> fixed CNN feature extractor
  -> flatten feature
  -> Linear(feature_dim, adapter_dim=128, bias=False)
  -> Linear(adapter_dim=128, 2, bias=False)
  -> classifier logits
  -> probability / prediction / metrics
```

The adapter dimension is config-driven. The first planned single-seed grid uses
`adapter_dim=128` to keep the `I60` model tractable.

Implemented model flow is the same as above. The current default parameter
counts are:
- I5 Linear adapter: `2,090,752`
- I20 Linear adapter: `6,515,200`
- I60 Linear adapter: `26,177,536`

Kaggle execution files:
- `notebooks/kaggle_stage3_linear_single_config_one_cell.md`
- `notebooks/kaggle_stage3_linear_grid_single_seed_one_cell.md`
- `notebooks/kaggle_stage3_results_viewer_one_cell.md`

Primary comparison:
- `36` Linear runs for seed `42`.
- Same axes as Stage 2:
  `I5/I20/I60 x R5/R20/R60 x ohlc/ohlc_vb/ohlc_ma/ohlc_ma_vb`.
- Join Stage 2 baseline and Stage 3 Linear outputs by image window, return
  horizon, image spec, and seed.

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
- 첫 비교의 기본값은 `adapter_dim=128`입니다.
- 단순 `Linear(feature_dim, feature_dim)`는 사용하지 않습니다. `I60`에서는 약
  `33.97B`개 weight가 필요하기 때문입니다.

예정 model flow:

```text
image
  -> fixed CNN feature extractor
  -> flatten feature
  -> Linear(feature_dim, adapter_dim=128, bias=False)
  -> Linear(adapter_dim=128, 2, bias=False)
  -> classifier logits
  -> probability / prediction / metrics
```

Adapter dimension은 config로 관리합니다. 첫 single-seed grid는 `I60` 모델을
계산 가능한 크기로 유지하기 위해 `adapter_dim=128`을 사용합니다.

구현된 model flow도 위와 같습니다. 현재 기본 parameter 수는 다음과 같습니다:
- I5 Linear adapter: `2,090,752`
- I20 Linear adapter: `6,515,200`
- I60 Linear adapter: `26,177,536`

Kaggle 실행 파일:
- `notebooks/kaggle_stage3_linear_single_config_one_cell.md`
- `notebooks/kaggle_stage3_linear_grid_single_seed_one_cell.md`
- `notebooks/kaggle_stage3_results_viewer_one_cell.md`

Primary comparison:
- seed `42` 기준 Linear run `36`개.
- Stage 2와 같은 축:
  `I5/I20/I60 x R5/R20/R60 x ohlc/ohlc_vb/ohlc_ma/ohlc_ma_vb`.
- Stage 2 baseline과 Stage 3 Linear output을 image window, return horizon,
  image spec, seed 기준으로 join합니다.
