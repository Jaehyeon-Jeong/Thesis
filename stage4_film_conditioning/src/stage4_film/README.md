# Stage 4 Python Package

## English

Stage 4 FiLM/context-conditioning implementation package.

Added in `4-I1`:
- `config.py`
- `paths.py`
- `runtime.py`
- `seed.py`

Added in `4-I2`:
- `context/sources.py`
- `context/features.py`
- `context/normalization.py`

Added in `4-I3`:
- `conditions/context_encoder.py`

Added in `4-I4`:
- `models/context_stock_cnn.py`

Updated in `4-I5`:
- `models/context_stock_cnn.py`: added `GatedContextStockCNN`.

Added in `4-I6`:
- `layers/film.py`: reusable feature-wise affine modulation layer.
- `conditions/film_generator.py`: context-embedding-to-gamma/beta generator
  for gamma-only and full FiLM.

Added in `4-I7`:
- `models/film_stock_cnn.py`: `FilmContextStockCNN` for `film_gamma` and
  `film_full`, with FiLM inserted after BatchNorm and before LeakyReLU in every
  Stock_CNN block.

Updated in `4-V7`:
- `models/film_stock_cnn.py`: `BoundedLastBlockFilmContextStockCNN` for
  `film_full_bounded_last_block`, with residual/bounded FiLM applied only to
  the final Stock_CNN block.
- `runners/context_experiment.py`: routes the new V7 method through the same
  fixed Stage 2 BTC data pipeline.

Added in `4-I8`:
- `training/loop.py`: context-aware training loop that calls
  `model(image, context)` and preserves identity initialization for gate/FiLM
  heads after generic weight initialization.
- `runners/context_experiment.py`: Stage 4 data/model runner that reuses Stage
  2 BTC samples, images, split, and pixel normalization, then attaches
  normalized context vectors.

Added in `4-I9`:
- `evaluation/prediction.py`: context-aware prediction helper that reloads a
  Stage 4 checkpoint and exports rows from `model(image, context)`.

Added in `4-I10`:
- `interpretability/gradcam_context.py`: Stage 4 Grad-CAM helper that calls
  `model(image, context)` and exports context/gate/gamma/beta interpretation
  metadata beside the Grad-CAM samples.

Stage 4 should import Stage 2 helpers through a configurable Stage 2 `src`
path. Do not duplicate the Stage 2 BTC pipeline unless a later implementation
blocker requires it.

## 한국어

Stage 4 FiLM/context-conditioning 구현 package입니다.

`4-I1`에서 추가한 module:
- `config.py`
- `paths.py`
- `runtime.py`
- `seed.py`

`4-I2`에서 추가한 module:
- `context/sources.py`
- `context/features.py`
- `context/normalization.py`

`4-I3`에서 추가한 module:
- `conditions/context_encoder.py`

`4-I4`에서 추가한 module:
- `models/context_stock_cnn.py`

`4-I5`에서 수정한 module:
- `models/context_stock_cnn.py`: `GatedContextStockCNN`을 추가했습니다.

`4-I6`에서 추가한 module:
- `layers/film.py`: 재사용 가능한 feature-wise affine modulation layer입니다.
- `conditions/film_generator.py`: context embedding에서 gamma/beta를 만드는
  gamma-only/full FiLM generator입니다.

`4-I7`에서 추가한 module:
- `models/film_stock_cnn.py`: `film_gamma`, `film_full`용
  `FilmContextStockCNN`입니다. 모든 Stock_CNN block에서 BatchNorm 뒤,
  LeakyReLU 전에 FiLM을 삽입합니다.

`4-V7`에서 수정한 module:
- `models/film_stock_cnn.py`: `film_full_bounded_last_block`용
  `BoundedLastBlockFilmContextStockCNN`을 추가했습니다. FiLM은 마지막
  Stock_CNN block에만 residual/bounded 방식으로 적용됩니다.
- `runners/context_experiment.py`: 새 V7 method도 같은 Stage 2 BTC data
  pipeline을 통해 실행되도록 연결했습니다.

`4-I8`에서 추가한 module:
- `training/loop.py`: `model(image, context)`를 호출하는 context-aware training
  loop입니다. 일반 weight initialization 뒤 gate/FiLM head를 identity로 다시
  보존합니다.
- `runners/context_experiment.py`: Stage 2 BTC sample, image, split,
  pixel normalization을 재사용하고 normalized context vector를 붙이는 Stage 4
  data/model runner입니다.

`4-I9`에서 추가한 module:
- `evaluation/prediction.py`: Stage 4 checkpoint를 로드하고
  `model(image, context)`의 prediction row를 export하는 helper입니다.

`4-I10`에서 추가한 module:
- `interpretability/gradcam_context.py`: `model(image, context)` 기준
  Stage 4 Grad-CAM을 만들고, 선택 sample의 context/gate/gamma/beta 해석 metadata를
  같이 export하는 helper입니다.

Stage 4는 configurable Stage 2 `src` path를 통해 Stage 2 helper를 import해야
합니다. 구현 blocker가 생기기 전에는 Stage 2 BTC pipeline을 중복 작성하지 않습니다.
