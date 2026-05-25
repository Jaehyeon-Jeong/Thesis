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

Planned next modules:
- `layers/film.py`
- `models/film_stock_cnn.py`
- `training/`
- `evaluation/`
- `interpretability/`
- `runners/`

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

다음 예정 module:
- `layers/film.py`
- `models/film_stock_cnn.py`
- `training/`
- `evaluation/`
- `interpretability/`
- `runners/`

Stage 4는 configurable Stage 2 `src` path를 통해 Stage 2 helper를 import해야
합니다. 구현 blocker가 생기기 전에는 Stage 2 BTC pipeline을 중복 작성하지 않습니다.
