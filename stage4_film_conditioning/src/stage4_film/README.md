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

Planned next modules:
- `layers/film.py`
- `models/film_stock_cnn.py`
- `models/context_stock_cnn.py`
- `conditions/`
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

다음 예정 module:
- `layers/film.py`
- `models/film_stock_cnn.py`
- `models/context_stock_cnn.py`
- `conditions/`
- `training/`
- `evaluation/`
- `interpretability/`
- `runners/`

Stage 4는 configurable Stage 2 `src` path를 통해 Stage 2 helper를 import해야
합니다. 구현 blocker가 생기기 전에는 Stage 2 BTC pipeline을 중복 작성하지 않습니다.
