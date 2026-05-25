# Stage 4 Python Package

## English

Placeholder for the Stage 4 FiLM implementation.

Planned modules:
- `config.py`
- `paths.py`
- `runtime.py`
- `seed.py`
- `layers/film.py`
- `models/film_stock_cnn.py`
- `models/context_stock_cnn.py`
- `conditions/`
- `context/`
- `training/`
- `evaluation/`
- `interpretability/`
- `runners/`

Stage 4 should import Stage 2 helpers through a configurable Stage 2 `src`
path. Do not duplicate the Stage 2 BTC pipeline unless a later implementation
blocker requires it.

## 한국어

Stage 4 FiLM 구현을 위한 placeholder입니다.

예정 module:
- `config.py`
- `paths.py`
- `runtime.py`
- `seed.py`
- `layers/film.py`
- `models/film_stock_cnn.py`
- `models/context_stock_cnn.py`
- `conditions/`
- `context/`
- `training/`
- `evaluation/`
- `interpretability/`
- `runners/`

Stage 4는 configurable Stage 2 `src` path를 통해 Stage 2 helper를 import해야
합니다. 구현 blocker가 생기기 전에는 Stage 2 BTC pipeline을 중복 작성하지 않습니다.
