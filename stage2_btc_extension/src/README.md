# Stage 2 Source

## English

Stage 2 implementation code will live under `src/stage2_btc/`.

Implemented modules:
- `config.py`: shared YAML config loader and schema checks
- `paths.py`: local/Kaggle path builder and output directory helper
- `runtime.py`: CPU/CUDA runtime selection
- `seed.py`: reproducibility helper
- `data/`: BTC OHLCV loading and cleaning
- `imaging/`: BTC OHLCV image generation
- `models/`: Stage 1 CNN-core reuse wrappers if needed
- `training/`: BTC baseline training runner helpers
- `evaluation/`: classification and trading metrics
- `interpretability/`: BTC Grad-CAM wrappers

## 한국어

Stage 2 구현 코드는 `src/stage2_btc/` 아래에 둡니다.

구현된 module:
- `config.py`: 공통 YAML config loader와 schema 검증
- `paths.py`: local/Kaggle path builder와 output directory helper
- `runtime.py`: CPU/CUDA runtime 선택
- `seed.py`: 재현성 helper
- `data/`: BTC OHLCV loading과 cleaning
- `imaging/`: BTC OHLCV image generation
- `models/`: 필요한 경우 Stage 1 CNN core reuse wrapper
- `training/`: BTC baseline training runner helper
- `evaluation/`: classification과 trading metric
- `interpretability/`: BTC Grad-CAM wrapper
