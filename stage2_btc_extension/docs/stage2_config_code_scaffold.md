# Stage 2 Shared Config/Code Scaffold

## English

Checklist item: `2-I1`

Purpose:
- Create the shared Stage 2 config and helper code that later BTC data loading,
  image generation, training, evaluation, trading metrics, and Grad-CAM will all
  use.
- Keep local and Kaggle execution on one codebase. Environment differences are
  config values, not separate implementations.

Created config files:
- `configs/env_local.yaml`
- `configs/env_kaggle.yaml`

Created code files:
- `src/stage2_btc/config.py`
- `src/stage2_btc/paths.py`
- `src/stage2_btc/runtime.py`
- `src/stage2_btc/seed.py`

Config decisions:
- Stage 2 default batch size remains `128`.
- Strict Stage 2 baseline defaults keep mixed precision off and DataParallel off.
- BTC report period is capped at `2024-12-31`; 2025-2026 remains optional later
  holdout.
- The four image specs are `ohlc`, `ohlc_vb`, `ohlc_ma`, and `ohlc_ma_vb`.
- Model selection is configured by image window:
  - `I5`: `stock_cnn_i5`
  - `I20`: `stock_cnn_i20`
  - `I60`: `stock_cnn_i60`

Implementation-source status:
- This item is implementation scaffolding, not a new paper decision.
- It follows the project root `PLAN.md` rule that local/Kaggle differences are
  controlled by config and that the same `src/` implementation is used in both
  environments.
- It follows the Stage 1 helper style in `stage1_reimage/config.py`,
  `paths.py`, `runtime.py`, and `seed.py`.

Verification:
- Python compile check passed for `src/stage2_btc`.
- Local config loads successfully.
- Local BTC source path resolves to the audited CSV.
- Kaggle config loads successfully. Calling `select_device()` on the Kaggle
  config from the local machine is expected to fail if CUDA is unavailable,
  because the Kaggle config intentionally requires CUDA for full runs.

Not implemented in this item:
- BTC OHLCV parsing.
- BTC image generation.
- Label/split/normalization execution.
- Model training, prediction export, trading metrics, or Grad-CAM.

## 한국어

체크리스트 항목: `2-I1`

목적:
- 이후 BTC data loading, image generation, training, evaluation, trading metric,
  Grad-CAM이 공통으로 사용할 Stage 2 config와 helper code를 만듭니다.
- local과 Kaggle 실행을 하나의 codebase로 유지합니다. 환경 차이는 별도 Python
  구현이 아니라 config 값으로 처리합니다.

생성한 config 파일:
- `configs/env_local.yaml`
- `configs/env_kaggle.yaml`

생성한 code 파일:
- `src/stage2_btc/config.py`
- `src/stage2_btc/paths.py`
- `src/stage2_btc/runtime.py`
- `src/stage2_btc/seed.py`

Config 결정:
- Stage 2 기본 batch size는 `128`입니다.
- strict Stage 2 baseline 기본값에서는 mixed precision과 DataParallel을 끕니다.
- BTC 기본 보고 기간은 `2024-12-31`까지이고, 2025-2026은 optional later holdout으로
  남깁니다.
- 네 가지 image spec은 `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`입니다.
- model은 image window 기준으로 선택합니다.
  - `I5`: `stock_cnn_i5`
  - `I20`: `stock_cnn_i20`
  - `I60`: `stock_cnn_i60`

구현 근거 상태:
- 이 항목은 새 논문 결정이 아니라 구현 scaffold입니다.
- local/Kaggle 차이를 config로 관리하고 같은 `src/` 구현을 사용한다는 root
  `PLAN.md` 규칙을 따릅니다.
- Stage 1 helper style인 `stage1_reimage/config.py`, `paths.py`, `runtime.py`,
  `seed.py` 구조를 따릅니다.

검증:
- `src/stage2_btc` Python compile check를 통과했습니다.
- local config가 정상적으로 load됩니다.
- local BTC source path가 audit했던 CSV로 resolve됩니다.
- Kaggle config도 정상적으로 load됩니다. 단, local machine에서 Kaggle config로
  `select_device()`를 호출하면 CUDA가 없을 때 실패하는 것이 정상입니다. Kaggle
  config는 full run에서 CUDA를 요구하도록 의도적으로 설정했습니다.

이번 항목에서 아직 구현하지 않은 것:
- BTC OHLCV parsing.
- BTC image generation.
- label/split/normalization 실행.
- model training, prediction export, trading metric, Grad-CAM.
