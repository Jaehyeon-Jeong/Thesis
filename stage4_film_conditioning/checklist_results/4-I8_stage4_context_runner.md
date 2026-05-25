# 4-I8 Stage 4 Context Runner

## English

Status: complete

Implemented the first Stage 4 training runner that keeps the Stage 2 BTC
image pipeline fixed and adds only the Stage 4 context branch.

Added files:
- `src/stage4_film/training/loop.py`
- `src/stage4_film/training/__init__.py`
- `src/stage4_film/runners/context_experiment.py`
- `src/stage4_film/runners/__init__.py`
- `scripts/run_stage4_context_model.py`

Updated files:
- `src/stage4_film/__init__.py`
- `src/stage4_film/conditions/film_generator.py`

What the runner does:
1. Reuses Stage 2 BTC data loading, sample construction, chart-image
   generation, train/validation/test split, and train-only pixel
   normalization.
2. Builds the Stage 4 structured context table from BTC OHLCV plus F&G.
3. Fits context preprocessing on the train split only.
4. Wraps each Stage 2 image sample with a normalized `context` tensor.
5. Builds one Stage 4 model: `concat`, `gating`, `film_gamma`, or `film_full`.
6. Trains the model with `model(image, context)`.
7. Writes checkpoint, training history, train metadata, context artifacts, and
   a run manifest.

Important implementation detail:
- The generic Stage 2 initializer initializes all Conv/Linear/BatchNorm
  modules.
- After that initializer runs, Stage 4 resets context-modulation heads back to
  identity:
  - gating head -> raw gate `0`, so `gate = 1`
  - FiLM gamma heads -> `delta_gamma = 0`, so `gamma = 1`
  - FiLM beta heads -> `beta = 0`
- This preserves the Stage 2 visual feature path at initialization.

Local validation:
- `python -m py_compile` passed for:
  - `src/stage4_film/training/loop.py`
  - `src/stage4_film/runners/context_experiment.py`
  - `scripts/run_stage4_context_model.py`
  - `src/stage4_film/conditions/film_generator.py`
- Smoke run passed for `concat`:
  - `I60/R20/ohlc_ma_vb`
  - seed `42`
  - one epoch
  - train/validation/test row limits `16/8/8`
  - wrote checkpoint, metrics, context artifacts, and manifest.
- Smoke run passed for `film_gamma`:
  - `I60/R20/ohlc_ma_vb`
  - seed `42`
  - one epoch
  - train/validation/test row limits `4/4/4`
  - confirmed FiLM model training path works with `model(image, context)`.

Current boundary:
- `4-I8` trains and saves training-side artifacts only.
- Prediction CSV, classification metrics, trading metrics, Grad-CAM, and
  context/gate/gamma/beta interpretation exports are intentionally left for
  `4-I9` and `4-I10`.

## 한국어

상태: 완료

Stage 2 BTC image pipeline은 그대로 고정하고, Stage 4 context branch만 추가해서
학습할 수 있는 첫 runner를 구현했습니다.

추가한 파일:
- `src/stage4_film/training/loop.py`
- `src/stage4_film/training/__init__.py`
- `src/stage4_film/runners/context_experiment.py`
- `src/stage4_film/runners/__init__.py`
- `scripts/run_stage4_context_model.py`

수정한 파일:
- `src/stage4_film/__init__.py`
- `src/stage4_film/conditions/film_generator.py`

Runner가 하는 일:
1. Stage 2 BTC data loading, sample 생성, chart image 생성,
   train/validation/test split, train-only pixel normalization을 재사용합니다.
2. BTC OHLCV와 F&G에서 Stage 4 structured context table을 만듭니다.
3. Context preprocessing 통계는 train split에서만 fit합니다.
4. Stage 2 image sample마다 normalized `context` tensor를 붙입니다.
5. `concat`, `gating`, `film_gamma`, `film_full` 중 하나의 Stage 4 model을
   만듭니다.
6. `model(image, context)` 형태로 학습합니다.
7. checkpoint, training history, train metadata, context artifacts, run
   manifest를 저장합니다.

중요한 구현 디테일:
- Stage 2 initializer는 모든 Conv/Linear/BatchNorm을 일반적으로 초기화합니다.
- 그 다음 Stage 4가 context modulation head를 identity로 다시 reset합니다.
  - gating head -> raw gate `0`, 그래서 `gate = 1`
  - FiLM gamma head -> `delta_gamma = 0`, 그래서 `gamma = 1`
  - FiLM beta head -> `beta = 0`
- 따라서 Stage 4 model은 처음에는 Stage 2 visual feature path를 보존한 상태에서
  시작합니다.

Local 검증:
- 아래 파일들의 `python -m py_compile` 통과:
  - `src/stage4_film/training/loop.py`
  - `src/stage4_film/runners/context_experiment.py`
  - `scripts/run_stage4_context_model.py`
  - `src/stage4_film/conditions/film_generator.py`
- `concat` smoke run 통과:
  - `I60/R20/ohlc_ma_vb`
  - seed `42`
  - 1 epoch
  - train/validation/test row limit `16/8/8`
  - checkpoint, metrics, context artifacts, manifest 저장 확인.
- `film_gamma` smoke run 통과:
  - `I60/R20/ohlc_ma_vb`
  - seed `42`
  - 1 epoch
  - train/validation/test row limit `4/4/4`
  - FiLM model도 `model(image, context)` 학습 경로가 작동함을 확인.

현재 경계:
- `4-I8`은 학습과 training-side artifact 저장까지만 담당합니다.
- Prediction CSV, classification metrics, trading metrics, Grad-CAM,
  context/gate/gamma/beta 해석 export는 `4-I9`, `4-I10`에서 구현합니다.
