# Stage 4 Source Code

## English

Stage 4 source code started in checklist item `4-I1`.

Package:
- `src/stage4_film/`

Added in `4-I1`:
- `config.py`
- `paths.py`
- `runtime.py`
- `seed.py`

Implementation readiness decision:
- Reuse Stage 2 `src/stage2_btc` for BTC data/image/split/evaluation helpers.
- Add Stage 4-specific code only for context features, context encoders,
  concat/gating/FiLM models, runner orchestration, output checks, and
  context/modulation exports.

## 한국어

Stage 4 source code는 checklist item `4-I1`부터 추가했습니다.

Package:
- `src/stage4_film/`

`4-I1`에서 추가한 module:
- `config.py`
- `paths.py`
- `runtime.py`
- `seed.py`

Implementation readiness 결정:
- BTC data/image/split/evaluation helper는 Stage 2 `src/stage2_btc`를 재사용합니다.
- Stage 4에는 context feature, context encoder, concat/gating/FiLM model,
  runner orchestration, output check, context/modulation export만 추가합니다.
