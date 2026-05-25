# Stage 4 Scripts

## English

Stage 4 scripts are added incrementally during implementation.

Added in `4-I1`:
- `_stage4_script_utils.py`: adds Stage 4 `src` and Stage 2 `src` to
  `sys.path`.
- `check_stage4_scaffold.py`: validates config, local BTC/F&G paths, Stage 2
  dependency, and primary experiment names.

Added in `4-I2`:
- `audit_stage4_context_sources.py`: audits BTC/F&G coverage and the as-of
  no-future-leakage merge policy.
- `build_stage4_context_features.py`: builds raw/normalized structured context
  features and the train-only context scaler.

Added in `4-I3`:
- `check_stage4_context_encoder.py`: validates the shared numeric context MLP
  on dummy tensors and, when available, real normalized context rows.

Added in `4-I4`:
- `check_stage4_model_shapes.py`: validates the Stage 4 concat model tensor
  path, parameter count, and real normalized context forward pass.

Planned next scripts:
- `check_stage4_film_layers.py`
- `run_stage4_context_model.py`
- `evaluate_stage4_predictions.py`
- `evaluate_stage4_trading.py`
- `generate_stage4_gradcam_context.py`
- `check_stage4_outputs.py`
- `summarize_stage4_results.py`

Checklist item 4-8 fixes the expected Kaggle execution order and backup
contract for these scripts.

Checklist item 4-I0 fixes that these scripts must add both Stage 4 `src` and
Stage 2 `src` to `sys.path`.

## 한국어

Stage 4 script는 구현 단계에서 순차적으로 추가합니다.

`4-I1`에서 추가한 script:
- `_stage4_script_utils.py`: Stage 4 `src`와 Stage 2 `src`를 `sys.path`에
  추가합니다.
- `check_stage4_scaffold.py`: config, local BTC/F&G path, Stage 2 dependency,
  primary experiment name을 검증합니다.

`4-I2`에서 추가한 script:
- `audit_stage4_context_sources.py`: BTC/F&G coverage와 as-of no-future-leakage
  merge policy를 감사합니다.
- `build_stage4_context_features.py`: raw/normalized structured context feature와
  train-only context scaler를 생성합니다.

`4-I3`에서 추가한 script:
- `check_stage4_context_encoder.py`: shared numeric context MLP를 dummy tensor와,
  가능하면 실제 normalized context row로 검증합니다.

`4-I4`에서 추가한 script:
- `check_stage4_model_shapes.py`: Stage 4 concat model의 tensor path,
  parameter count, 실제 normalized context forward pass를 검증합니다.

다음 예정 script:
- `check_stage4_film_layers.py`
- `run_stage4_context_model.py`
- `evaluate_stage4_predictions.py`
- `evaluate_stage4_trading.py`
- `generate_stage4_gradcam_context.py`
- `check_stage4_outputs.py`
- `summarize_stage4_results.py`

4-8에서 이 script들의 Kaggle 실행 순서와 backup 계약을 고정했습니다.

4-I0에서 이 script들이 Stage 4 `src`와 Stage 2 `src`를 모두 `sys.path`에
추가해야 한다고 고정했습니다.
