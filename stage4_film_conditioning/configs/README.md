# Stage 4 Configs

## English

Stage 4 configs were added in checklist item `4-I1`.

Config files:
- `env_local.yaml`
- `env_kaggle.yaml`

Required config additions beyond Stage 2:
- `stage2_dependency`: Stage 2 project root, Stage 2 `src` path, and optional
  Stage 2 baseline output root.
- `context`: F&G source filename/path, context window, selected context
  features, transform/imputation/clipping/scaling policy.
- `stage4_model`: ablation name, context encoder dimensions, gate/FiLM
  initialization policy.

Validation:
- `python scripts/check_stage4_scaffold.py --config configs/env_local.yaml`
  confirms that local BTC, local F&G, and Stage 2 `src` paths are available.

## 한국어

Stage 4 config는 체크리스트 `4-I1`에서 추가했습니다.

Config 파일:
- `env_local.yaml`
- `env_kaggle.yaml`

Stage 2 대비 추가해야 할 config:
- `stage2_dependency`: Stage 2 project root, Stage 2 `src` path, optional Stage 2
  baseline output root.
- `context`: F&G source filename/path, context window, selected context
  features, transform/imputation/clipping/scaling policy.
- `stage4_model`: ablation name, context encoder dimension, gate/FiLM
  initialization policy.

검증:
- `python scripts/check_stage4_scaffold.py --config configs/env_local.yaml`로
  local BTC, local F&G, Stage 2 `src` path가 사용 가능함을 확인합니다.
