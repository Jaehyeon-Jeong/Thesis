# Stage 4 Configs

## English

Stage 4 configs will be added after checklist item `4-I1`.

Planned config files:
- `env_local.yaml`
- `env_kaggle.yaml`

Required config additions beyond Stage 2:
- `stage2_dependency`: Stage 2 project root, Stage 2 `src` path, and optional
  Stage 2 baseline output root.
- `context`: F&G source filename/path, context window, selected context
  features, transform/imputation/clipping/scaling policy.
- `stage4_model`: ablation name, context encoder dimensions, gate/FiLM
  initialization policy.

## 한국어

Stage 4 config는 체크리스트 `4-I1` 이후 추가합니다.

예정 config 파일:
- `env_local.yaml`
- `env_kaggle.yaml`

Stage 2 대비 추가해야 할 config:
- `stage2_dependency`: Stage 2 project root, Stage 2 `src` path, optional Stage 2
  baseline output root.
- `context`: F&G source filename/path, context window, selected context
  features, transform/imputation/clipping/scaling policy.
- `stage4_model`: ablation name, context encoder dimension, gate/FiLM
  initialization policy.
