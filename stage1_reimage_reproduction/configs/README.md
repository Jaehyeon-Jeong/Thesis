# Configs

## English

Configuration files for Stage 1 local smoke tests and Kaggle full runs.

Current configs:
- `env_local.yaml`: local smoke-test paths and CPU/auto runtime.
- `env_kaggle.yaml`: Kaggle Notebook paths and CUDA runtime.

Both configs define:
- data paths and expected public `monthly_20d` shards
- deterministic split and train-only normalization rules
- `StockCNNI20` model source metadata
- training settings
- evaluation/prediction-output settings

## 한국어

1단계 local smoke test와 Kaggle full run을 위한 config입니다.

현재 config:
- `env_local.yaml`: local smoke-test path와 CPU/auto runtime.
- `env_kaggle.yaml`: Kaggle Notebook path와 CUDA runtime.

두 config 모두 다음을 정의합니다:
- data path와 public `monthly_20d` shard 기대값
- deterministic split과 train-only normalization 규칙
- `StockCNNI20` model source metadata
- training settings
- evaluation/prediction-output settings
