# Stage 2 Configs

## English

Stage 2 now has two environment configs:
- `env_local.yaml`: local smoke-test and documentation environment.
- `env_kaggle.yaml`: Kaggle full-run environment.

Both configs use the same schema. Local/Kaggle differences are handled through
path and runtime values, not through separate Python implementations.

Default training policy:
- `batch_size: 128`
- shared Stage 1 CNN core unless explicitly changed later
- strict default uses no mixed precision and no DataParallel

## 한국어

Stage 2 config는 두 개의 실행 환경 파일로 나눕니다.
- `env_local.yaml`: local smoke-test와 문서 확인용 환경.
- `env_kaggle.yaml`: Kaggle full-run 환경.

두 config는 같은 schema를 사용합니다. local/Kaggle 차이는 Python 코드를 두 벌로
나누지 않고 path와 runtime 값으로만 처리합니다.

기본 학습 정책:
- `batch_size: 128`
- 나중에 명시적으로 변경하지 않는 한 Stage 1 CNN core 공유
- strict 기본값에서는 mixed precision과 DataParallel을 사용하지 않음
