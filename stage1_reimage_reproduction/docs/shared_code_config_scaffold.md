# 1-I1 Shared Code/Config Scaffold

## English

Status:
- Completed on 2026-04-30.

Purpose:
- Create the shared Stage 1 package structure used by local smoke tests and
  Kaggle full runs.
- Keep runtime and path differences in config, not in separate codebases.

Implemented files:
- `src/stage1_reimage/__init__.py`
- `src/stage1_reimage/config.py`
- `src/stage1_reimage/paths.py`
- `src/stage1_reimage/runtime.py`
- `src/stage1_reimage/seed.py`
- `scripts/check_scaffold.py`

Implemented behavior:
- Load and validate Stage 1 YAML config sections.
- Build explicit local/Kaggle path objects from config.
- Create configured output directories when requested.
- Select runtime device from config.
- Set Python, NumPy, and PyTorch seeds when available.
- Run a local import/config smoke check without loading image data or training.

Scope limits:
- No `.dat` image loading was implemented in this gate.
- No label, split, normalization, model, training, evaluation, or Grad-CAM code
  was implemented in this gate.
- The next implementation gate remains `1-I2. Data Loading Implementation`.

Validation command:

```bash
python scripts/check_scaffold.py --config configs/env_local.yaml --create-output-dirs
```

Validation result:
- Passed locally.

## 한국어

상태:
- 2026-04-30 완료.

목적:
- local smoke test와 Kaggle full run이 같이 사용할 1단계 공통 package 구조를
  만듭니다.
- runtime/path 차이는 별도 코드베이스가 아니라 config로 처리합니다.

구현한 파일:
- `src/stage1_reimage/__init__.py`
- `src/stage1_reimage/config.py`
- `src/stage1_reimage/paths.py`
- `src/stage1_reimage/runtime.py`
- `src/stage1_reimage/seed.py`
- `scripts/check_scaffold.py`

구현한 동작:
- Stage 1 YAML config section을 로드하고 검증합니다.
- config에서 local/Kaggle path 객체를 명시적으로 만듭니다.
- 요청 시 설정된 output directory를 생성합니다.
- config에 따라 runtime device를 선택합니다.
- 가능한 경우 Python, NumPy, PyTorch seed를 설정합니다.
- image data loading이나 training 없이 local import/config smoke check를 실행합니다.

범위 제한:
- 이 gate에서는 `.dat` image loading을 구현하지 않았습니다.
- label, split, normalization, model, training, evaluation, Grad-CAM 코드도
  아직 구현하지 않았습니다.
- 다음 구현 gate는 그대로 `1-I2. Data Loading Implementation`입니다.

검증 명령:

```bash
python scripts/check_scaffold.py --config configs/env_local.yaml --create-output-dirs
```

검증 결과:
- 로컬에서 통과했습니다.
