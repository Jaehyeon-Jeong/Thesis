# 2-I1 Shared Stage 2 Config/Code Scaffold

## English

Status: complete

Implemented:
- Added local/Kaggle configs with the same schema:
  - `stage2_btc_extension/configs/env_local.yaml`
  - `stage2_btc_extension/configs/env_kaggle.yaml`
- Added shared Stage 2 helper modules:
  - `stage2_btc_extension/src/stage2_btc/config.py`
  - `stage2_btc_extension/src/stage2_btc/paths.py`
  - `stage2_btc_extension/src/stage2_btc/runtime.py`
  - `stage2_btc_extension/src/stage2_btc/seed.py`
- Updated package exports and README files.
- Updated the Stage 2 Kaggle runner draft to patch both `paths.source_file` and
  `data.source_file`.

Key fixed defaults:
- Stage 2 default `batch_size` is `128`.
- Strict baseline defaults keep mixed precision off and DataParallel off.
- BTC source file is local-fixed in `env_local.yaml` and auto-detected under
  `/kaggle/input` in `env_kaggle.yaml`.
- Image specs and I5/I20/I60 model choices are stored in config.

Verification:
- `python -m compileall stage2_btc_extension/src/stage2_btc`
- Local config load, I20 image/model config lookup, experiment-name generation,
  and local BTC source-file existence check.

Detailed report:
- [Stage 2 shared config/code scaffold](../docs/stage2_config_code_scaffold.md)

## 한국어

상태: 완료

구현한 내용:
- 같은 schema를 가진 local/Kaggle config를 추가했습니다.
  - `stage2_btc_extension/configs/env_local.yaml`
  - `stage2_btc_extension/configs/env_kaggle.yaml`
- Stage 2 공통 helper module을 추가했습니다.
  - `stage2_btc_extension/src/stage2_btc/config.py`
  - `stage2_btc_extension/src/stage2_btc/paths.py`
  - `stage2_btc_extension/src/stage2_btc/runtime.py`
  - `stage2_btc_extension/src/stage2_btc/seed.py`
- package export와 README를 업데이트했습니다.
- Stage 2 Kaggle runner draft에서 `paths.source_file`과 `data.source_file`을
  함께 patch하도록 수정했습니다.

고정한 기본값:
- Stage 2 기본 `batch_size`는 `128`입니다.
- strict baseline 기본값은 mixed precision off, DataParallel off입니다.
- local config는 BTC CSV 경로를 고정하고, Kaggle config는 `/kaggle/input` 아래에서
  auto-detect하도록 둡니다.
- image spec과 I5/I20/I60 model 선택 정보는 config에 저장했습니다.

검증:
- `python -m compileall stage2_btc_extension/src/stage2_btc`
- local config load, I20 image/model config 조회, experiment name 생성, local BTC
  source file 존재 확인.

상세 보고:
- [Stage 2 shared config/code scaffold](../docs/stage2_config_code_scaffold.md)
