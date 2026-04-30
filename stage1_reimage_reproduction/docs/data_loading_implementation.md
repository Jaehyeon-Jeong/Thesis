# 1-I2 Data Loading Implementation

## English

Status:
- Completed on 2026-04-30.

Purpose:
- Implement lazy/memmap loading for the public `monthly_20d` Stage 1 data.
- Validate row alignment between `.dat` image shards and `.feather` label files.
- Return image tensors with the Stage 1 shape contract.

Implemented files:
- `src/stage1_reimage/data/__init__.py`
- `src/stage1_reimage/data/monthly20.py`
- `scripts/check_data_loading.py`

Implemented behavior:
- Deterministic discovery for `1993..2019`.
- Image file pattern:
  `20d_month_has_vb_20_ma_{year}_images.dat`.
- Label file pattern:
  `20d_month_has_vb_20_ma_{year}_labels_w_delay.feather`.
- `.dat` image files are read as `uint8` memmaps.
- Per-sample image tensor is returned as `(1, 64, 60)` `float32`, scaled to
  `[0, 1]`.
- Required label columns are validated.
- Image row count is inferred from file size and checked against label rows.
- Label `Date` values are checked against the file year.
- Dataset samples return:
  - `image`: model input tensor.
  - `metadata`: original label columns plus `year` and `local_row`.

Scope limits:
- Horizon-specific labels are not constructed in this gate.
- Train/validation/test split and train-only normalization are not implemented
  in this gate.
- DataLoader worker settings are not implemented in this gate.
- No model, training, evaluation, or Grad-CAM code is implemented here.

Validation command:

```bash
python scripts/check_data_loading.py --config configs/env_local.yaml --sample-indices 0 -1
```

Validation result:
- Passed locally.
- Confirmed 27 shards.
- Confirmed 2,196,994 total raw image/label rows before horizon-specific NaN
  filtering.
- Confirmed sample image shape `(1, 64, 60)` and value range `[0, 1]`.

## 한국어

상태:
- 2026-04-30 완료.

목적:
- public `monthly_20d` Stage 1 data를 lazy/memmap 방식으로 읽습니다.
- `.dat` image shard와 `.feather` label file의 row alignment를 검증합니다.
- Stage 1 shape contract에 맞는 image tensor를 반환합니다.

구현한 파일:
- `src/stage1_reimage/data/__init__.py`
- `src/stage1_reimage/data/monthly20.py`
- `scripts/check_data_loading.py`

구현한 동작:
- `1993..2019` deterministic discovery.
- image file pattern:
  `20d_month_has_vb_20_ma_{year}_images.dat`.
- label file pattern:
  `20d_month_has_vb_20_ma_{year}_labels_w_delay.feather`.
- `.dat` image file은 `uint8` memmap으로 읽습니다.
- sample image tensor는 `(1, 64, 60)` `float32`이고 `[0, 1]`로 scaling합니다.
- 필수 label column을 검증합니다.
- image row count를 file size에서 추정하고 label row 수와 비교합니다.
- label `Date` 값이 해당 file year 안에 있는지 확인합니다.
- dataset sample은 다음을 반환합니다.
  - `image`: model input tensor.
  - `metadata`: 원본 label column과 `year`, `local_row`.

범위 제한:
- horizon-specific label은 이 gate에서 만들지 않았습니다.
- train/validation/test split과 train-only normalization은 아직 구현하지 않았습니다.
- DataLoader worker setting은 아직 구현하지 않았습니다.
- model, training, evaluation, Grad-CAM 코드도 여기서는 구현하지 않았습니다.

검증 명령:

```bash
python scripts/check_data_loading.py --config configs/env_local.yaml --sample-indices 0 -1
```

검증 결과:
- 로컬에서 통과했습니다.
- shard 27개를 확인했습니다.
- horizon별 NaN filtering 전 raw image/label row 2,196,994개를 확인했습니다.
- sample image shape `(1, 64, 60)`과 value range `[0, 1]`를 확인했습니다.
