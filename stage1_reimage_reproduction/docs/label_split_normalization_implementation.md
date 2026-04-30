# 1-I3 Label, Split, and Normalization Implementation

## English

Status:
- Completed on 2026-04-30.

Purpose:
- Implement horizon-specific labels for `Ret_5d`, `Ret_20d`, and `Ret_60d`.
- Implement the Stage 1 train/validation/test split.
- Implement train-only pixel normalization metadata.

Implemented files:
- `src/stage1_reimage/data/label_split.py`
- `scripts/check_label_split_normalization.py`
- Updated `configs/env_local.yaml`
- Updated `configs/env_kaggle.yaml`

Implemented behavior:
- Horizon names:
  - `stage1_i20_r5` -> `Ret_5d`
  - `stage1_i20_r20` -> `Ret_20d`
  - `stage1_i20_r60` -> `Ret_60d`
- Binary label rule:
  - `label = 1` if selected target return `> 0`.
  - `label = 0` otherwise.
- Missing target returns are dropped separately per horizon.
- Train/validation pool:
  - 1993-2000.
- Test period:
  - 2001-2019.
- Train/validation split:
  - deterministic non-stratified 70/30 split.
  - split seed `42`.
  - exact train count uses `round(num_train_val_rows * 0.70)`.
- Pixel normalization:
  - fit scalar mean/std on training images only.
  - apply per horizon because filtered rows differ by target.

Validation command:

```bash
python scripts/check_label_split_normalization.py --config configs/env_local.yaml --normalization-max-images 2048
```

Validation result:
- Passed locally.
- Split counts:
  - `stage1_i20_r5`: train 553,554, validation 237,237, test 1,399,933.
  - `stage1_i20_r20`: train 550,736, validation 236,029, test 1,393,845.
  - `stage1_i20_r60`: train 542,499, validation 232,499, test 1,376,215.
- Wrote smoke-check `split_summary.json` and `normalization.json` under
  `outputs/metrics/<horizon>/`.
- The local validation used 2,048 training images per horizon for normalization
  speed, so those normalization JSON files are smoke-check metadata, not final
  reproduction statistics.

Scope limits:
- No model, training loop, evaluation metrics, prediction CSV, or Grad-CAM code
  was implemented in this gate.
- Full train-only normalization for reproduction should be run on Kaggle with
  `--normalization-max-images 0`.

## 한국어

상태:
- 2026-04-30 완료.

목적:
- `Ret_5d`, `Ret_20d`, `Ret_60d` horizon별 label을 구현합니다.
- Stage 1 train/validation/test split을 구현합니다.
- training split만 사용한 pixel normalization metadata를 구현합니다.

구현한 파일:
- `src/stage1_reimage/data/label_split.py`
- `scripts/check_label_split_normalization.py`
- `configs/env_local.yaml` 업데이트
- `configs/env_kaggle.yaml` 업데이트

구현한 동작:
- horizon 이름:
  - `stage1_i20_r5` -> `Ret_5d`
  - `stage1_i20_r20` -> `Ret_20d`
  - `stage1_i20_r60` -> `Ret_60d`
- Binary label rule:
  - 선택한 target return이 `> 0`이면 `label = 1`.
  - 그 외에는 `label = 0`.
- missing target return은 horizon별로 따로 제거합니다.
- Train/validation pool:
  - 1993-2000.
- Test period:
  - 2001-2019.
- Train/validation split:
  - deterministic non-stratified 70/30 split.
  - split seed `42`.
  - 정확한 train count는 `round(num_train_val_rows * 0.70)`로 계산합니다.
- Pixel normalization:
  - training image만 사용해 scalar mean/std를 fitting합니다.
  - target별 filtering row가 다르므로 horizon별로 따로 저장합니다.

검증 명령:

```bash
python scripts/check_label_split_normalization.py --config configs/env_local.yaml --normalization-max-images 2048
```

검증 결과:
- 로컬에서 통과했습니다.
- Split count:
  - `stage1_i20_r5`: train 553,554, validation 237,237, test 1,399,933.
  - `stage1_i20_r20`: train 550,736, validation 236,029, test 1,393,845.
  - `stage1_i20_r60`: train 542,499, validation 232,499, test 1,376,215.
- `outputs/metrics/<horizon>/` 아래 smoke-check `split_summary.json`과
  `normalization.json`을 작성했습니다.
- 로컬 검증은 속도를 위해 horizon별 training image 2,048개만 사용했으므로,
  이 normalization JSON은 최종 재현 통계가 아니라 smoke-check metadata입니다.

범위 제한:
- 이 gate에서는 model, training loop, evaluation metric, prediction CSV,
  Grad-CAM 코드를 구현하지 않았습니다.
- 최종 재현용 full train-only normalization은 Kaggle에서
  `--normalization-max-images 0`으로 실행해야 합니다.
