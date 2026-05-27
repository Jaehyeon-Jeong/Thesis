# 4-V4. Stage 4 v2 OHLC + Technical-Only Context + Full FiLM

Status: five-seed Kaggle runner ready; full result pending.

## Experiment

```text
priority: 4-V4
image_window: I60
return_horizon: R20
image_spec: ohlc
context_window: 60
context scope: technical-only
context features:
  - bb_percent_b_60
  - bb_bandwidth_60
  - mfi_60
  - rv_60
fusion method: film_full
run_seeds: 42, 43, 44, 45, 46
```

## Why This Test Exists

`4-V3` showed that F&G-only context was not enough to robustly improve over the
plain OHLC baseline. The next question is whether the useful part of Stage 4
context is the OHLCV-derived technical information instead.

`4-V4` isolates the technical-context path:

- It keeps the image plain `ohlc`.
- It removes Fear & Greed from the primary context vector.
- It keeps only Bollinger, MFI, and realized-volatility features derived from
  OHLCV data available at image end date `t`.

This directly tests the duplicate-feature hypothesis: if technical-only context
helps plain OHLC but still does not match `ohlc_ma_vb`, then the MA/VB image is
likely a stronger representation of similar information than the current FiLM
context path.

## Output Naming

This run uses an explicit feature-set suffix:

```text
context.feature_set_name = technical_only
stage4_model.experiment_suffix = technical_only
```

Therefore outputs are separated from `4-V2` and `4-V3`:

```text
stage4_film_full_i60_ohlc_r20_c60_technical_only
stage4_context_i60_ohlc_r20_c60_technical_only
```

This avoids overwriting the all-context and F&G-only results.

## Runner

Use:

```text
notebooks/kaggle_stage4_v2_p5_ohlc_technical_only_film_full_five_seed_one_cell.md
```

The runner defaults to five seeds and `SAVE_BACKUP_ZIPS=False`.

## Interpretation Rule

Compare in this order:

1. `4-V4` vs `4-V1`: does technical-only context beat plain OHLC?
2. `4-V4` vs `4-V3`: do BB/MFI/RV help more than F&G-only context?
3. `4-V4` vs `4-V2`: does adding F&G to technical context help or dilute the
   signal?
4. `4-V4` vs `4-V0`: how far is technical-only FiLM from the strong
   `ohlc_ma_vb` visual baseline?

If `4-V4` improves over `4-V1`, then technical context carries useful signal
outside the plain OHLC image. If `4-V4` approaches `4-V0`, then FiLM can recover
some of the information that MA/VB images provide. If it remains below `4-V0`,
then the chart overlay is still the stronger encoding.

# 4-V4. Stage 4 v2 OHLC + technical-only Context + Full FiLM

상태: five-seed Kaggle runner 준비 완료, full 결과 대기 중.

## 실험

```text
priority: 4-V4
image_window: I60
return_horizon: R20
image_spec: ohlc
context_window: 60
context scope: technical-only
context features:
  - bb_percent_b_60
  - bb_bandwidth_60
  - mfi_60
  - rv_60
fusion method: film_full
run_seeds: 42, 43, 44, 45, 46
```

## 이 실험의 이유

`4-V3` 결과에서 F&G-only context는 plain OHLC baseline을 안정적으로 개선하지
못했습니다. 다음 질문은 Stage 4 context에서 실제로 도움이 될 수 있는 부분이
OHLCV에서 파생한 technical information인지 확인하는 것입니다.

`4-V4`는 technical-context 경로만 분리합니다.

- 이미지는 plain `ohlc`로 유지합니다.
- primary context vector에서 Fear & Greed를 제거합니다.
- image end date `t`까지 사용 가능한 OHLCV로 계산한 Bollinger, MFI,
  realized-volatility feature만 유지합니다.

이 실험은 duplicate-feature 가설을 직접 확인합니다. Technical-only context가
plain OHLC를 개선하지만 `ohlc_ma_vb`에는 못 미친다면, 현재 FiLM context path보다
MA/VB chart overlay가 비슷한 정보를 더 강하게 표현한다는 해석이 가능합니다.

## Output Naming

이 run은 명시적인 feature-set suffix를 사용합니다.

```text
context.feature_set_name = technical_only
stage4_model.experiment_suffix = technical_only
```

따라서 output은 `4-V2`, `4-V3`와 분리됩니다.

```text
stage4_film_full_i60_ohlc_r20_c60_technical_only
stage4_context_i60_ohlc_r20_c60_technical_only
```

이렇게 해야 all-context와 F&G-only 결과를 덮어쓰지 않습니다.

## Runner

사용할 파일:

```text
notebooks/kaggle_stage4_v2_p5_ohlc_technical_only_film_full_five_seed_one_cell.md
```

runner는 기본값이 five seeds이고 `SAVE_BACKUP_ZIPS=False`입니다.

## 결과 해석 규칙

비교 순서는 다음입니다.

1. `4-V4` vs `4-V1`: technical-only context가 plain OHLC를 이기는가?
2. `4-V4` vs `4-V3`: BB/MFI/RV가 F&G-only context보다 더 도움이 되는가?
3. `4-V4` vs `4-V2`: F&G를 technical context에 추가하는 것이 도움이 되는가,
   아니면 signal을 희석하는가?
4. `4-V4` vs `4-V0`: technical-only FiLM이 강한 `ohlc_ma_vb` visual baseline과
   얼마나 차이나는가?

`4-V4`가 `4-V1`보다 좋아지면 plain OHLC 이미지 밖의 technical context가 유효한
signal을 가진다는 근거가 됩니다. `4-V4`가 `4-V0`에 가까워지면 FiLM이 MA/VB
이미지에 들어 있던 일부 정보를 회복한다고 볼 수 있습니다. 그래도 `4-V0`보다
낮으면 chart overlay가 아직 더 강한 encoding이라는 해석이 맞습니다.
