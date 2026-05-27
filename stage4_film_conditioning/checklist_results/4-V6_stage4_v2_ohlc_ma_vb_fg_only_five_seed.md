# 4-V6. Stage 4 v2 OHLC_MA_VB + F&G-Only Context + Full FiLM, Five Seeds

Status: five-seed Kaggle runner ready; full result pending.

## Experiment

```text
priority: 4-V6
image_window: I60
return_horizon: R20
image_spec: ohlc_ma_vb
context_window: 60
context scope: F&G-only
context features:
  - fg_value
  - fg_mean_60
  - fg_delta_60
  - fg_std_60
fusion method: film_full
run_seeds: 42, 43, 44, 45, 46
```

## Why This Test Comes Before Changing FiLM Structure

F&G is not derived from the chart/OHLCV data. It is external sentiment/regime
context. Therefore it is the cleanest candidate for an incremental signal on
top of the strongest visual baseline.

`4-V6` asks:

```text
Does external F&G context improve the already-strong I60/R20/ohlc_ma_vb visual
baseline?
```

This test should come before bounded/last-block FiLM because it still uses the
same FiLM-full structure as the previous context tests. That keeps the
comparison clean: first close the question of whether the context signal helps,
then change the FiLM architecture if instability remains.

## Interpretation Rule

- If `4-V6` beats the Stage 2 `ohlc_ma_vb` five-seed baseline, then F&G provides
  incremental external context.
- If `4-V6` is flat or worse, then F&G does not help under the current FiLM-full
  injection even when the visual baseline is strong.
- If `4-V6` is unstable across seeds, bounded/last-block FiLM becomes the next
  architecture-level fix.

## Output Naming

This run uses the F&G-only suffix with the strong visual image spec:

```text
context.feature_set_name = fg_only
stage4_model.experiment_suffix = fg_only
```

Expected output names:

```text
stage4_film_full_i60_ohlc_ma_vb_r20_c60_fg_only
stage4_context_i60_ohlc_ma_vb_r20_c60_fg_only
```

This is separate from the plain-OHLC F&G-only run because the image spec is part
of the output name.

## Runner

Use:

```text
notebooks/kaggle_stage4_v2_p7_ohlc_ma_vb_fg_only_film_full_five_seed_one_cell.md
```

The runner defaults to five seeds and `SAVE_BACKUP_ZIPS=False`.

# 4-V6. Stage 4 v2 OHLC_MA_VB + F&G-only Context + Full FiLM, Five Seeds

상태: five-seed Kaggle runner 준비 완료, full 결과 대기 중.

## 실험

```text
priority: 4-V6
image_window: I60
return_horizon: R20
image_spec: ohlc_ma_vb
context_window: 60
context scope: F&G-only
context features:
  - fg_value
  - fg_mean_60
  - fg_delta_60
  - fg_std_60
fusion method: film_full
run_seeds: 42, 43, 44, 45, 46
```

## FiLM 구조를 바꾸기 전에 이 실험을 하는 이유

F&G는 chart/OHLCV에서 파생된 정보가 아닙니다. 외부 sentiment/regime
context입니다. 따라서 가장 강한 visual baseline 위에서 incremental signal을
줄 수 있는지 확인하기에 가장 깨끗한 후보입니다.

`4-V6`의 질문은 다음입니다.

```text
외부 F&G context가 이미 강한 I60/R20/ohlc_ma_vb visual baseline을 추가로
개선하는가?
```

이 실험은 bounded/last-block FiLM보다 먼저 해야 합니다. 아직 같은 FiLM-full
구조를 유지하기 때문에 비교가 깨끗합니다. 먼저 context signal이 도움이 되는지
닫고, 그 다음 seed instability가 남으면 FiLM architecture를 바꾸는 순서가 맞습니다.

## 결과 해석 규칙

- `4-V6`가 Stage 2 `ohlc_ma_vb` five-seed baseline을 이기면, F&G는 incremental
  external context signal을 제공합니다.
- `4-V6`가 비슷하거나 낮으면, 현재 FiLM-full 주입 방식에서는 강한 visual baseline
  위에서도 F&G가 추가 성능을 주지 못한 것입니다.
- `4-V6`가 seed별로 불안정하면, bounded/last-block FiLM이 다음 architecture-level
  수정입니다.

## Output Naming

이 run은 강한 visual image spec과 F&G-only suffix를 함께 사용합니다.

```text
context.feature_set_name = fg_only
stage4_model.experiment_suffix = fg_only
```

예상 output 이름:

```text
stage4_film_full_i60_ohlc_ma_vb_r20_c60_fg_only
stage4_context_i60_ohlc_ma_vb_r20_c60_fg_only
```

image spec이 output 이름에 포함되므로 plain-OHLC F&G-only run과 분리됩니다.

## Runner

사용할 파일:

```text
notebooks/kaggle_stage4_v2_p7_ohlc_ma_vb_fg_only_film_full_five_seed_one_cell.md
```

runner는 기본값이 five seeds이고 `SAVE_BACKUP_ZIPS=False`입니다.
