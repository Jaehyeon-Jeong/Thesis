# 4-V5. Stage 4 v2 OHLC + All Structured Context + Full FiLM, Five Seeds

Status: five-seed Kaggle runner ready; full result pending.

## Experiment

```text
priority: 4-V5
image_window: I60
return_horizon: R20
image_spec: ohlc
context_window: 60
context scope: all structured context
context features:
  - fg_value
  - fg_mean_60
  - fg_delta_60
  - fg_std_60
  - bb_percent_b_60
  - bb_bandwidth_60
  - mfi_60
  - rv_60
fusion method: film_full
run_seeds: 42, 43, 44, 45, 46
```

## Why This Test Comes Next

The earlier `4-V2` all-context diagnostic was only seed `42` and showed partial
recovery over the plain OHLC control:

```text
accuracy: 0.5725
ROC-AUC:  0.5573
```

But the two isolated five-seed tests were weak:

- `4-V3` F&G-only: accuracy mean `0.5586`, ROC-AUC mean `0.5523`.
- `4-V4` technical-only: accuracy mean `0.5603`, ROC-AUC mean `0.5546`.

Therefore `4-V5` checks whether all-context performance was a real combination
effect or just a favorable seed-42 result.

## Interpretation Rule

- If `4-V5` is strong across five seeds, then F&G and technical features may
  work jointly even though each isolated path was weak.
- If `4-V5` falls near `4-V3`/`4-V4`, then the earlier seed-42 gain was likely
  not robust.
- If `4-V5` still underperforms `4-V0`, then the strong `ohlc_ma_vb` visual
  encoding remains the main baseline to beat.

## Output Naming

This run uses an explicit feature-set suffix:

```text
context.feature_set_name = all_context
stage4_model.experiment_suffix = all_context
```

Therefore outputs are separated from the earlier seed-42 `4-V2` diagnostic:

```text
stage4_film_full_i60_ohlc_r20_c60_all_context
stage4_context_i60_ohlc_r20_c60_all_context
```

This forces a clean five-seed all-context run instead of silently reusing or
overwriting the old seed-42 output.

## Runner

Use:

```text
notebooks/kaggle_stage4_v2_p6_ohlc_all_context_film_full_five_seed_one_cell.md
```

The runner defaults to five seeds and `SAVE_BACKUP_ZIPS=False`.

# 4-V5. Stage 4 v2 OHLC + All Structured Context + Full FiLM, Five Seeds

상태: five-seed Kaggle runner 준비 완료, full 결과 대기 중.

## 실험

```text
priority: 4-V5
image_window: I60
return_horizon: R20
image_spec: ohlc
context_window: 60
context scope: all structured context
context features:
  - fg_value
  - fg_mean_60
  - fg_delta_60
  - fg_std_60
  - bb_percent_b_60
  - bb_bandwidth_60
  - mfi_60
  - rv_60
fusion method: film_full
run_seeds: 42, 43, 44, 45, 46
```

## 이 실험이 다음 순서인 이유

이전 `4-V2` all-context diagnostic은 seed `42` 하나였고, plain OHLC control
대비 일부 회복을 보였습니다.

```text
accuracy: 0.5725
ROC-AUC:  0.5573
```

하지만 분리해서 본 five-seed 실험은 둘 다 약했습니다.

- `4-V3` F&G-only: accuracy mean `0.5586`, ROC-AUC mean `0.5523`.
- `4-V4` technical-only: accuracy mean `0.5603`, ROC-AUC mean `0.5546`.

따라서 `4-V5`는 all-context 성능이 실제 조합 효과였는지, 아니면 seed-42가
좋았던 것인지 확인하는 실험입니다.

## 결과 해석 규칙

- `4-V5`가 five-seed에서도 강하면, F&G와 technical feature가 각각은 약해도
  조합으로는 도움이 될 가능성이 있습니다.
- `4-V5`가 `4-V3`/`4-V4` 근처로 떨어지면, 이전 seed-42 개선은 robust하지
  않았다고 해석합니다.
- `4-V5`도 `4-V0`보다 낮으면, 강한 `ohlc_ma_vb` visual encoding이 여전히
  가장 중요한 baseline입니다.

## Output Naming

이 run은 명시적인 feature-set suffix를 사용합니다.

```text
context.feature_set_name = all_context
stage4_model.experiment_suffix = all_context
```

따라서 output은 이전 seed-42 `4-V2` diagnostic과 분리됩니다.

```text
stage4_film_full_i60_ohlc_r20_c60_all_context
stage4_context_i60_ohlc_r20_c60_all_context
```

이렇게 해야 예전 seed-42 output을 조용히 재사용하거나 덮어쓰지 않고, 깨끗한
five-seed all-context run으로 실행됩니다.

## Runner

사용할 파일:

```text
notebooks/kaggle_stage4_v2_p6_ohlc_all_context_film_full_five_seed_one_cell.md
```

runner는 기본값이 five seeds이고 `SAVE_BACKUP_ZIPS=False`입니다.
