# 4-V5. Stage 4 v2 OHLC + All Structured Context + Full FiLM, Five Seeds

Status: planned; runner not yet generated.

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

# 4-V5. Stage 4 v2 OHLC + All Structured Context + Full FiLM, Five Seeds

상태: 계획 확정, runner는 아직 생성 전.

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
