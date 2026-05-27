# 4-V3. Stage 4 v2 OHLC + F&G-Only Context + Full FiLM

Status: five-seed Kaggle run complete; result reviewed.

## Experiment

```text
priority: 4-V3
image_window: I60
return_horizon: R20
image_spec: ohlc
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

## Why This Test Exists

`4-V2` showed that `ohlc + all structured context + film_full` can improve the
seed-42 plain-OHLC control. The next question is whether the improvement comes
from image-external market sentiment/regime information or from OHLCV-derived
technical indicators.

`4-V3` isolates the external market-context path:

- It keeps the image plain `ohlc`.
- It removes Bollinger, MFI, and realized volatility from the context vector.
- It keeps only Fear & Greed features, which are not drawn directly into the
  chart image.

## Output Naming

This run uses an explicit feature-set suffix:

```text
context.feature_set_name = fg_only
stage4_model.experiment_suffix = fg_only
```

Therefore outputs are separated from `4-V2`:

```text
stage4_film_full_i60_ohlc_r20_c60_fg_only
stage4_context_i60_ohlc_r20_c60_fg_only
```

This avoids overwriting the all-context `4-V2` results.

## Runner

Use:

```text
notebooks/kaggle_stage4_v2_p4_ohlc_fg_only_film_full_five_seed_one_cell.md
```

The runner defaults to five seeds and `SAVE_BACKUP_ZIPS=False`.

## Interpretation Rule

Compare in this order:

1. `4-V3` vs `4-V1`: does F&G-only context beat plain OHLC?
2. `4-V3` vs `4-V2`: is F&G alone enough, or did the all-context gain come from
   BB/MFI/RV?
3. `4-V3` vs `4-V0`: how far is F&G-only FiLM from the strong `ohlc_ma_vb`
   visual baseline?

If `4-V3` improves over `4-V1`, then F&G carries useful external market-context
signal. If `4-V2` is better than `4-V3`, then technical context likely explains
part of the all-context gain.

## Observed Result

Exported result tables:

```text
stage4_v2_p4_ohlc_fg_only_film_full_five_seed_seed_results.csv
stage4_v2_p4_ohlc_fg_only_film_full_five_seed_mean_std_results.csv
```

Five-seed summary:

```text
accuracy_mean: 0.558640
accuracy_std:  0.018433
roc_auc_mean:  0.552286
roc_auc_std:   0.016494
accuracy_mean_minus_stage2_ohlc_mean:  0.000555
roc_auc_mean_minus_stage2_ohlc_mean:  -0.007932
predicted_positive_rate_mean: 0.771131
```

Interpretation:

- F&G-only FiLM is not useless, but it does not materially improve over the
  plain OHLC baseline.
- The average accuracy is nearly identical to the Stage 2 OHLC baseline mean.
- ROC-AUC is lower than the Stage 2 OHLC baseline mean.
- The high predicted-positive rate shows remaining Up-class bias.
- Seed `45` was strong, but the five-seed average is not robust enough to claim
  that F&G-only context is the useful Stage 4 signal.

# 4-V3. Stage 4 v2 OHLC + F&G-only Context + Full FiLM

상태: five-seed Kaggle run 완료, 결과 검토 완료.

## 실험

```text
priority: 4-V3
image_window: I60
return_horizon: R20
image_spec: ohlc
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

## 이 실험의 이유

`4-V2`는 seed-42 기준 `ohlc + all structured context + film_full`이 plain-OHLC
control보다 좋아질 수 있음을 보여줬습니다. 다음 질문은 그 개선이 이미지 밖의
market sentiment/regime 정보 때문인지, 아니면 OHLCV에서 파생한 기술적 지표
때문인지입니다.

`4-V3`는 외부 market-context 경로만 분리합니다.

- 이미지는 plain `ohlc`로 유지합니다.
- Bollinger, MFI, realized volatility는 context vector에서 제거합니다.
- chart image에 직접 그려지지 않는 Fear & Greed feature만 유지합니다.

## Output Naming

이 run은 명시적인 feature-set suffix를 사용합니다.

```text
context.feature_set_name = fg_only
stage4_model.experiment_suffix = fg_only
```

따라서 output은 `4-V2`와 분리됩니다.

```text
stage4_film_full_i60_ohlc_r20_c60_fg_only
stage4_context_i60_ohlc_r20_c60_fg_only
```

이렇게 해야 all-context `4-V2` 결과를 덮어쓰지 않습니다.

## Runner

사용할 파일:

```text
notebooks/kaggle_stage4_v2_p4_ohlc_fg_only_film_full_five_seed_one_cell.md
```

runner는 기본값이 five seeds이고 `SAVE_BACKUP_ZIPS=False`입니다.

## 결과 해석 규칙

비교 순서는 다음입니다.

1. `4-V3` vs `4-V1`: F&G-only context가 plain OHLC를 이기는가?
2. `4-V3` vs `4-V2`: F&G만으로 충분한가, 아니면 all-context 개선이 BB/MFI/RV
   때문인가?
3. `4-V3` vs `4-V0`: F&G-only FiLM이 강한 `ohlc_ma_vb` visual baseline과
   얼마나 차이나는가?

`4-V3`가 `4-V1`보다 좋아지면 F&G가 외부 market-context signal을 제공한다는
근거가 됩니다. `4-V2`가 `4-V3`보다 좋으면 all-context 개선에는 technical
context가 일부 기여했을 가능성이 커집니다.

## 관찰된 결과

Export 결과표:

```text
stage4_v2_p4_ohlc_fg_only_film_full_five_seed_seed_results.csv
stage4_v2_p4_ohlc_fg_only_film_full_five_seed_mean_std_results.csv
```

Five-seed summary:

```text
accuracy_mean: 0.558640
accuracy_std:  0.018433
roc_auc_mean:  0.552286
roc_auc_std:   0.016494
accuracy_mean_minus_stage2_ohlc_mean:  0.000555
roc_auc_mean_minus_stage2_ohlc_mean:  -0.007932
predicted_positive_rate_mean: 0.771131
```

해석:

- F&G-only FiLM이 완전히 무의미하다고 보기는 어렵지만, plain OHLC baseline을
  실질적으로 개선하지 못했습니다.
- 평균 accuracy는 Stage 2 OHLC baseline mean과 거의 같습니다.
- ROC-AUC는 Stage 2 OHLC baseline mean보다 낮습니다.
- predicted-positive rate가 높아서 Up-class bias가 아직 남아 있습니다.
- seed `45`는 강했지만, five-seed 평균 기준으로 F&G-only context가 Stage 4의
  핵심 signal이라고 주장하기에는 부족합니다.
