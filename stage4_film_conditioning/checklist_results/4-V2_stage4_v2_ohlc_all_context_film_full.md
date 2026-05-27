# 4-V2. Stage 4 v2 OHLC + All Structured Context + Full FiLM

Status: seed `42` diagnostic complete; five-seed repeat not run.

## Experiment

```text
priority: 4-V2
image_window: I60
return_horizon: R20
image_spec: ohlc
context_window: 60
context scope: all structured context
context features: F&G value/mean/delta/std, Bollinger %B/bandwidth, MFI, realized volatility
fusion method: film_full
run_seed: default 42
```

## Why This Is the Next Test

`4-V0` showed that the Stage 4 v2 visual-only runner can reproduce the strong
Stage 2 `I60/R20/ohlc_ma_vb` seed-42 result. `4-V1` showed that plain
`I60/R20/ohlc` is much weaker and tends to predict Up for almost every test
sample.

Therefore `4-V2` asks the core diagnostic question:

> If MA/VB are removed from the image, can structured market context recover
> useful information through FiLM?

This is the first v2 experiment that should be judged mainly against `4-V1`,
not directly against `4-V0`.

## Comparison Targets

| Target | Accuracy | ROC-AUC | Role |
| --- | ---: | ---: | --- |
| `4-V1` seed-42 `I60/R20/ohlc`, visual-only | `0.5420` | `0.5441` | direct control |
| Stage 2 `I60/R20/ohlc`, five-seed mean | `0.5581` | `0.5602` | robust plain-OHLC reference |
| `4-V0` seed-42 `I60/R20/ohlc_ma_vb`, visual-only | `0.6031` | `0.6170` | strong visual upper reference |
| Stage 2 `I60/R20/ohlc_ma_vb`, five-seed mean | `0.5793` | `0.5849` | robust selected visual baseline |
| Stage 4 v1 `film_full` on `ohlc_ma_vb`, five-seed mean | `0.5510` | `0.5677` | previous all-context FiLM result |

Completed seed-42 result:

| Run | Accuracy | Majority Acc. | ROC-AUC | F1 | Predicted Up Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4-V2` `I60/R20/ohlc + all context + film_full`, seed 42 | `0.5725` | `0.5413` | `0.5573` | `0.6771` | `0.7828` |

This partially recovers performance relative to the plain-OHLC `4-V1` seed-42
control (`0.5420` accuracy, `0.5441` ROC-AUC), but it does not reach the strong
`4-V0` `ohlc_ma_vb` seed-42 visual baseline (`0.6031` accuracy, `0.6170`
ROC-AUC).

## Runner

Use:

```text
notebooks/kaggle_stage4_v2_p3_ohlc_all_context_film_full_one_cell.md
```

The runner executes one method only:

```text
context build -> train film_full -> prediction metrics -> trading metrics
-> Grad-CAM/context/modulation export -> output check -> compact summary
```

`SAVE_BACKUP_ZIPS=False` is the default to avoid Kaggle disk pressure. The full
outputs remain under:

```text
/kaggle/working/stage4_film_conditioning/outputs/stage4
```

Summary outputs are written to:

```text
/kaggle/working/stage4_film_conditioning/reports/tables/stage4_v2_p3_ohlc_all_context_film_full_seed42_run_summary.csv
/kaggle/working/stage4_film_conditioning/reports/tables/stage4_v2_p3_ohlc_all_context_film_full_seed42_run_summary.json
```

## Interpretation Rule

- If `4-V2` is clearly above `4-V1`, then context is helping when the visual
  image no longer already contains MA/VB-style information.
- If `4-V2` is similar to `4-V1`, then all structured context plus current
  full-FiLM still does not recover the missing MA/VB visual information.
- If `4-V2` is below `4-V1`, then current full-FiLM is actively hurting even
  when visual-context redundancy is reduced.

The most important columns after running are:

```text
accuracy
accuracy_minus_stage2_ohlc_mean
roc_auc
roc_auc_minus_stage2_ohlc_mean
predicted_positive_rate
f1
brier_score
long_flat_sharpe_net
long_short_sharpe_net
```

# 4-V2. Stage 4 v2 OHLC + 전체 Structured Context + Full FiLM

상태: seed `42` diagnostic 완료, five-seed 반복은 아직 실행하지 않음.

## 실험

```text
priority: 4-V2
image_window: I60
return_horizon: R20
image_spec: ohlc
context_window: 60
context scope: all structured context
context features: F&G value/mean/delta/std, Bollinger %B/bandwidth, MFI, realized volatility
fusion method: film_full
run_seed: 기본 42
```

## 왜 이 실험이 다음인가

`4-V0`은 Stage 4 v2 visual-only runner가 강한 Stage 2
`I60/R20/ohlc_ma_vb` seed-42 결과를 재현한다는 것을 보여줬습니다. `4-V1`은
plain `I60/R20/ohlc`가 훨씬 약하고, test sample 대부분을 Up으로 예측하는
경향이 있다는 것을 보여줬습니다.

따라서 `4-V2`의 핵심 질문은 다음입니다.

> 이미지에서 MA/VB를 제거했을 때, structured market context가 FiLM을 통해
> 빠진 정보를 보완할 수 있는가?

이 실험은 `4-V0`이 아니라 우선 `4-V1`과 비교해야 합니다.

## 비교 기준

| 기준 | Accuracy | ROC-AUC | 역할 |
| --- | ---: | ---: | --- |
| `4-V1` seed-42 `I60/R20/ohlc`, visual-only | `0.5420` | `0.5441` | 직접 control |
| Stage 2 `I60/R20/ohlc`, five-seed mean | `0.5581` | `0.5602` | robust plain-OHLC 기준 |
| `4-V0` seed-42 `I60/R20/ohlc_ma_vb`, visual-only | `0.6031` | `0.6170` | 강한 visual upper reference |
| Stage 2 `I60/R20/ohlc_ma_vb`, five-seed mean | `0.5793` | `0.5849` | robust selected visual baseline |
| Stage 4 v1 `film_full` on `ohlc_ma_vb`, five-seed mean | `0.5510` | `0.5677` | 이전 all-context FiLM 결과 |

완료된 seed-42 결과:

| Run | Accuracy | Majority Acc. | ROC-AUC | F1 | Predicted Up Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4-V2` `I60/R20/ohlc + all context + film_full`, seed 42 | `0.5725` | `0.5413` | `0.5573` | `0.6771` | `0.7828` |

이 결과는 plain-OHLC `4-V1` seed-42 control (`0.5420` accuracy, `0.5441`
ROC-AUC)보다 일부 회복됐지만, 강한 `4-V0` `ohlc_ma_vb` seed-42 visual
baseline (`0.6031` accuracy, `0.6170` ROC-AUC)에는 아직 미치지 못했습니다.

## Runner

사용할 파일:

```text
notebooks/kaggle_stage4_v2_p3_ohlc_all_context_film_full_one_cell.md
```

runner는 한 방법만 실행합니다.

```text
context build -> film_full train -> prediction metrics -> trading metrics
-> Grad-CAM/context/modulation export -> output check -> compact summary
```

Kaggle disk 압박을 피하기 위해 `SAVE_BACKUP_ZIPS=False`가 기본값입니다. 전체
output은 아래에 남습니다.

```text
/kaggle/working/stage4_film_conditioning/outputs/stage4
```

summary는 아래에 저장됩니다.

```text
/kaggle/working/stage4_film_conditioning/reports/tables/stage4_v2_p3_ohlc_all_context_film_full_seed42_run_summary.csv
/kaggle/working/stage4_film_conditioning/reports/tables/stage4_v2_p3_ohlc_all_context_film_full_seed42_run_summary.json
```

## 결과 해석 규칙

- `4-V2`가 `4-V1`보다 확실히 높으면, 이미지가 MA/VB 정보를 갖고 있지 않을 때
  context가 도움을 준다는 근거가 됩니다.
- `4-V2`가 `4-V1`과 비슷하면, 전체 structured context와 현재 full-FiLM이 빠진
  MA/VB visual information을 회복하지 못했다는 뜻입니다.
- `4-V2`가 `4-V1`보다 낮으면, visual-context 중복을 줄였는데도 현재 full-FiLM이
  오히려 해롭다는 뜻입니다.

실행 후 가장 먼저 볼 column은 다음입니다.

```text
accuracy
accuracy_minus_stage2_ohlc_mean
roc_auc
roc_auc_minus_stage2_ohlc_mean
predicted_positive_rate
f1
brier_score
long_flat_sharpe_net
long_short_sharpe_net
```
