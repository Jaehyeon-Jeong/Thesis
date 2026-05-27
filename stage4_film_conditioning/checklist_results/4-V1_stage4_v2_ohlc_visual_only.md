# 4-V1. Stage 4 v2 OHLC Visual-Only Control

Status: complete for seed `42`.

## Experiment

```text
priority: 4-V1
image_window: I60
return_horizon: R20
image_spec: ohlc
context: none
model: Stage 2 Stock_CNN I60 visual-only baseline
seeds: default [42], configurable
```

## Why This Control Exists

`4-V0` confirmed that the selected `I60/R20/ohlc_ma_vb` visual-only setup can
reproduce the strong Stage 2 seed-42 result. `4-V1` removes the MA and volume
bar overlays from the image and keeps everything else visual-only.

This is the second diagnostic control before changing FiLM again:

- It measures how much the strong Stage 2 result depends on the `ohlc_ma_vb`
  image encoding.
- It prepares the duplicate-feature test in `4-V2`, where the image is plain
  `ohlc` and BB/MFI/RV/F&G enter only through the context pathway.
- It does not use FiLM, gating, concat context, F&G, Bollinger, MFI, or realized
  volatility.

## Existing Reference

From the Stage 2 selected `I20/I60, R20, four image specs, five seeds` table:

| Experiment | Accuracy Mean | ROC-AUC Mean | Note |
| --- | ---: | ---: | --- |
| `I60/R20/ohlc` | `0.5581` | `0.5602` | 4-V1 reference |
| `I60/R20/ohlc_ma_vb` | `0.5793` | `0.5849` | selected visual baseline |
| Stage 4 v1 `film_full` | `0.5510` | `0.5677` | context FiLM v1 |

Completed `4-V1` seed-42 result:

| Run | Accuracy | Majority Acc. | ROC-AUC | F1 | Predicted Up Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4-V1` `I60/R20/ohlc`, visual-only, seed 42 | `0.5420` | `0.5413` | `0.5441` | `0.6956` | `0.9632` |

The model is almost at majority-class accuracy and predicts Up for most test
samples. This confirms that plain OHLC alone is much weaker than
`ohlc_ma_vb`, and that MA/VB visual overlays carried useful signal in the Stage
2 selected baseline.

## Runner

Use:

```text
notebooks/kaggle_stage4_v2_p2_ohlc_visual_only_one_cell.md
```

The runner deliberately calls the Stage 2 visual-only training/evaluation
scripts. That is correct because `4-V1` is not a Stage 4 context model; it is a
control to establish what plain OHLC images can do before context is re-added.

## Expected Interpretation

If `4-V1` is materially below `4-V0`, then the MA/VB visual overlays are carrying
useful information and the v2 context experiments should not compare plain OHLC
models directly against `ohlc_ma_vb` without noting this gap.

If `4-V2` then improves over `4-V1`, the structured context pathway is adding
information that was removed from the image. If `4-V2` does not improve over
`4-V1`, then the context design is still not helping even after reducing visual
feature overlap.

# 4-V1. Stage 4 v2 OHLC Visual-Only Control

상태: Kaggle runner 준비 완료, full 결과 대기 중.

## 실험

```text
priority: 4-V1
image_window: I60
return_horizon: R20
image_spec: ohlc
context: 없음
model: Stage 2 Stock_CNN I60 visual-only baseline
seeds: 기본 [42], 수정 가능
```

## 이 Control을 하는 이유

`4-V0`은 selected baseline인 `I60/R20/ohlc_ma_vb` visual-only가 Stage 2 seed-42
결과를 그대로 재현한다는 것을 확인하는 실험입니다. `4-V1`은 여기서 이미지 안의
MA와 volume bar overlay를 제거하고, 나머지는 visual-only로 유지합니다.

이 실험은 FiLM을 다시 고치기 전 두 번째 진단 control입니다.

- 강한 Stage 2 결과가 `ohlc_ma_vb` 이미지 인코딩에 얼마나 의존하는지 확인합니다.
- `4-V2`의 duplicate-feature test를 준비합니다. `4-V2`에서는 이미지는 plain
  `ohlc`로 두고 BB/MFI/RV/F&G는 context pathway로만 넣습니다.
- 이 실험은 FiLM, gating, concat context, F&G, Bollinger, MFI, realized
  volatility를 쓰지 않습니다.

## 기존 기준값

Stage 2 selected `I20/I60, R20, four image specs, five seeds` table 기준:

| 실험 | Accuracy Mean | ROC-AUC Mean | 의미 |
| --- | ---: | ---: | --- |
| `I60/R20/ohlc` | `0.5581` | `0.5602` | 4-V1 기준 |
| `I60/R20/ohlc_ma_vb` | `0.5793` | `0.5849` | selected visual baseline |
| Stage 4 v1 `film_full` | `0.5510` | `0.5677` | context FiLM v1 |

완료된 `4-V1` seed-42 결과:

| Run | Accuracy | Majority Acc. | ROC-AUC | F1 | Predicted Up Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4-V1` `I60/R20/ohlc`, visual-only, seed 42 | `0.5420` | `0.5413` | `0.5441` | `0.6956` | `0.9632` |

모델은 거의 majority-class accuracy 수준이고 test sample 대부분을 Up으로
예측했습니다. 따라서 plain OHLC만으로는 약하고, Stage 2 selected baseline에서
MA/VB visual overlay가 유용한 signal을 제공했다는 해석이 가능합니다.

## Runner

사용할 파일:

```text
notebooks/kaggle_stage4_v2_p2_ohlc_visual_only_one_cell.md
```

이 runner는 의도적으로 Stage 2 visual-only training/evaluation script를 호출합니다.
`4-V1`은 Stage 4 context model이 아니라, context를 다시 넣기 전에 plain OHLC
image만으로 어느 정도 가능한지 확인하는 control이기 때문입니다.

## 결과 해석 기준

`4-V1`이 `4-V0`보다 확실히 낮으면 MA/VB visual overlay가 유용한 정보를 담고
있다는 뜻입니다. 이 경우 v2 context 실험은 plain OHLC와 `ohlc_ma_vb`를 직접
동일선상에서 비교하지 말고, 이미지 정보량 차이를 명시해야 합니다.

그 다음 `4-V2`가 `4-V1`보다 좋아지면 structured context pathway가 이미지에서
제거한 정보를 다시 보완한 것으로 볼 수 있습니다. 반대로 `4-V2`도 좋아지지
않으면, visual overlap을 줄여도 현재 context 설계가 도움을 주지 못한다는
해석이 가능합니다.
