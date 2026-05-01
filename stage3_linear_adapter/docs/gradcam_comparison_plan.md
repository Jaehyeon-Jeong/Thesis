# 3-4 Grad-CAM Comparison Plan

## English

Purpose:
- Keep Grad-CAM mandatory in Stage 3.
- Compare where the baseline CNN and Linear-adapter model attend.

Grad-CAM target:
- Use pre-softmax predicted-class logit as the target score.
- Hook the same CNN convolution layers as Stage 2.
- The Linear adapter changes the final score path, but the heatmap is still
  computed with respect to CNN feature maps.

Output levels:

| Output | Samples | Use |
|:---|---:|:---|
| Quick check | `2` predicted-up + `2` predicted-down | Verify rendering for every run or selected run. |
| Report figure | `10` predicted-up + `10` predicted-down | Figure-13-style report figure for selected configurations. |

Comparison rule:
- For direct baseline-vs-Linear comparison, use the same date/sample whenever
  possible.
- If the two models predict different classes for the same sample, record both
  predicted classes, `prob_up`, true label, and correctness.
- The Grad-CAM target remains each model's own predicted class unless a report
  explicitly states that a fixed class target is being used.

Minimum Stage 3 Grad-CAM outputs:
1. Linear model Grad-CAM for `I60/R20/ohlc_ma_vb`, seed `42`.
2. Baseline-vs-Linear side-by-side Grad-CAM for the same configuration.
3. Final selected Linear configuration with `10` predicted-up and `10`
   predicted-down examples.

Metadata to save:
- `Date`
- `image_window`
- `return_horizon`
- `image_spec`
- `run_seed`
- `model_variant`: `baseline` or `linear_adapter`
- `pred_class`
- `prob_up`
- `label`
- `correct`
- `future_return`
- `sample_index`

## 한국어

목적:
- Stage 3에서도 Grad-CAM을 필수로 유지합니다.
- baseline CNN과 Linear-adapter model이 어느 chart region을 보는지 비교합니다.

Grad-CAM target:
- softmax 이전 predicted-class logit을 target score로 사용합니다.
- Stage 2와 같은 CNN convolution layer에 hook을 겁니다.
- Linear adapter는 final score path를 바꾸지만, heatmap은 여전히 CNN feature map에
  대한 gradient로 계산합니다.

Output 수준:

| Output | Sample 수 | 용도 |
|:---|---:|:---|
| Quick check | predicted-up `2`개 + predicted-down `2`개 | 모든 run 또는 선택 run에서 rendering 확인 |
| Report figure | predicted-up `10`개 + predicted-down `10`개 | Figure 13 스타일 report figure |

비교 규칙:
- baseline-vs-Linear를 직접 비교할 때는 가능하면 같은 date/sample을 사용합니다.
- 같은 sample에서 두 model의 예측 class가 다르면, 두 model의 predicted class,
  `prob_up`, true label, correctness를 모두 기록합니다.
- 별도로 fixed class target이라고 명시하지 않는 한, Grad-CAM target은 각 model의
  자체 predicted class입니다.

최소 Stage 3 Grad-CAM output:
1. `I60/R20/ohlc_ma_vb`, seed `42` Linear model Grad-CAM.
2. 같은 configuration의 baseline-vs-Linear side-by-side Grad-CAM.
3. 최종 선택된 Linear configuration에 대해 predicted-up 10개와 predicted-down
   10개 Figure 13 스타일 figure.

저장할 metadata:
- `Date`
- `image_window`
- `return_horizon`
- `image_spec`
- `run_seed`
- `model_variant`: `baseline` 또는 `linear_adapter`
- `pred_class`
- `prob_up`
- `label`
- `correct`
- `future_return`
- `sample_index`
