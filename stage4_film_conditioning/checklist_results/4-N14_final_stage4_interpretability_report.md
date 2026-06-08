# 4-N14 Final Stage 4 Interpretability Report

## 1. Stage 2 Baseline 정리

Stage 4의 기준 baseline은 Stage 2에서 가장 강했던 BTC chart-only 모델입니다.

```text
Baseline: Stage2 I60/R20/ohlc_ma_vb
Image window: 60 days
Return horizon: 20 days
Image spec: OHLC + moving average + volume bar
Seeds: 42, 43, 44, 45, 46
Accuracy mean: 0.579320
ROC-AUC mean: 0.584862
```

이 baseline은 Stage 4 전체 실험에서 계속 기준점으로 사용했습니다. 중요한 점은
`ohlc_ma_vb` 이미지 자체가 이미 가격, 추세, 거래량 정보를 많이 담고 있기 때문에
단순 technical context를 추가해도 큰 개선이 나오기 어려웠다는 것입니다.

관련 표:

```text
reports/tables/stage2_n15a_i60_r20_four_image_specs_five_seed_mean_std_results.csv
checklist_results/4-N15_integrated_result_summary.md
```

## 2. Stage 4 전체 결론

Stage 4의 핵심 질문은 chart image에 market context를 어떻게 붙일 것인가였습니다.
처음에는 concat, gating, FiLM gamma-only, FiLM full을 모두 scratch training으로
비교했지만, 강한 Stage 2 visual baseline을 안정적으로 넘지 못했습니다.

이후 실험 방향을 바꿔 Stage 2에서 이미 학습된 visual CNN/classifier를 고정하고,
context encoder와 bounded final-block FiLM만 학습하는 baseline-preserving 구조로
정리했습니다.

최종적으로 확인된 결론은 다음과 같습니다.

1. `BB/MFI/RV` 같은 OHLCV-derived technical context는 대부분 이미지와 중복되어
   큰 효과가 없었습니다.
2. `F&G`는 외부 market-regime 정보라서 technical context보다 방어 가능하지만,
   개선 폭은 작았습니다.
3. headline news TF-IDF/SVD는 ROC-AUC/Brier에는 약간의 신호가 있었지만 hard
   decision 개선은 작았습니다.
4. FSI/RORO macro context는 안정적이었지만 Stage 2 baseline을 의미 있게 넘지
   못했습니다.
5. derivatives/leverage context는 전체 최고 baseline을 넘지는 못했지만,
   weaker volume-aware image인 `ohlc_vb`에서는 작은 same-image 개선과 해석 가능한
   correction pattern을 보였습니다.

따라서 Stage 4의 balanced conclusion은 다음입니다.

```text
Context-FiLM does not substantially outperform the strongest Stage 2 visual
baseline, but a frozen baseline-preserving FiLM protocol can make small,
interpretable corrections. The strongest positive evidence comes from
derivatives/leverage context on the ohlc_vb image variant.
```

관련 결과:

```text
checklist_results/4-N12-D_context_source_comparison.md
checklist_results/4-N13-5_macro_context_source_comparison.md
checklist_results/4-N15_integrated_result_summary.md
checklist_results/4-N16-5_derivatives_interpretability_export.md
```

## 3. 가장 쓸 만한 Positive Case

가장 논문에 쓸 수 있는 positive case는 N16의 derivatives/leverage context입니다.

```text
Stage2 same-image baseline: stage2_i60_ohlc_vb_r20
Stage4 candidate: ohlc_vb + funding_plus_cftc_oi
Context: BitMEX funding + release-lagged CFTC/CME Bitcoin futures OI/positioning
FiLM: bounded final-block gamma/beta
Scale: 0.02
Frozen policy: Stage2 visual CNN frozen, Stage2 classifier frozen
```

결과:

| Model | Accuracy mean | ROC-AUC mean | Brier mean | Predicted-Up mean |
| --- | ---: | ---: | ---: | ---: |
| Stage2 `ohlc_vb` | 0.567384 | 0.561247 | 0.258306 | 0.741013 |
| N16 `ohlc_vb + funding+CFTC` | 0.569466 | 0.561820 | 0.257451 | 0.728938 |
| Delta | +0.002082 | +0.000573 | -0.000855 | -0.012075 |

Correction/regression:

```text
Stage2 wrong -> FiLM correct: 66
Stage2 correct -> FiLM wrong: 51
Net corrections: +15
Changed decisions: 117 / 7205 = 1.6239%
```

이 결과는 전체 최고 모델인 `ohlc_ma_vb` baseline보다 강한 모델이라는 뜻은 아닙니다.
정확한 해석은 다음입니다.

```text
When the visual image is volume-aware but weaker than the full ohlc_ma_vb image,
derivatives/leverage context can provide a small but measurable same-image
correction.
```

관련 파일:

```text
reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_mean_std_results.csv
reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_correction_transition_summary.csv
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_report.md
```

## 4. 해석 가능성 정리

N16-5에서 확인한 핵심 해석은 derivatives/leverage context가 약한 bullish
prediction을 낮추는 방향으로 작동했다는 점입니다.

Correction sample의 평균 future return은 음수였고, regression sample의 평균 future
return은 양수였습니다.

```text
Correction samples future_return mean: -0.0610
Regression samples future_return mean:  0.1270
```

FiLM이 만든 평균 확률 변화는 다음과 같습니다.

```text
Correction samples mean prob_up delta: -0.0128
Regression samples mean prob_up delta: -0.0138
Correction samples mean true-prob delta: 0.0148
Regression samples mean true-prob delta: -0.0151
```

즉 FiLM은 전반적으로 `prob_up`을 조금 낮췄습니다. 이 보정이 실제로 false-Up
sample에서는 도움이 되었고, true-Up sample에서는 regression으로 작동했습니다.

Targeted Grad-CAM/gamma-beta export에서 FiLM modulation은 매우 보수적이었습니다.

```text
block4_gamma_mean:       1.000082
block4_delta_gamma_mean: 0.000082
block4_beta_mean:        0.000018
block4_delta_gamma_l2:   0.027434
block4_beta_l2:          0.028753
```

따라서 해석은 “CNN이 보는 차트 영역을 크게 바꿨다”라기보다는 다음에 가깝습니다.

```text
The derivatives context acted as a small final-block correction signal,
suppressing weak bullish visual predictions during hotter leverage/funding
conditions.
```

이 해석은 교수님 방향성의 “질문은 고정되어 있고, 변하는 것은 market context이므로
context vector가 FiLM gamma/beta를 만든다”는 구조와 맞습니다. 다만 현재 실험에서는
그 조절폭이 성능을 크게 바꿀 만큼 강하지 않았고, 오히려 보수적 correction으로
작동했습니다.

## 5. 그림/표 정리

N14에서 사용할 핵심 artifact는 다음입니다.

Stage2 vs Stage4 targeted Grad-CAM:

```text
stage2_btc_extension/reports/figures/gradcam/*n16_5_derivatives_targeted_label*
stage4_film_conditioning/reports/figures/gradcam/*n16_5_derivatives_targeted_label*
```

Correction/regression sample tables:

```text
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_selected_for_gradcam_augmented.csv
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_selected_sample_view.csv
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_transition_context_summary.csv
```

Gamma/beta and probability-shift summaries:

```text
reports/figures/gradcam/*n16_5_derivatives_targeted_label*modulation_summary*.csv
reports/figures/gradcam/*n16_5_derivatives_targeted_label*modulation_values*.json
reports/tables/stage4_n16_5_ohlc_vb_derivatives_interpretability_report.md
```

Final report table candidates:

```text
reports/tables/stage4_n12d_context_source_comparison_compact.csv
reports/tables/stage4_n15c_fg_only_across_image_specs_mean_std_results.csv
reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_mean_std_results.csv
reports/tables/stage4_n16_4_ohlc_vb_derivatives_repeat_correction_transition_summary.csv
```

## Final Thesis Claim

현재 결과로 주장할 수 있는 thesis contribution은 다음입니다.

1. Re-image 논문 pipeline을 재현하고, BTC 데이터셋으로 Stage 2 baseline을 확장했습니다.
2. BTC에서는 `I60/R20/ohlc_ma_vb`가 가장 강한 visual baseline으로 확인되었습니다.
3. 단순 context concat/gating/FiLM scratch training은 강한 visual baseline을 안정적으로
   넘지 못했습니다.
4. Stage 2 checkpoint를 고정한 bounded FiLM 구조가 가장 안정적인 context-conditioning
   protocol이었습니다.
5. 대부분의 context는 성능 개선 폭이 매우 작았지만, derivatives/leverage context는
   `ohlc_vb`에서 작은 same-image improvement와 해석 가능한 bearish correction
   pattern을 보였습니다.

논문에서 과장하지 말아야 할 부분:

```text
Stage 4 does not currently provide a large overall accuracy gain over the
strongest Stage 2 visual baseline.
```

논문에서 방어 가능하게 쓸 수 있는 부분:

```text
Context-FiLM is useful as a controlled interpretability and correction module:
it can preserve a strong visual baseline and reveal how external market-regime
signals, especially derivatives/leverage context, modulate weak visual
predictions.
```

## Final Scope Decision

현재 Stage 4는 추가 grid를 계속 늘리기보다 여기서 정리하는 것이 합리적입니다.
추가 실험 후보인 FinBERT/LLM sentiment/event feature는 future work 또는 교수님 피드백
후 선택사항으로 남깁니다.

교수님께 확인할 질문:

```text
이 정도의 Stage 4 결과를 “large performance gain”이 아니라
“baseline-preserving context-FiLM interpretability/correction analysis”로
정리하고 논문 초안 작성으로 넘어가도 되는지 확인.
```
