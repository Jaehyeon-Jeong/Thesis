# Stage 5-15 Sample-Level FiLM Interpretation Report

## 1. 분석 목적

이 분석은 최종 후보 모델인 `I60/R20/ohlc_ma_vb + FinBERT + F&G +
frozen bounded FiLM`이 실제로 어떤 샘플에서 chart-only 예측을 바꾸는지
확인하기 위한 후처리 분석이다. 새 모델 학습은 하지 않았다.

핵심 질문은 다음과 같다.

- `Stage2 chart-only wrong -> FinBERT+F&G FiLM correct` 사례에서 context는
  어떤 상태였는가?
- FiLM 적용 후 `P(up)`은 어느 방향으로 움직였는가?
- gamma/beta modulation은 correction과 regression에서 얼마나 달랐는가?
- Grad-CAM과 연결했을 때 “해석 가능성”을 어디까지 주장할 수 있는가?

주의할 점은 correction sample이 9개가 아니라는 것이다. 전체 matched
decision 기준 transition은 다음과 같다.

| Transition | Count |
|---|---:|
| both correct | 4088 |
| both wrong | 2936 |
| correction | 95 |
| regression | 86 |
| net correction | +9 |

즉 `+9`는 net correction count이고, 실제 correction sample은 `95`개다.

## 2. 산출물

- Selected case table:
  `stage5_llm_news_embedding/reports/tables/stage5_5_15_selected_film_interpretation_cases.csv`
- Compact selected case table:
  `stage5_llm_news_embedding/reports/tables/stage5_5_15_selected_film_interpretation_cases_compact.csv`
- Gamma/beta summary:
  `stage5_llm_news_embedding/reports/tables/stage5_5_15_gamma_beta_correction_vs_regression_summary.csv`
- Compact gamma/beta summary:
  `stage5_llm_news_embedding/reports/tables/stage5_5_15_gamma_beta_correction_vs_regression_summary_compact.csv`
- Gamma/beta plot:
  `stage5_llm_news_embedding/reports/figures/stage5_5_15_gamma_beta_correction_vs_regression_plot.png`
- Grad-CAM correction case figure:
  `stage5_llm_news_embedding/reports/figures/gradcam/stage5_5_15_gradcam_correction_cases.png`

## 3. 선택된 대표 case

본문 Figure 7 후보와 맞추기 위해 correction 3개는 기존 Grad-CAM correction
case와 같은 날짜를 사용했다. 여기에 해석이 더 깔끔한 bearish-context
borderline correction 1개를 추가했다. Regression은 probability 변화가 크고
modulation export가 존재하는 샘플 중에서 4개를 선택했다.

| Case | Type | Date | True | Chart-only pred | FiLM pred | Chart-only P(up) | FiLM P(up) | Delta P(up) | F&G | News 20d | FinBERT positive 20d | FinBERT negative 20d |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | correction | 2021-04-20 | Down | Up | Down | 0.540032 | 0.490026 | -0.050006 | 73 | 288 | 0.298611 | 0.215278 |
| C2 | correction | 2021-05-19 | Down | Up | Down | 0.532897 | 0.485639 | -0.047258 | 23 | 257 | 0.233463 | 0.252918 |
| C3 | correction | 2021-07-02 | Down | Up | Down | 0.528077 | 0.477073 | -0.051004 | 21 | 366 | 0.229508 | 0.357923 |
| C4 | correction | 2021-06-26 | Down | Up | Down | 0.506000 | 0.486897 | -0.019104 | 20 | 422 | 0.206161 | 0.315166 |
| R1 | regression | 2021-10-22 | Up | Up | Down | 0.529816 | 0.444130 | -0.085687 | 75 | 391 | 0.409207 | 0.130435 |
| R2 | regression | 2021-10-26 | Up | Up | Down | 0.506138 | 0.433640 | -0.072499 | 76 | 390 | 0.387179 | 0.135897 |
| R3 | regression | 2024-11-25 | Up | Up | Down | 0.509975 | 0.441396 | -0.068579 | 82 | 296 | 0.456081 | 0.111486 |
| R4 | regression | 2021-02-24 | Up | Up | Down | 0.548703 | 0.488375 | -0.060328 | 76 | 360 | 0.325000 | 0.163889 |

### 3.1 직관적 상황 해설

아래 해석은 “왜 이 샘플에서 FiLM이 맞췄는가/틀렸는가”를 직관적으로 읽기
위한 설명이다. 단, 이것은 causal explanation이 아니라 context 값과 prediction
transition을 연결한 post-hoc diagnostic interpretation이다.

| Case | 직관적 상황 해설 | 해석 강도 |
|---|---|---|
| C1 | Mixed sentiment-deterioration case. F&G는 아직 `73`으로 greed 구간이지만, 60일 전 대비 `-20` 하락했고 뉴스량도 높다. 즉 아직 분위기는 좋지만 이전의 extreme-greed 과열이 식는 구간으로 볼 수 있다. FiLM은 이 mixed context에서 약한 bullish chart-only 확률을 낮췄다. | 중간 |
| C2 | Sharp fear-transition case. F&G는 `23`으로 낮고, 60일 평균 `61.45`보다 크게 낮아졌으며 60일 변화도 `-52`이다. FinBERT도 약하게 negative-leaning이다. FiLM이 weak false-Up chart prediction을 Down으로 낮춘 해석이 비교적 자연스럽다. | 좋음 |
| C3 | Clean fear and negative-news case. F&G는 `21`로 extreme fear이고, 60일 평균도 `28.23`으로 낮으며, FinBERT negative ratio가 positive ratio보다 높다. 가장 깔끔한 bearish-context correction 사례다. | 가장 좋음 |
| C4 | Borderline fear correction. Stage2 `P(up)`이 `0.506`으로 거의 0.5 근처였고, F&G는 `20`, 60일 변화는 `-30`, FinBERT negative ratio도 positive보다 높으며 news count도 높다. 작은 FiLM 보정만으로도 Down 전환이 가능한 샘플이다. | 좋지만 probability shift는 작음 |
| R1 | Positive-context over-correction. 실제 label은 Up이고 chart-only도 맞췄지만, F&G는 `75`, FinBERT는 positive-leaning이다. 그럼에도 FiLM이 `P(up)`을 크게 낮춰 Down으로 틀렸다. Downward calibration의 한계를 보여준다. | 명확한 limitation |
| R2 | Greed-context over-correction. F&G는 `76`, FinBERT도 positive-leaning이고 실제 label은 Up이다. FiLM이 같은 downward calibration을 적용해 weak correct-Up prediction을 wrong-Down으로 바꿨다. | 명확한 limitation |
| R3 | Late-sample positive-regime regression. F&G는 `82`로 extreme greed이고 FinBERT positive ratio도 높다. 하지만 FiLM이 Up probability를 과하게 낮춰 실제 Up을 놓쳤다. | 명확한 limitation |
| R4 | Mixed but positive-leaning regression. F&G는 `76`으로 높고 FinBERT도 positive-leaning이지만, F&G는 이전보다 약화된 상태다. 그래도 실제 label은 Up이었기 때문에, FiLM의 보수적 보정이 regression을 만든 사례다. | 중간 limitation |

## 4. Case-level 해석

### Correction cases

네 correction 사례는 모두 chart-only model이 `Up`으로 예측했지만 실제 label은
`Down`이었다. FinBERT+F&G FiLM은 네 경우 모두 `P(up)`을 낮춰서 hard
prediction을 `Down`으로 바꾸었다.

- C2, C3, C4는 F&G가 낮고, FinBERT 20-day sentiment도 중립 또는
  negative-leaning에 가깝다. 이 경우 “context가 bullish chart-only
  probability를 보수적으로 낮췄다”는 해석이 자연스럽다.
- C1은 F&G와 FinBERT ratio만 보면 positive-leaning이지만, F&G delta가
  `-20`으로 빠르게 악화된 시점이다. 따라서 단순히 “positive sentiment이면
  Up”이라고 해석하면 안 된다. FiLM context branch는 여러 context feature의
  조합을 사용하고, 이 사례에서는 chart-only의 weak bullish signal을 낮추는
  쪽으로 작동했다.
- C4는 probability shift는 작지만 context 해석이 가장 깔끔한 borderline
  correction이다. F&G는 `20`, 60-day F&G delta는 `-30`, FinBERT negative
  ratio가 positive ratio보다 높고, 20-day news count도 높다. 따라서 fear와
  negative-news context에서 weak Up probability가 Down으로 보정된 사례로
  읽기 좋다.

### Regression cases

R1부터 R4는 실제 label이 `Up`이고 chart-only도 `Up`으로 맞췄지만, FiLM이
`P(up)`을 더 크게 낮춰 `Down`으로 바꾼 사례다. 네 샘플은 오히려 F&G와
FinBERT sentiment가 positive-leaning이다.

이 점은 중요하다. FinBERT+F&G FiLM이 단순한 sentiment rule이 아니라는 뜻이다.
하지만 동시에 같은 downward calibration이 correction뿐 아니라 regression도
만든다는 한계를 보여준다. 따라서 논문에서는 “FiLM이 특정 context에서 항상
정답 방향으로 움직인다”가 아니라, “bounded FiLM이 inspectable한 probability
calibration을 수행하고, 그중 일부가 false-Up correction으로 이어졌다”고
표현해야 한다.

## 5. Gamma/Beta modulation summary

Correction과 regression의 평균 modulation은 다음과 같다.

| Group | n | mean delta gamma | mean beta | mean delta-gamma L2 | mean beta L2 |
|---|---:|---:|---:|---:|---:|
| correction | 20 | 0.000334 | 0.000079 | 0.052070 | 0.055608 |
| regression | 20 | 0.000349 | 0.000082 | 0.055658 | 0.058997 |

해석:

- gamma와 beta는 거의 identity에 가깝다.
- regression group의 modulation magnitude가 correction보다 약간 크지만,
  차이는 매우 작다.
- 이는 최종 모델이 visual feature를 크게 재작성하지 않고, small bounded
  calibration만 수행한다는 기존 해석과 일치한다.
- gamma/beta channel은 익명 CNN channel이므로 “이 channel은 fear다”처럼
  직접 경제적 의미를 부여하면 안 된다.

## 6. Grad-CAM 연결

Grad-CAM figure는 다음 파일에 정리했다.

`stage5_llm_news_embedding/reports/figures/gradcam/stage5_5_15_gradcam_correction_cases.png`

이 figure는 세 correction case만 보여준다.

- 위 행: chart-only CNN.
- 아래 행: FinBERT+F&G context-FiLM.
- 세 case 모두 chart-only는 weak `Up` probability를 냈고, context-FiLM은
  `P(up)`을 0.5 아래로 낮춰 `Down`으로 맞췄다.

논문에 넣을 때는 이 figure를 causal proof로 쓰면 안 된다. 더 안전한 표현은
다음과 같다.

> The Grad-CAM panels show the visual evidence inspected by the CNN before and
> after context conditioning. Together with the probability transition and
> gamma/beta summaries, they indicate that FinBERT+F&G FiLM acted as a small
> conservative calibration layer in these false-Up correction cases.

## 7. 논문에 사용할 수 있는 claim

강하게 말할 수 있는 것:

- FinBERT+F&G FiLM은 7,205 matched decisions 중 181개만 hard decision을
  바꿨다.
- 이 중 correction은 95개, regression은 86개로 net `+9`이다.
- 대표 correction case에서는 `P(up)`을 약 0.05 낮춰 false-Up을 Down으로
  고쳤다.
- gamma/beta 값은 작고 bounded되어 있어 visual representation을 크게
  바꾸지 않는다.
- 따라서 FiLM은 replacement predictor가 아니라 conservative calibration
  mechanism으로 해석하는 것이 맞다.

강하게 말하면 안 되는 것:

- “뉴스/F&G가 해당 날짜 하락의 원인이다.”
- “특정 gamma channel이 특정 경제적 의미를 가진다.”
- “FiLM이 일반적으로 항상 성능을 크게 높인다.”
- “positive news이면 Up, negative news이면 Down으로 작동한다.”

## 8. Thesis 반영 권장 위치

Thesis 본문에는 마지막 정리 단계에서만 반영한다.

- Methodology의 interpretability protocol:
  correction/regression, context table, gamma/beta, Grad-CAM을 함께 보는
  분석 절차를 1문단으로 추가.
- Section 6.4:
  대표 case table 1개와 Grad-CAM figure 1개를 넣고, gamma/beta summary는
  숫자 몇 개로만 설명.
- Conclusion/RQ3:
  “diagnostic interpretability, not causal explanation”으로 정리.

## 9. 결론

이 분석은 최종 모델의 해석 가능성을 “큰 성능 향상”이 아니라 “작고 추적 가능한
보정”으로 보여준다. 특히 false-Up correction 사례에서는 context-FiLM이
chart-only probability를 보수적으로 낮추는 방향으로 작동했다. 다만 같은
downward calibration이 true-Up regression도 만들기 때문에, 최종 논문에서는
성능 claim보다 diagnostic calibration claim을 중심에 두는 것이 안전하다.
