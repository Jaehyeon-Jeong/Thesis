# 5-15. FinBERT+F&G FiLM 해석 분석 플랜

## 1. 목표와 기준

- 목표: `Stage2 chart-only wrong -> FinBERT+F&G FiLM correct` 사례에서
  **context가 어떤 상황이었고, gamma/beta가 visual feature를 어떻게
  보정했는지** 설명한다.
- 기준 모델: 최종 thesis model인
  `I60/R20/ohlc_ma_vb + FinBERT + F&G + frozen bounded FiLM`.
- 중요한 정정: “correct 사례가 9개”가 아니라 **net correction이 +9**이다.
  전체 transition은 `corrections 95`, `regressions 86`, `net +9`로 보는 게
  맞다.
- 해석 수준: gamma vector 자체가 “이 channel은 fear다”라고 직접 말해주지는
  않는다. 해석은 `context 값 + prediction transition + P(up) 변화 +
  gamma/beta 변화 + Grad-CAM`을 연결해서 한다.

## 2. 분석 절차

### [x] 5-15A. Transition 샘플 분리

- 전체 test matched rows에서 네 그룹으로 나눈다.
  - `correction`: Stage2 wrong, FiLM correct.
  - `regression`: Stage2 correct, FiLM wrong.
  - `both_correct`.
  - `both_wrong`.
- 사용 파일:
  - `stage5_5_11_finbert_fg_condition_analysis_merged_decisions.csv`
  - `stage5_5_11_finbert_fg_condition_analysis_selected_samples_for_5_12.csv`
  - `stage5_5_12_finbert_fg_targeted_gradcam_modulation_summary.csv`
- 확인된 transition:
  - `corrections 95`
  - `regressions 86`
  - `net +9`

### [x] 5-15B. Correction 대표 샘플 선택

- thesis 본문/부록용 대표 correction 4개를 고른다.
- 우선순위:
  - `abs(P_film_up - P_stage2_up)`가 큰 샘플.
  - Stage2가 Up으로 틀렸고 FiLM이 Down으로 맞춘 샘플.
  - context가 해석 가능한 샘플:
    high FinBERT negative, low/medium F&G, high news count 등.
- 비교용 regression 4개도 고른다.
- 목적:
  - “FiLM이 항상 좋은 게 아니라, 같은 조절이 어떤 경우에는 correction,
    어떤 경우에는 regression이 된다”는 점을 보여준다.

선택 결과:

- Correction:
  - `2021-04-20`, seed `43`, sample `1087`
  - `2021-05-19`, seed `43`, sample `1116`
  - `2021-07-02`, seed `43`, sample `1160`
  - `2021-06-26`, seed `42`, sample `1154`
- Regression:
  - `2021-10-22`, seed `43`, sample `1272`
  - `2021-10-26`, seed `43`, sample `1276`
  - `2024-11-25`, seed `44`, sample `2402`
  - `2021-02-24`, seed `43`, sample `1032`

### [x] 5-15C. Context 해석 테이블 생성

- 각 선택 샘플에 대해 아래 값을 하나의 표로 정리한다.
  - `date`
  - `true label`
  - `Stage2 pred`
  - `FiLM pred`
  - `Stage2 P(up)`
  - `FiLM P(up)`
  - `delta P(up)`
  - `F&G level / F&G rolling mean`
  - `FinBERT positive / neutral / negative ratio`
  - `news count 7d/20d/60d`
- 해석 문장 예시:
  - “negative sentiment + fear/low greed + high news intensity에서 FiLM이
    bullish probability를 낮췄다.”
- 생성 파일:
  - `reports/tables/stage5_5_15_selected_film_interpretation_cases.csv`
  - `reports/tables/stage5_5_15_selected_film_interpretation_cases_compact.csv`

### [x] 5-15D. Gamma/Beta modulation 분석

- `gamma`는 직접값보다 `delta_gamma = gamma - 1`로 본다.
- correction vs regression 그룹별로 아래 값을 비교한다.
  - mean `delta_gamma`
  - mean `beta`
  - L2 norm of `delta_gamma`
  - L2 norm of `beta`
  - largest absolute modulation channels top-k
- 시각화:
  - correction/regression 평균 `delta_gamma` bar plot.
  - correction/regression 평균 `beta` bar plot.
- 해석 원칙:
  - `delta_gamma > 0`: 해당 visual channel 강화.
  - `delta_gamma < 0`: 해당 visual channel 약화.
  - `beta > 0`: feature activation upward shift.
  - `beta < 0`: feature activation downward shift.
- 단, channel 이름을 경제적 의미로 직접 번역하지 않는다.
  - 예: “channel 17 = fear”라고 말하지 않는다.
  - 대신 “context branch changed the frozen visual representation in this
    direction”으로만 해석한다.
- 생성 파일:
  - `reports/tables/stage5_5_15_gamma_beta_correction_vs_regression_summary.csv`
  - `reports/tables/stage5_5_15_gamma_beta_correction_vs_regression_summary_compact.csv`
  - `reports/figures/stage5_5_15_gamma_beta_correction_vs_regression_plot.png`

핵심 결과:

| Group | n | mean delta gamma | mean beta | mean delta-gamma L2 | mean beta L2 |
|---|---:|---:|---:|---:|---:|
| correction | 20 | 0.000334 | 0.000079 | 0.052070 | 0.055608 |
| regression | 20 | 0.000349 | 0.000082 | 0.055658 | 0.058997 |

해석:

- correction과 regression 모두 modulation이 매우 작다.
- regression 쪽 modulation magnitude가 약간 더 크지만 차이는 크지 않다.
- 즉 FinBERT+F&G FiLM은 visual feature를 크게 재작성하는 것이 아니라,
  frozen visual model 위에서 작은 calibration을 수행한다.

### [x] 5-15E. Grad-CAM 비교

- 선택한 correction 샘플에 대해 아래를 비교한다.
  - Chart-only Grad-CAM.
  - FinBERT+F&G FiLM Grad-CAM.
  - 같은 날짜, 같은 label 기준.
- Figure 7은 `Stage2 wrong -> FiLM correct` correction 샘플만 사용한다.
- 추천 캡션:
  - `Figure 7 — Correction examples with Grad-CAM and context-FiLM modulation`
- 그림 아래 본문에서 설명할 것:
  - Stage2가 이미지에서 어떤 구간을 강조했는지.
  - FiLM 이후 attention/activation이 어떻게 바뀌었는지.
  - 그날 context가 왜 bearish/bullish 보정으로 작동했는지.
- 생성 파일:
  - `reports/figures/gradcam/stage5_5_15_gradcam_correction_cases.png`
  - `Thesis/thesis_draft/figures/gradcam_correction_cases.png`
  - `Thesis/thesis_draft/figures/gradcam_finbert_fg_corrections_chronological.png`
- thesis에는 시간순으로 재정렬한
  `gradcam_finbert_fg_corrections_chronological.png`를 Figure 7로 사용한다.

## 3. 논문 반영 위치

분석 파일 생성과 thesis 본문 반영을 모두 완료했다.

### [x] 4.x Interpretability protocol

- 분석 절차만 짧게 정의한다.
- correction/regression, context table, gamma/beta summary, Grad-CAM을 어떻게
  연결할지 설명한다.
- 반영 위치:
  - `Thesis/thesis_draft/Thesis_Jaehyeon_Jeong.tex`
  - Section `4.6 Interpretability protocol`

### [x] 6.4 Sample-level FiLM interpretation

실제 case study를 넣는다. 추천 구조:

1. Paragraph 1:
   - 왜 sample-level analysis가 필요한지 설명.
   - 전체 accuracy gain이 작기 때문에, context가 실제로 어떤 decision을
     바꿨는지 봐야 한다고 설명.
2. Table:
   - selected correction/regression samples and context values.
3. Figure:
   - Grad-CAM comparison.
4. Paragraph 2:
   - gamma/beta가 어떻게 보정했는지 설명.
5. Paragraph 3:
   - 한계 명시.
   - gamma vector는 causal/event label이 아니라 diagnostic modulation이다.
- 반영 위치:
  - `Thesis/thesis_draft/Thesis_Jaehyeon_Jeong.tex`
  - Section `6.4 Interpretability of FiLM modulation`
  - Table 10, Table 11, Figure 7

### [x] Conclusion / RQ3

- “FiLM provides diagnostic interpretability, not causal explanation”으로
  정리한다.
- 강한 claim 금지:
  - “뉴스/F&G가 원인이다”라고 말하지 않는다.
  - 대신 “이 context regime에서 FiLM이 visual prediction을 보수적으로
    보정했다”라고 쓴다.
- 반영 위치:
  - `Thesis/thesis_draft/Thesis_Jaehyeon_Jeong.tex`
  - Conclusion, `Answers to research questions`, `RQ3`

## 4. 산출물

- `selected_film_interpretation_cases.csv`
  - correction 4개, regression 4개, 각 context/probability/gamma-beta 요약 포함.
- `gamma_beta_correction_vs_regression_summary.csv`
  - correction/regression/unchanged 그룹별 modulation 통계.
- `gamma_beta_correction_vs_regression_plot.png`
  - 평균 `delta_gamma`/`beta` 비교 플롯.
- `gradcam_correction_cases.png`
  - 원본 Grad-CAM artifact.
- `gradcam_finbert_fg_corrections_chronological.png`
  - thesis Figure 7 최종 사용본.
- analysis report:
  - `reports/tables/stage5_5_15_sample_level_film_interpretation_report.md`
- thesis text update:
  - 완료.
  - `4.x interpretability protocol`, `6.4 sample-level interpretation`,
    `Conclusion/RQ3`에 반영.

## 5. 검증 기준

- 선택된 correction 샘플은 실제로 `Stage2 wrong -> FiLM correct`여야 한다.
- regression 샘플은 실제로 `Stage2 correct -> FiLM wrong`이어야 한다.
- probability 변화 방향과 설명이 일치해야 한다.
  - bearish correction이면 보통 `P(up)` 감소.
  - bullish correction이면 보통 `P(up)` 증가.
- gamma/beta 해석은 channel-level causal claim을 하지 않는다.
- thesis 본문에서는 “해석 가능성”을 다음 세 가지 연결로 제한한다.
  - context regime
  - prediction/probability transition
  - bounded FiLM modulation and Grad-CAM diagnostic

## 6. 기본 가정

- 새 모델 학습은 하지 않는다.
- 기존 5-11, 5-12 결과 파일을 우선 사용한다.
- 부족하면 추가로 필요한 것은 training이 아니라 post-hoc export/plot 생성이다.
- 본문에는 1개 대표 Figure와 1개 compact Table만 넣고, 나머지는 appendix 또는
  GitHub artifact로 연결한다.

## 7. 현재 결론

- FinBERT+F&G FiLM은 7,205 matched decisions 중 181개만 hard decision을
  바꿨다.
- Correction은 95개, regression은 86개, net correction은 +9이다.
- 대표 correction case에서는 chart-only의 weak `Up` probability를 약 0.05
  낮춰 false-Up을 Down으로 고쳤다.
- 하지만 같은 downward calibration이 true-Up regression도 만든다.
- 따라서 논문에서는 “large performance gain”이 아니라 “small, bounded,
  inspectable calibration”으로 해석하는 것이 안전하다.
