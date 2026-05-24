# Professor Meeting Brief: Stage 1-4 Direction and Stage 4 Experiment Plan

## 한국어

## 보고 목적

교수님께 확인받고 싶은 핵심은 다음입니다.

```text
Stage 4에서 제가 이해한 방향이 맞는가?

이미지에 Bollinger/F&G/MFI를 추가로 그리는 것이 아니라,
이미지는 강한 CNN baseline으로 유지하고,
market context를 별도 condition vector로 넣어
CNN feature를 concat/gating/FiLM 방식으로 조절하는 방향이 맞는가?
```

제가 이렇게 이해한 이유는 교수님이 보내주신 방향성 파일
`film_chart_research_summary.md`가 Stage 4를 full VQA나 단순 이미지 확장이 아니라
**chart-image feature의 conditional modulation** 문제로 정리하고 있기 때문입니다.

## 전체 진행 헤드라인

| Stage | 실험 위치 | 현재 해석 | 다음 결정 |
|:---|:---|:---|:---|
| Stage 1 | Original Re-image pipeline | 주식 chart image CNN 파이프라인 재현. `Ret_5d/20d/60d > 0` label, train/validation/test, CNN 학습, prediction/metrics/Grad-CAM 저장 구조 확정 | 재현 기준 baseline으로 유지 |
| Stage 2 | Original pipeline + asset shift to BTC | 같은 image-CNN 구조를 BTC OHLCV로 확장. `I5/I20/I60`, `R5/R20/R60`, 4개 image spec 비교. Selected five-seed check에서 `I60/R20/ohlc_ma_vb`가 가장 강함: accuracy mean `0.5793`, ROC-AUC mean `0.5849` | Stage 4의 고정 image baseline으로 사용 |
| Stage 3 | BTC CNN + Linear adapter | CNN feature 뒤에 단순 Linear adapter를 추가. `CNN -> flatten -> Linear(feature_dim, 128, bias=False) -> Linear(128, 2, bias=False)`. 첫 테스트에서 Stage 2 best single-seed accuracy `0.603053`이 Stage 3 `0.541291`로 하락 | 단순 parameter/Linear 추가만으로는 개선되지 않았다는 negative ablation |
| Stage 4 | BTC CNN + market context conditioning | 이미지 자체를 더 복잡하게 만들기보다, structured market context가 CNN feature를 어떻게 조절하는지 비교 | 교수님께 4-A/B/C/D 실험 방향 확인 필요 |

## 교수님 방향성 파일에서 읽은 핵심

| 교수님 파일 발췌 | 제가 이해한 의미 | Stage 4 설계 반영 |
|:---|:---|:---|
| "does not treat the task as a full visual question answering problem" | 이미지+텍스트 질문을 매번 넣는 VQA 문제가 아님 | 질문은 "이후 R20일 Up/Down?"으로 고정 |
| "conditional modulation mechanism for chart-image features" | 핵심은 image feature를 조건부로 조절하는 것 | market context -> CNN feature modulation |
| "strong no-conditioning baseline" | chart-image CNN 자체는 이미 강한 baseline | Stage 2 best를 고정 baseline으로 둠 |
| "the prediction query is effectively fixed" | 질문 encoder/RNN이 필수 아님 | text question 대신 market context vector 사용 |
| "compact structured metadata" | condition은 구조화된 작은 numeric vector로 가능 | F&G, Bollinger, MFI, volatility 사용 |
| "an MLP or embedding-based condition encoder is a cleaner design than an RNN" | numeric context는 MLP로 encoding하는 것이 자연스러움 | `market context vector -> MLP condition embedding` |
| "CNN + naive condition concatenation" | FiLM과 비교할 단순 fusion baseline 필요 | Stage 4-A |
| "Optional attention-based fusion" | FiLM이 아닌 강한 비교대상 필요 | Stage 4-B gating |
| "CNN + FiLM" | 핵심 모델은 FiLM modulation | Stage 4-C/D |
| "The path from condition input to visual feature modulation is explicit" | FiLM의 중요한 장점은 해석 가능성 | gamma/beta/gate 저장 및 분석 |

## Stage 4에서 제가 확정하고 싶은 구조

교수님 의도는 다음이 아니라고 이해했습니다.

```text
X: Bollinger band, F&G, MFI를 이미지 위에 추가로 그려서
   더 복잡한 chart image를 만드는 방향
```

제가 이해한 방향은 다음입니다.

```text
chart image -> CNN visual feature
market context vector -> MLP condition embedding
condition embedding -> concat / gate / FiLM gamma-beta
```

즉 chart image는 Stage 2에서 가장 강했던 `I60/R20/ohlc_ma_vb`를 유지하고,
F&G/Bollinger/MFI/volatility는 별도 numeric context vector로 넣습니다.

## 왜 실험을 4-A/B/C/D로 나누는가

교수님 파일에서 "FiLM의 gain이 단순히 parameter 추가나 side information 추가 때문인지
구분해야 한다"는 취지로 이해했습니다. 그래서 다음처럼 단계적으로 비교하려고 합니다.

| Track | Model | 구조 | 하는 이유 | 교수님 파일과의 연결 |
|:---|:---|:---|:---|:---|
| Stage 4-A | CNN + context concat | CNN feature와 context embedding을 classifier 직전에 붙임 | context를 단순히 붙이기만 해도 성능이 오르는지 확인 | "CNN + naive condition concatenation" |
| Stage 4-B | CNN + context gating | context embedding으로 feature/channel gate를 만들고 CNN feature에 곱함 | FiLM보다 단순한 multiplicative modulation 비교 | "Optional attention-based fusion"을 단순 gating 대안으로 구현 |
| Stage 4-C | CNN + context FiLM gamma-only | context가 block별 `gamma`만 생성하고 `F' = gamma * F` 적용 | scaling modulation만으로 충분한지 확인. Stage 3 Linear와도 구분되는 channel-wise modulation | "feature-wise affine modulation itself" 중 scale 효과 분리 |
| Stage 4-D | CNN + context FiLM full | context가 block별 `gamma`, `beta` 생성. `F' = gamma * F + beta` | main FiLM 모델. 성능뿐 아니라 gamma/beta 해석 가능 | "CNN + FiLM", `FiLM(F)=gamma*F+beta` |

이렇게 나누면 다음을 분리해서 설명할 수 있습니다.

```text
1. context 정보 자체가 도움이 되는가?        -> concat
2. context로 feature를 곱해 조절하면 좋은가? -> gating
3. FiLM scaling만으로 충분한가?              -> gamma-only
4. full FiLM이 성능/해석력에서 더 나은가?    -> gamma + beta
```

## Market Context 설계

Stage 4의 primary image baseline은 Stage 2 selected best인:

```text
I60 / R20 / ohlc_ma_vb
```

따라서 context도 image window와 맞춰 60일 요약을 primary로 둡니다.

```text
context_window = image_window
```

Primary context vector:

```text
[
  fg_value,
  fg_mean_60,
  fg_delta_60,
  fg_std_60,
  bb_percent_b_60,
  bb_bandwidth_60,
  mfi_60,
  rv_60,
]
```

이렇게 하는 이유:
- CNN이 보는 chart가 60일이면, context도 같은 60일 시장 상태를 요약하는 것이 더
  논리적입니다.
- `BB20`이나 `MFI14`는 기술적 분석에서 흔한 기본값이지만, Stage 4의 main thesis인
  "same chart, different market context"에는 `BB60/MFI60/F&G60`이 더 잘 맞습니다.
- `BB20/MFI14/F&G20`은 버리지 않고, 나중에 `standard_window` 또는 `multi_scale`
  diagnostic으로 비교합니다.

## News/LLM은 어떻게 둘 것인가

뉴스는 버리지 않습니다. 다만 first main run은 structured numeric context로 둡니다.

이유:
- F&G/Bollinger/MFI/volatility는 날짜 정렬과 leakage audit이 쉽습니다.
- 뉴스는 publication time, daily aggregation, embedding/cache, LLM version 문제가 먼저
  정리되어야 합니다.
- 따라서 news context는 second-phase로 두고, 같은 방식의 concat/gating/FiLM 비교에
  붙입니다.

## 교수님께 확인받을 질문

1. Stage 4를 "이미지에 지표를 더 그리는 실험"이 아니라
   `market context vector -> condition embedding -> CNN feature modulation`으로 이해한 것이 맞는지.
2. Stage 4 primary context를 `I60`에 맞춰 `BB60/MFI60/F&G60/RV60`로 두고,
   `BB20/MFI14`는 나중의 ablation으로 두는 것이 맞는지.
3. 4-A concat, 4-B gating, 4-C gamma-only FiLM, 4-D full FiLM의 비교가
   교수님이 말씀하신 ablation 의도와 맞는지.
4. News/LLM은 바로 main model에 넣기보다 numeric context 실험 이후 second-phase로 두는 것이 맞는지.

## English

## Purpose

The meeting question is:

```text
Is my Stage 4 interpretation correct?

Instead of drawing Bollinger/F&G/MFI directly into the chart image,
the image CNN remains the strong visual baseline,
and structured market context is encoded as a separate condition vector
that modulates CNN features through concat, gating, or FiLM.
```

This interpretation follows `film_chart_research_summary.md`, which frames the
next step as **conditional modulation of chart-image features**, not a full VQA
task and not merely a more decorated chart image.

## Stage-Level Headline

| Stage | Position | Current interpretation | Next decision |
|:---|:---|:---|:---|
| Stage 1 | Original Re-image pipeline | Reproduce the stock chart-image CNN pipeline: future return label, split, train-only normalization, CNN, metrics, Grad-CAM | Keep as original baseline reference |
| Stage 2 | Original pipeline + BTC asset shift | Apply the same image-CNN logic to BTC OHLCV. Best selected five-seed check: `I60/R20/ohlc_ma_vb`, accuracy mean `0.5793`, ROC-AUC mean `0.5849` | Use as fixed Stage 4 image baseline |
| Stage 3 | BTC CNN + Linear adapter | Add a post-flatten Linear adapter: `CNN -> flatten -> Linear(feature_dim, 128, bias=False) -> Linear(128, 2, bias=False)`. First test drops from Stage 2 best single-seed accuracy `0.603053` to `0.541291` | Treat as negative/simple-parameter ablation |
| Stage 4 | BTC CNN + market context conditioning | Keep the image baseline fixed and test how market context should condition visual features | Confirm 4-A/B/C/D plan |

## Advisor-Note Mapping

| Excerpt from advisor note | Interpretation | Stage 4 decision |
|:---|:---|:---|
| "does not treat the task as a full visual question answering problem" | This is not a full VQA task | Fixed Up/Down prediction query |
| "conditional modulation mechanism for chart-image features" | The contribution is feature modulation | Context-conditioned CNN features |
| "strong no-conditioning baseline" | The image CNN is already meaningful | Use Stage 2 best as fixed baseline |
| "the prediction query is effectively fixed" | No RNN question encoder is needed | Use numeric market context instead of text question |
| "compact structured metadata" | Condition can be a low-dimensional context vector | F&G, Bollinger, MFI, volatility |
| "an MLP or embedding-based condition encoder is a cleaner design than an RNN" | Encode context with MLP | `context vector -> MLP condition embedding` |
| "CNN + naive condition concatenation" | Need simple fusion baseline | Stage 4-A |
| "Optional attention-based fusion" | Need stronger non-FiLM comparison | Stage 4-B gating |
| "CNN + FiLM" | Main conditional modulation model | Stage 4-C/D |
| "The path from condition input to visual feature modulation is explicit" | FiLM supports interpretability | Save/analyze gamma, beta, gates |

## Stage 4 Model Split

| Track | Model | Structure | Purpose | Advisor-note link |
|:---|:---|:---|:---|:---|
| 4-A | CNN + context concat | Append context embedding before classifier | Test whether simple side information is enough | "CNN + naive condition concatenation" |
| 4-B | CNN + context gating | Context creates feature/channel gates | Compare against a simple attention/gating alternative | "Optional attention-based fusion" |
| 4-C | CNN + FiLM gamma-only | Context creates block-wise `gamma`; `F' = gamma * F` | Test scale-only modulation | "feature-wise affine modulation" |
| 4-D | CNN + FiLM full | Context creates block-wise `gamma`, `beta`; `F' = gamma * F + beta` | Main FiLM and interpretability model | "CNN + FiLM" |

## Primary Context Design

The primary Stage 4 baseline is:

```text
I60 / R20 / ohlc_ma_vb
```

Therefore the primary context uses matched 60-day summaries:

```text
[
  fg_value,
  fg_mean_60,
  fg_delta_60,
  fg_std_60,
  bb_percent_b_60,
  bb_bandwidth_60,
  mfi_60,
  rv_60,
]
```

`BB20`, `MFI14`, and short F&G summaries remain as later diagnostics under
`standard_window` or `multi_scale`, not as the main Stage 4 setting.

## Questions For Advisor

1. Is Stage 4 correctly framed as market-context-conditioned visual feature
   modulation rather than drawing more indicators into the image?
2. Should the primary context match the image window, so `I60` uses
   `BB60/MFI60/F&G60/RV60`?
3. Do 4-A concat, 4-B gating, 4-C gamma-only FiLM, and 4-D full FiLM match the
   intended ablation logic?
4. Should news/LLM context remain a second-phase extension after the structured
   numeric context comparison?
