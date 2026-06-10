# Missing Evidence and Optional Experiments

이 문서는 thesis draft v0 이후에 확인하거나 보강하면 좋은 항목을 정리합니다.
현재 초안은 이미 작성 가능하지만, 아래 항목을 채우면 논문 방어력이 좋아집니다.

## 1. 반드시 확인해야 할 자료

### 1.1 Bibliography / citation verification

필수입니다. 현재 `thesis_draft.md`의 references는 working placeholder입니다.
최종 제출 전 아래 source의 정확한 bibliographic entry를 확인해야 합니다.

- Re-Imag(in)ing Price Trends.
- FiLM: Visual Reasoning with a General Conditioning Layer.
- Grad-CAM.
- Expected Returns and Large Language Models.
- Can ChatGPT Forecast Stock Price Movements?
- FinBERT / ProsusAI model source.
- OpenAI `text-embedding-3-small` documentation.
- F&G source.
- CFTC/CME Bitcoin futures source.
- OFR FSI source.
- BTC OHLCV dataset source.

필요 작업:

```text
Create final GOST-compatible bibliography entries.
Make sure every reference in the list is cited in the thesis body.
Remove any source that is not actually cited.
```

### 1.2 Final department formatting confirmation

MIPT rulebook 기준은 정리했지만, 학과/department template가 추가 요구를 줄 수
있습니다.

확인 필요:

- final title page wording,
- English thesis caption wording: `Figure` / `Table` 사용 가능 여부,
- bibliography style for English references,
- whether an AI-use declaration is required by the department.

### 1.3 Stage1 final evidence

현재 Stage1은 pipeline reproduction evidence로 충분하지만, strict full
five-seed reproduction은 없습니다. 초안에서는 Stage1을 "pipeline reference"로만
쓰고, 강한 성능 claim은 하지 않는 것이 안전합니다.

필요할 때만 추가:

```text
Stage1 I20/R5, I20/R20, I20/R60 full result table.
```

하지만 thesis 핵심은 BTC Stage2-Stage5이므로 필수는 아닙니다.

## 2. 논문을 더 설득력 있게 만드는 보강 분석

### 2.0 Part 6 interpretation status

Part 6의 현재 역할은 새 SOTA claim을 만드는 것이 아니라, Parts 1-5에서
확정된 결과를 방어 가능한 해석으로 바꾸는 것입니다. 지금 Part 6에서 이미
쓸 수 있는 근거와 아직 보강하면 좋은 근거는 아래처럼 구분합니다.

이미 충분히 쓸 수 있는 내용:

- 강한 chart-only baseline은 쉽게 이기기 어렵다.
- technical/chart-derived context는 대부분 redundant하다.
- TF-IDF/SVD와 generic dense embedding은 큰 hard-accuracy 개선을 만들지
  못했고, FinBERT처럼 sentiment-aligned text feature가 더 설득력 있다.
- FinBERT+F&G는 전체 gain은 작지만 Stage2 uncertain sample, greed regime,
  high-news-count bucket에서 조건부 net correction이 더 좋다.
- Grad-CAM/gamma-beta export 결과상 FiLM은 강한 feature rewrite가 아니라
  conservative calibration으로 작동한다.

보강하면 좋은 내용:

- derivatives/leverage context의 "hotter leverage regime에서 weak bullish
  prediction을 낮춘다"는 주장은 현재 N16-5 selected-sample evidence와
  transition summary로는 쓸 수 있지만, Part 6에서 더 강하게 쓰려면
  N14-B2-B6 bucket table이 필요하다.
- gamma/beta가 어떤 context feature에 반응했는지는 아직 channel-level
  correlation/top-k 분석으로 더 보강할 수 있다. 다만 현재 gamma/beta magnitude가
  작으므로 "strong amplification/suppression"이 아니라 "small bounded
  modulation"으로 써야 한다.

따라서 Part 6 작성 기준은 다음과 같습니다.

```text
Performance claim: small and conditional.
Interpretability claim: diagnostic and model-internal, not causal.
Robustness claim: strongest for FinBERT+F&G buckets; derivatives needs
additional bucketed support if stated strongly.
```

### 2.1 Stage4 N14-B2-B6 conditional regime analysis

우선순위: 높음.

이미 checklist에 남아 있는 보강 분석입니다. 새 모델 훈련이 아니라 post-training
analysis입니다.

대상:

```text
N16 ohlc_vb + funding_plus_cftc_oi
```

분석 bucket:

- high funding vs low funding,
- high CFTC OI / OI-change vs low,
- Stage2 uncertain samples,
- Stage2 false-Up cases,
- high leverage/funding pressure periods.

기대 효과:

```text
N16 결과를 단순 +0.002 accuracy가 아니라,
"derivatives/leverage context suppresses weak bullish calls under hotter
leverage regimes"라는 조건부 해석으로 강화.
```

이 분석은 논문 Chapter 7에 바로 들어갈 수 있습니다.

### 2.2 Stage5 5-12 figure selection cleanup

우선순위: 중간.

5-12 Grad-CAM/gamma-beta export는 이미 완료됐지만, thesis에 넣을 figure는
선별해야 합니다.

필요 작업:

- correction sample 1-2개,
- regression sample 1개,
- Stage2 Grad-CAM vs Stage5 Grad-CAM,
- context values,
- prob_up change,
- gamma/beta summary.

목적:

```text
News/F&G FiLM is a conservative calibration layer.
```

주의:

```text
Do not claim that Grad-CAM proves strong visual attention change.
```

### 2.3 N16 figure selection cleanup

우선순위: 중간.

N16 derivatives/leverage 쪽도 figure를 1-2개 고르면 좋습니다.

추천 figure:

- one false-Up correction where N16 lowers prob_up,
- one true-Up regression caused by the same bearish correction mechanism,
- table showing funding/OI context values.

목적:

```text
Show that the same FiLM correction mechanism can help or hurt depending on the
future return.
```

## 3. 선택 실험: 지금은 하지 않아도 됨

### 3.1 More prompt/event-label news experiment

우선순위: 낮음.

Stage5 5-13에서 이미 닫았습니다. supervisor가 "LLM/news"를 제목에 더 강하게
넣고 싶어할 때만 실행합니다.

구조:

```text
headline -> fixed prompt -> BTC relevance, direction, event type, horizon,
confidence -> numeric context -> frozen bounded FiLM
```

위험:

- prompt output reproducibility,
- manual validation burden,
- more experiments without guaranteed improvement.

### 3.2 More FiLM scale/unfreeze grids

우선순위: 낮음.

이미 larger scale / classifier unfreeze 계열은 큰 개선을 만들지 못했습니다.
현재 논문의 결론은 "bounded calibration"이므로, 더 넓은 grid는 초안을 늦출
가능성이 큽니다.

### 3.3 Full Stage2 five-seed grid for all I/R combinations

우선순위: 낮음 to 중간.

교수님이 "왜 I60/R20만 가져왔나?"라고 강하게 물을 경우에만 필요합니다.
현재는 selected five-seed robustness check로 충분히 방어 가능합니다.

## 4. 현재 초안에서 조심해야 할 claim

쓰면 안 되는 표현:

```text
LLM/news strongly improves BTC prediction.
FiLM significantly outperforms the chart-only baseline.
The final task is market-regime classification.
Gamma/beta changes prove causal market interpretation.
Grad-CAM proves the model changed its attention.
```

써도 되는 표현:

```text
The strongest chart-only baseline is difficult to improve.
Context-FiLM provides small conditional corrections.
F&G and FinBERT+F&G are useful as bounded calibration context.
Derivatives/leverage context improves the same-image ohlc_vb baseline modestly.
Frozen Stage2 + bounded final-block FiLM is more stable than scratch-trained
context fusion.
```

## 5. Recommended next writing order

1. Add exact citations and convert reference placeholders.
2. Select 3-5 final tables:
   - Stage2 visual baselines,
   - Stage4 structured context summary,
   - N16 same-image result,
   - Stage5 news/sentiment result,
   - correction/regression summary.
3. Select 2-4 final figures:
   - pipeline architecture,
   - Stage2/FiLM architecture,
   - one Stage5 Grad-CAM correction panel,
   - one N16 derivatives correction panel.
4. Run or finish N14-B2-B6 only if we need stronger conditional evidence.
5. Convert Markdown to DOCX using `VKR_format_requirements.md`.
