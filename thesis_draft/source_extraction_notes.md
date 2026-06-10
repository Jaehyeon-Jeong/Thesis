# Source Extraction Notes

이 문서는 thesis 초안에 반영할 초기 교수님 자료와 현재 실험 결과의 연결점을 정리합니다. 본문 초안은 영어로 쓰지만, 이 파일은 작업용이므로 한국어로 유지합니다.

## 1. 유지할 핵심 Research Gap

`자료조사/Research Idea.docx`와 `자료조사/Thesis plan.docx`에서 유지할 핵심은 다음입니다.

1. 가격 차트를 단순한 hand-crafted signal 모음이 아니라 visual object로 보고 CNN이 직접 predictive pattern을 학습한다.
2. 기존 multimodal financial prediction은 가격/텍스트를 단순 concat하거나 black-box attention으로 결합하는 경우가 많아, context가 내부 price/chart representation을 어떻게 바꾸는지 설명하기 어렵다.
3. FiLM은 context가 feature-wise gamma/beta를 만들어 intermediate visual feature를 조절하므로, 성능뿐 아니라 modulation path를 해석할 수 있다.
4. 뉴스/시장심리/레짐 context는 가격 차트만으로 보기 어려운 외부 market state를 제공할 수 있다.

## 2. 현재 실험에 맞게 바뀐 부분

초기 계획과 현재 실험의 차이는 thesis에서 명확히 설명해야 합니다.

| 초기 계획 | 현재 확정/실험된 형태 | thesis에서의 처리 |
|---|---|---|
| Surge / crash / neutral regime classification | R20 forward return Up/Down binary classification | 실험 안정성과 Stage1/Stage2 pipeline 연결을 위해 binary directional task로 정리 |
| Broad social media and news | BTC headline news, F&G, FinBERT sentiment, OpenAI embedding, derivatives/leverage context | 데이터 availability와 leakage control이 가능한 context만 사용 |
| Full LLM reasoning | embedding / FinBERT / numeric context vector | LLM을 직접 predictor로 쓰지 않고 context feature extractor로 사용 |
| Direct LLM-to-FiLM main model | frozen Stage2 CNN + bounded last-block FiLM | strong visual baseline을 보존하는 baseline-preserving protocol로 정리 |
| General market regime | BTC R20 Up/Down with contextual correction analysis | 전체 accuracy와 조건부 correction을 분리해서 해석 |

## 3. 교수님 방향성과 연결되는 최종 설계

Stage4/Stage5의 최종 구조는 다음 문장으로 정리할 수 있습니다.

```text
The prediction query is fixed: given a BTC chart image, predict whether the
20-day forward return is positive. The changing part is the market context,
which is encoded as a vector and used to generate FiLM gamma/beta parameters
for the chart CNN feature representation.
```

즉 교수님 방향성의 핵심은 full VQA가 아니라 chart-image feature의 conditional modulation입니다.

```text
chart image -> Stage2 CNN feature
context vector -> MLP -> bounded last-block FiLM gamma/beta
modulated feature -> frozen classifier -> Up/Down prediction
```

## 4. 논문에서 강조할 Contribution 후보

현재 결과 기준으로 과장 없이 쓸 수 있는 contribution은 다음입니다.

1. Re-image style chart-CNN pipeline을 재현하고 BTC 자산군으로 확장했다.
2. BTC에서 image window/return horizon/image spec을 비교하여 `I60/R20/ohlc_ma_vb`가 강한 visual baseline임을 확인했다.
3. Scratch-trained context fusion/FiLM은 강한 visual baseline을 안정적으로 넘지 못한다는 negative result를 확인했다.
4. Stage2 pretrained CNN/classifier를 고정하고 context encoder + bounded last-block FiLM만 학습하는 baseline-preserving protocol을 제안/검증했다.
5. F&G, FinBERT+F&G, derivatives/leverage context는 작은 성능 변화와 조건부 correction을 보였으며, 특히 modulation이 보수적인 correction signal로 작동하는 경향을 보였다.

## 5. 논문에서 조심해야 할 부분

- "large performance improvement"라고 쓰면 안 됩니다. 전체 성능 gain은 작습니다.
- "LLM reasoning improves BTC prediction"이라고 쓰면 안 됩니다. 현재 Stage5는 LLM-derived or model-derived numeric context입니다.
- "FiLM strongly changes visual attention"이라고 쓰면 안 됩니다. gamma/beta 변화는 작고, 주로 final-block correction으로 해석하는 편이 맞습니다.
- 뉴스 원문은 재배포하지 않습니다. derived feature, embedding, sentiment summary만 논문/코드에서 다룹니다.
