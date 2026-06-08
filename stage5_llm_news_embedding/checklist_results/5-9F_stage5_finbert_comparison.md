# 5-9F Comparison: Stage2 vs F&G vs FinBERT vs FinBERT + F&G

## English

### Compared Models

All context-FiLM rows use the same core protocol: Stage2 `I60/R20/ohlc_ma_vb`
checkpoint loaded, visual CNN/classifier frozen, bounded last-block FiLM,
scale `0.02`, seeds `42-46`.

| Label | Context |
|---|---|
| Stage2 baseline | No context, visual-only `ohlc_ma_vb` |
| N8B F&G-only | F&G raw regime features |
| 5-9D FinBERT-only | headline-level FinBERT sentiment aggregates |
| 5-9E FinBERT + F&G | FinBERT sentiment aggregates + F&G raw regime features |

### Mean Metric Comparison

| Model | Accuracy | ROC-AUC | AP | Brier | Predicted positive rate |
|---|---:|---:|---:|---:|---:|
| Stage2 baseline | 0.579320 | 0.584862 | 0.611256 | 0.274337 | 0.664400 |
| N8B F&G-only | 0.580291 | 0.584930 | 0.611272 | 0.274004 | 0.660652 |
| 5-9D FinBERT-only | 0.578487 | 0.586072 | 0.611943 | 0.272739 | 0.637196 |
| 5-9E FinBERT + F&G | 0.580569 | 0.585843 | 0.611899 | 0.272701 | 0.639278 |

### Main Takeaway

The combined FinBERT + F&G model is the best row by mean accuracy among these
four, but only by a very small margin:

- `+0.001249` accuracy over Stage2 baseline.
- `+0.000278` accuracy over F&G-only.
- `+0.000981` ROC-AUC over Stage2 baseline.
- `+0.000913` ROC-AUC over F&G-only.

Correction/regression counting against Stage2 confirms the same small effect:

- Corrections: `95`.
- Regressions: `86`.
- Net corrections: `+9`.
- Changed predictions: `181 / 7205 = 2.51%`.

This does not support a strong performance-improvement claim. It does support a
more cautious thesis claim: explicit news sentiment can be added as a numeric
context without destabilizing the baseline-preserving FiLM structure, but
headline-level sentiment is still too weak/coarse to be the main source of
performance improvement.

### Research Decision

Stage 5 should not continue with broad embedding or FinBERT scale grids. The
remaining meaningful work is:

1. correction/regression analysis to see where FinBERT + F&G changes decisions;
2. if more news work is needed, prompt/event labels that separate BTC relevance,
   direction, event type, horizon, and confidence.

## 한국어

### 비교 모델

Context-FiLM 모델은 모두 같은 핵심 protocol을 사용한다. Stage2
`I60/R20/ohlc_ma_vb` checkpoint를 load하고, visual CNN/classifier는 freeze,
bounded last-block FiLM, scale `0.02`, seeds `42-46`이다.

| Label | Context |
|---|---|
| Stage2 baseline | context 없는 visual-only `ohlc_ma_vb` |
| N8B F&G-only | F&G raw regime feature |
| 5-9D FinBERT-only | headline-level FinBERT sentiment aggregate |
| 5-9E FinBERT + F&G | FinBERT sentiment aggregate + F&G raw regime feature |

### 평균 지표 비교

| Model | Accuracy | ROC-AUC | AP | Brier | Predicted positive rate |
|---|---:|---:|---:|---:|---:|
| Stage2 baseline | 0.579320 | 0.584862 | 0.611256 | 0.274337 | 0.664400 |
| N8B F&G-only | 0.580291 | 0.584930 | 0.611272 | 0.274004 | 0.660652 |
| 5-9D FinBERT-only | 0.578487 | 0.586072 | 0.611943 | 0.272739 | 0.637196 |
| 5-9E FinBERT + F&G | 0.580569 | 0.585843 | 0.611899 | 0.272701 | 0.639278 |

### 핵심 결론

FinBERT + F&G가 이 네 가지 중 평균 accuracy는 가장 높다. 하지만 margin은
작다.

- Stage2 baseline 대비 accuracy `+0.001249`.
- F&G-only 대비 accuracy `+0.000278`.
- Stage2 baseline 대비 ROC-AUC `+0.000981`.
- F&G-only 대비 ROC-AUC `+0.000913`.

Stage2와 직접 correction/regression을 세어도 효과는 같은 방향으로 작다.

- Corrections: `95`.
- Regressions: `86`.
- Net corrections: `+9`.
- Changed predictions: `181 / 7205 = 2.51%`.

따라서 강한 성능 개선이라고 쓰면 안 된다. 더 방어 가능한 표현은 다음이다.

> 명시적인 headline-level news sentiment를 F&G와 결합해도
> baseline-preserving frozen Stage2 + bounded FiLM 구조는 무너지지 않았고,
> accuracy/ranking/calibration을 아주 작게 개선했다. 그러나 headline-level
> sentiment는 아직 성능 향상의 핵심 source라고 보기에는 약하다.

### 연구 판단

Stage5에서 broad embedding/FinBERT scale grid를 더 도는 것은 우선순위가 낮다.
남은 의미 있는 작업은 두 가지다.

1. correction/regression 분석으로 FinBERT + F&G가 어느 샘플을 바꾸는지 확인.
2. 뉴스 실험을 더 한다면 단순 sentiment가 아니라 BTC relevance, direction,
   event type, horizon, confidence를 분리하는 prompt/event label로 넘어가기.
