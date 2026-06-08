# 5-9E Result: FinBERT + F&G Bounded FiLM

## English

### Experiment

- Model: Stage2 `I60/R20/ohlc_ma_vb` checkpoint loaded and frozen.
- Context: FinBERT headline sentiment aggregates plus F&G raw regime features.
- Context dimension: `83` features = `79` FinBERT numeric features + `4` F&G features.
- FiLM: bounded last-block FiLM.
- Scale: `0.02`.
- Seeds: `42, 43, 44, 45, 46`.
- Result bundle inspected:
  `/Users/jaehyeonjeong/Desktop/논문/5_9e_results`.

### Mean Result

| Model | Accuracy | ROC-AUC | AP | Brier | Predicted positive rate | Long-flat Sharpe | Long-short Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage2 `ohlc_ma_vb` baseline | 0.579320 | 0.584862 | 0.611256 | 0.274337 | 0.664400 | 3.442312 | 2.407759 |
| N8B F&G-only bounded FiLM | 0.580291 | 0.584930 | 0.611272 | 0.274004 | 0.660652 | 3.444515 | 2.393370 |
| 5-9D FinBERT-only bounded FiLM | 0.578487 | 0.586072 | 0.611943 | 0.272739 | 0.637196 | 3.400253 | 2.233801 |
| 5-9E FinBERT + F&G bounded FiLM | 0.580569 | 0.585843 | 0.611899 | 0.272701 | 0.639278 | 3.443674 | 2.319576 |

### Delta Reading

- Versus Stage2 baseline:
  - Accuracy: `+0.001249`
  - ROC-AUC: `+0.000981`
  - AP: `+0.000643`
  - Brier: `-0.001636`
  - Long-flat Sharpe: `+0.001362`
  - Long-short Sharpe: `-0.088184`
- Versus N8B F&G-only:
  - Accuracy: `+0.000278`
  - ROC-AUC: `+0.000913`
  - AP: `+0.000627`
  - Brier: `-0.001304`
  - Long-flat Sharpe: `-0.000841`
  - Long-short Sharpe: `-0.073795`

### Seed-Level Reading

| Seed | Stage2 Acc | F&G Acc | FinBERT Acc | FinBERT + F&G Acc | FinBERT + F&G vs Stage2 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.603053 | 0.603747 | 0.598196 | 0.602359 | -0.000694 |
| 43 | 0.574601 | 0.573213 | 0.576683 | 0.573907 | -0.000694 |
| 44 | 0.562804 | 0.566967 | 0.561416 | 0.566967 | +0.004164 |
| 45 | 0.562804 | 0.562804 | 0.565579 | 0.565579 | +0.002776 |
| 46 | 0.593338 | 0.594726 | 0.590562 | 0.594032 | +0.000694 |

### Interpretation

FinBERT + F&G is a small positive result, but not a strong performance jump.
It slightly improves the Stage2 baseline and very slightly improves F&G-only
on accuracy, ROC-AUC, AP, and Brier score. The gain is too small to claim that
FinBERT sentiment is a major new signal.

The useful interpretation is narrower: FinBERT sentiment can be combined with
F&G without destabilizing the frozen Stage2 + bounded FiLM protocol, and it
slightly reduces the overly high predicted-positive rate from the Stage2/F&G
models. However, it does not materially improve trading metrics or solve the
threshold-decision problem.

### Decision

5-9E closes the first FinBERT sentiment branch. For the thesis, report it as a
near-tie/small positive ablation:

- F&G remains the stronger compact regime source.
- FinBERT-only improves ranking/calibration but loses accuracy.
- FinBERT + F&G slightly improves accuracy over Stage2 and F&G-only, but the
  margin is too small to use as the final headline model.

The next useful Stage 5 work is not another broad scale grid. It is either:

1. conditional correction/regression analysis, or
2. explicit prompt/event labels if a richer, interpretable news feature is still
   needed.

## 한국어

### 실험

- 모델: Stage2 `I60/R20/ohlc_ma_vb` checkpoint load/freeze.
- Context: FinBERT headline sentiment aggregate + F&G raw regime feature.
- Context dimension: `83`개 = FinBERT numeric feature `79`개 + F&G feature `4`개.
- FiLM: bounded last-block FiLM.
- Scale: `0.02`.
- Seeds: `42, 43, 44, 45, 46`.
- 확인한 결과 폴더:
  `/Users/jaehyeonjeong/Desktop/논문/5_9e_results`.

### 평균 결과

| 모델 | Accuracy | ROC-AUC | AP | Brier | Predicted positive rate | Long-flat Sharpe | Long-short Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage2 `ohlc_ma_vb` baseline | 0.579320 | 0.584862 | 0.611256 | 0.274337 | 0.664400 | 3.442312 | 2.407759 |
| N8B F&G-only bounded FiLM | 0.580291 | 0.584930 | 0.611272 | 0.274004 | 0.660652 | 3.444515 | 2.393370 |
| 5-9D FinBERT-only bounded FiLM | 0.578487 | 0.586072 | 0.611943 | 0.272739 | 0.637196 | 3.400253 | 2.233801 |
| 5-9E FinBERT + F&G bounded FiLM | 0.580569 | 0.585843 | 0.611899 | 0.272701 | 0.639278 | 3.443674 | 2.319576 |

### 해석

5-9E는 작은 positive result다. Stage2 baseline보다 accuracy `+0.001249`,
ROC-AUC `+0.000981`, Brier `-0.001636`이다. F&G-only와 비교해도 accuracy는
`+0.000278`, ROC-AUC는 `+0.000913` 개선된다.

하지만 이 개선폭은 매우 작다. 따라서 논문에서 “FinBERT + F&G가 강하게
baseline을 이겼다”고 쓰기는 어렵다. 더 정확한 해석은 다음과 같다.

- F&G는 여전히 가장 compact한 regime source다.
- FinBERT-only는 accuracy는 낮지만 ROC-AUC/AP/Brier를 아주 작게 개선했다.
- FinBERT + F&G는 F&G의 안정성을 유지하면서 FinBERT의 ranking/calibration
  효과를 약간 더한 near-tie/small-positive ablation이다.
- predicted positive rate가 `0.639278`로 Stage2/F&G-only보다 낮아져 Up-bias는
  조금 줄었지만, trading metric은 강하게 좋아지지 않았다.

### 결론

5-9E로 FinBERT sentiment branch의 1차 실험은 닫을 수 있다. 최종 headline
model로 밀기에는 개선폭이 작지만, “뉴스 sentiment를 명시적으로 만든 뒤
F&G와 결합해도 frozen Stage2 + bounded FiLM 구조는 무너지지 않고, 작은
ranking/calibration 개선을 보인다”는 근거로 쓸 수 있다.

다음은 무작정 scale/grid를 더 돌리기보다 correction/regression 조건부 분석,
또는 필요 시 prompt/event label로 넘어가는 것이 합리적이다.
