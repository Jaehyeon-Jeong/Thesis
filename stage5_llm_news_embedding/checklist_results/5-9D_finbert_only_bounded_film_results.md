# 5-9D Result: FinBERT-Only Bounded FiLM

## English

### Experiment

- Model: Stage2 `I60/R20/ohlc_ma_vb` checkpoint loaded and frozen.
- Context: FinBERT headline sentiment aggregate features only.
- FiLM: bounded last-block FiLM.
- Scale: `0.02`.
- Seeds: `42, 43, 44, 45, 46`.
- Result bundle inspected:
  `/Users/jaehyeonjeong/Desktop/논문/5_9d_results`.

### Mean Result

| Model | Accuracy | ROC-AUC | AP | Brier | Predicted positive rate | Long-flat Sharpe | Long-short Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage2 `ohlc_ma_vb` baseline | 0.579320 | 0.584862 | 0.611256 | 0.274337 | 0.664400 | 3.442312 | 2.407759 |
| N8B F&G-only bounded FiLM | 0.580291 | 0.584930 | 0.611272 | 0.274004 | 0.660652 | 3.444515 | 2.393370 |
| 5-9D FinBERT-only bounded FiLM | 0.578487 | 0.586072 | 0.611943 | 0.272739 | 0.637196 | 3.400253 | 2.233801 |

### Seed-Level Reading

- Best seed: `42`, accuracy `0.598196`, ROC-AUC `0.618168`.
- Weakest seed: `44`, accuracy `0.561416`, predicted positive rate `0.699514`.
- Accuracy delta vs Stage2:
  - seed 42: `-0.004858`
  - seed 43: `+0.002082`
  - seed 44: `-0.001388`
  - seed 45: `+0.002776`
  - seed 46: `-0.002776`
- ROC-AUC improves slightly against Stage2 for every seed, but the gains are
  very small.

### Interpretation

FinBERT-only context does not improve the main accuracy target. It slightly
improves ranking/calibration metrics:

- ROC-AUC: `+0.001210` vs Stage2.
- Average precision: `+0.000687` vs Stage2.
- Brier score: `-0.001598` vs Stage2.

But the effect is not enough to claim a stronger final model:

- Accuracy is `-0.000833` below Stage2.
- Accuracy is `-0.001804` below N8B F&G-only FiLM.
- Long-short Sharpe falls from Stage2 `2.407759` to `2.233801`.

The context audit shows no missing features, so the weak result is not caused
by feature coverage. The likely issue is distribution shift and coarse signal:
test-period FinBERT sentiment/news-count features are much more positive than
train/validation, which pushes the model toward Up predictions. This is visible
in the predicted positive rate: `0.637196` mean vs true positive rate
`0.541291`, with seed `44` reaching `0.699514`.

### Decision

5-9D is a negative or near-tie result. It is useful for the thesis as evidence
that headline-level FinBERT tone alone is not a strong enough regime context
for the frozen Stage2 chart CNN. It supports moving either to:

1. FinBERT + F&G combined context, if one more compact sentiment/regime test is
   needed.
2. Prompt/event features that explicitly separate BTC relevance, direction,
   horizon, event type, and confidence.

## 한국어

### 실험

- 모델: Stage2 `I60/R20/ohlc_ma_vb` checkpoint를 load/freeze.
- Context: FinBERT headline sentiment aggregate feature만 사용.
- FiLM: bounded last-block FiLM.
- Scale: `0.02`.
- Seeds: `42, 43, 44, 45, 46`.
- 확인한 결과 폴더:
  `/Users/jaehyeonjeong/Desktop/논문/5_9d_results`.

### 평균 결과

| 모델 | Accuracy | ROC-AUC | AP | Brier | Predicted positive rate | Long-flat Sharpe | Long-short Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage2 `ohlc_ma_vb` baseline | 0.579320 | 0.584862 | 0.611256 | 0.274337 | 0.664400 | 3.442312 | 2.407759 |
| N8B F&G-only bounded FiLM | 0.580291 | 0.584930 | 0.611272 | 0.274004 | 0.660652 | 3.444515 | 2.393370 |
| 5-9D FinBERT-only bounded FiLM | 0.578487 | 0.586072 | 0.611943 | 0.272739 | 0.637196 | 3.400253 | 2.233801 |

### 해석

FinBERT-only는 accuracy 기준으로 Stage2 baseline을 넘지 못했다. 다만
ROC-AUC/AP/Brier는 아주 작게 개선되었다. 즉, 방향성 ranking/calibration은
조금 좋아졌지만, 실제 0.5 threshold decision을 더 잘 바꾸지는 못했다.

중요한 점은 feature coverage 문제가 아니라는 것이다. FinBERT context feature는
train/validation/test 모두 missing rate `0.0`이다. 문제는 test 구간에서
FinBERT positive/news-count feature가 train보다 크게 높아져 Up-bias를 만든
것으로 보인다. 평균 predicted positive rate는 `0.637196`이고, seed `44`는
`0.699514`까지 올라간다. 실제 positive rate는 `0.541291`이다.

### 결론

5-9D는 final 성능 모델 후보라기보다는 negative/near-tie ablation이다. 논문에는
"FinBERT headline tone만으로는 frozen Stage2 chart CNN을 의미 있게 개선하지
못했다"는 근거로 쓸 수 있다.

다음 선택지는 두 가지다.

1. FinBERT + F&G combined context를 한 번 확인한다.
2. 단순 긍/부정 tone이 아니라 BTC relevance, direction, horizon, event type,
   confidence를 분리하는 prompt/event feature로 넘어간다.
