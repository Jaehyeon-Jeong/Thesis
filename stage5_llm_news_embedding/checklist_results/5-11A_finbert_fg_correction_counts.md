# 5-11A: Stage2 vs FinBERT + F&G Correction Counts

## English

### Purpose

This is the first correction/regression check for Stage 5. It compares the
Stage2 visual-only `I60/R20/ohlc_ma_vb` baseline against the 5-9E
FinBERT + F&G bounded FiLM model on matched test samples and matched seeds.

### Result

| Seed | Stage2 Acc | FinBERT + F&G Acc | Corrections | Regressions | Net corrections | Changed predictions |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.603053 | 0.602359 | 9 | 10 | -1 | 19 |
| 43 | 0.574601 | 0.573907 | 26 | 27 | -1 | 53 |
| 44 | 0.562804 | 0.566967 | 34 | 28 | +6 | 62 |
| 45 | 0.562804 | 0.565579 | 12 | 8 | +4 | 20 |
| 46 | 0.593338 | 0.594032 | 14 | 13 | +1 | 27 |
| Total / Mean | 0.579320 | 0.580569 | 95 | 86 | +9 | 181 |

Definitions:

- Correction: Stage2 wrong -> FinBERT + F&G correct.
- Regression: Stage2 correct -> FinBERT + F&G wrong.
- Net corrections: corrections minus regressions.

### Interpretation

The 5-9E accuracy gain is real but small. Across all five seeds and `7,205`
matched test decisions, the model creates `95` corrections and `86`
regressions, for net `+9` corrected decisions.

This matches the mean accuracy delta:

- Stage2 mean accuracy: `0.579320`.
- FinBERT + F&G mean accuracy: `0.580569`.
- Accuracy delta: `+0.001249`.

The model changes only `181 / 7205 = 2.51%` of decisions. This confirms that
bounded FiLM acts conservatively: it slightly adjusts Stage2 decisions rather
than replacing the visual model.

### Thesis Reading

This is not a large performance improvement. The defensible claim is:

> FinBERT + F&G produces a small net correction over the frozen visual
> baseline. The gain is modest, but the correction/regression table shows that
> the context branch can alter a small number of decisions in the right
> direction without destabilizing the Stage2 chart CNN.

The next step is to inspect the corrected/regressed samples, especially:

- seed `44`, where net correction is `+6`;
- seed `45`, where net correction is `+4`;
- high FinBERT sentiment or high F&G regime days;
- probability changes near the decision boundary.

## 한국어

### 목적

Stage5의 첫 correction/regression 확인이다. Stage2 visual-only
`I60/R20/ohlc_ma_vb` baseline과 5-9E `FinBERT + F&G` bounded FiLM 모델을
같은 test sample, 같은 seed 기준으로 비교했다.

### 결과

| Seed | Stage2 Acc | FinBERT + F&G Acc | Corrections | Regressions | Net corrections | Changed predictions |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.603053 | 0.602359 | 9 | 10 | -1 | 19 |
| 43 | 0.574601 | 0.573907 | 26 | 27 | -1 | 53 |
| 44 | 0.562804 | 0.566967 | 34 | 28 | +6 | 62 |
| 45 | 0.562804 | 0.565579 | 12 | 8 | +4 | 20 |
| 46 | 0.593338 | 0.594032 | 14 | 13 | +1 | 27 |
| Total / Mean | 0.579320 | 0.580569 | 95 | 86 | +9 | 181 |

정의:

- Correction: Stage2는 틀렸지만 FinBERT + F&G가 맞춘 경우.
- Regression: Stage2는 맞췄지만 FinBERT + F&G가 틀린 경우.
- Net corrections: corrections - regressions.

### 해석

5-9E의 accuracy 개선은 실제로 존재하지만 매우 작다. 전체 `5 seeds x 1441`
test decision, 즉 `7,205`개 matched decision에서 correction `95`, regression
`86`, net correction `+9`다.

이는 평균 accuracy delta와 일치한다.

- Stage2 mean accuracy: `0.579320`.
- FinBERT + F&G mean accuracy: `0.580569`.
- Accuracy delta: `+0.001249`.

또한 모델이 바꾼 prediction은 `181 / 7205 = 2.51%`뿐이다. 즉 bounded FiLM은
visual model을 크게 뒤집는 구조가 아니라, Stage2 decision을 아주 보수적으로
조정하는 구조로 작동했다.

### 논문용 해석

큰 성능 향상은 아니다. 방어 가능한 표현은 다음이다.

> FinBERT + F&G는 frozen visual baseline 대비 작은 net correction을 만든다.
> 개선폭은 작지만, context branch가 Stage2 chart CNN을 무너뜨리지 않고 일부
> decision을 올바른 방향으로 바꿀 수 있음을 correction/regression table로
> 확인했다.

다음으로 볼 샘플은 다음이다.

- net correction이 `+6`인 seed `44`;
- net correction이 `+4`인 seed `45`;
- FinBERT sentiment 또는 F&G regime이 강한 날;
- decision boundary 근처에서 probability가 바뀐 샘플.
