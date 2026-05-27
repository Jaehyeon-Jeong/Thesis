# 교수님 1페이지 진행 보고: Stage 1-4 현황과 다음 실험

Date: 2026-05-27

## 핵심 진행 상황

| Stage | 현재 결과 | 판단 |
|:---|:---|:---|
| Stage 1 | Re-image/Stock_CNN 원 pipeline 재현 진행. `I20/R60` fast diagnostic: accuracy `0.5312`, ROC-AUC `0.5298`. | baseline protocol과 five-seed convention 기준으로 유지. Strict batch-128/five-seed는 later. |
| Stage 2 | BTC로 자산군 변경. seed42 36-run grid 완료. Selected five-seed는 `I20/R20`, `I60/R20` 수행. Best: `I60/R20/ohlc_ma_vb`, accuracy mean `0.5793`, ROC-AUC mean `0.5849`. | `I60/R20/ohlc_ma_vb`를 Stage 4 visual baseline으로 고정. |
| Stage 3 | Stage 2 best에 Linear adapter 추가. seed42 accuracy `0.5413`, ROC-AUC `0.5221`로 하락. | 단순 parameter 추가는 효과 없음. Negative/simple-parameter ablation으로 정리. |
| Stage 4 | market context를 concat/gating/FiLM으로 붙이는 ablation 진행. v1 best `film_full`도 Stage 2보다 낮음: accuracy mean `0.5510`. | context 가능성은 있으나, 현재 full-FiLM 주입은 seed-unstable. |

## Stage 2 선택 근거

- Stage 2는 먼저 `I5/I20/I60 x R5/R20/R60 x image spec 4개`를 seed `42`로 screening했습니다.
- `I5`는 평균적으로 majority-class baseline보다 낮았고, best `I5`도 accuracy 약 `0.524`였습니다.
- 그래서 five-seed는 `I20/R20`, `I60/R20`에 집중했고, 결과적으로 `I60/R20`만 robust하게 살아남았습니다.

## Stage 4 현재 해석

- 교수님 방향은 “강한 chart-image CNN은 유지하고, market context가 visual feature를 조건부로 조절하는지 확인”하는 것으로 해석했습니다.
- Context는 chart image에 그려 넣지 않고 별도 numeric vector로 사용했습니다.
- 실험한 context:
  - 외부 context: Fear & Greed.
  - OHLCV-derived context: Bollinger %B, Bollinger bandwidth, MFI, realized volatility.
- 결과적으로 `ohlc_ma_vb` image 자체가 이미 많은 OHLCV-derived technical information을 담고 있어, BB/MFI/RV는 중복 정보 또는 noise가 될 가능성이 있습니다.
- F&G는 chart-derived 정보가 아니어서 가장 깨끗한 외부 regime/sentiment context입니다. 다만 현재 full-FiLM에서는 seed별 collapse가 발생했습니다.

## 중요 결과

| Experiment | Result | Interpretation |
|:---|:---|:---|
| Visual-only `I60/R20/ohlc_ma_vb` | seed42 accuracy `0.6031` 재현 | Stage 4 sample pipeline 자체는 문제 아님. |
| Visual-only `I60/R20/ohlc` | accuracy `0.5420`, predicted-positive rate `0.9632` | plain OHLC는 약함. MA+VB image가 중요한 signal. |
| `ohlc + all context + full FiLM`, five seeds | accuracy mean `0.5574`, ROC-AUC `0.5519` | seed42 개선은 robust하지 않음. |
| `ohlc_ma_vb + F&G-only + full FiLM`, five seeds | accuracy mean `0.5524`, ROC-AUC `0.5465`; seed 42/45/46은 양호, seed 43/44는 Up collapse | F&G 가능성은 있으나 full-FiLM 구조가 불안정. |

## 다음 실험 제안

다음은 더 많은 context 추가가 아니라 **bounded/residual last-block FiLM**으로 FiLM 구조를 안정화하는 실험이 필요합니다.

```text
gamma = 1 + s * tanh(raw_gamma)
beta  = s * tanh(raw_beta)
```

FiLM은 먼저 마지막 high-level CNN block에만 적용합니다. 목적은 low-level chart feature를 보존하고, F&G가 high-level visual evidence만 제한적으로 조절하게 만드는 것입니다.

다음 비교:

```text
Stage 2 visual baseline:
  I60/R20/ohlc_ma_vb

Stage 4 stabilized FiLM:
  I60/R20/ohlc_ma_vb + F&G-only + bounded last-block FiLM
```

실행은 기존 protocol과 맞춰 seed `42, 43, 44, 45, 46` five-seed로 진행하겠습니다.
