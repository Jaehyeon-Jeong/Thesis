# Stage 4 실험 방향 확정 요청 보고서

작성일: 2026-05-21  
GitHub: `https://github.com/Jaehyeon-Jeong/Thesis`

## 1. 보고 목적

현재 논문 실험은 Stage 3까지 구조화하여 진행했습니다. 다음 핵심 단계는 Stage 4
FiLM 실험인데, FiLM을 어떤 condition source와 결합할지에 따라 연구 질문과 실험
범위가 달라집니다.

따라서 본 보고서는 지금까지의 진행 상황을 요약하고, 교수님께서 말씀해주신 방향성에
맞춰 가능한 Stage 4 실험 옵션을 나눈 뒤, 어떤 옵션을 main experiment로 확정할지
확인받기 위한 문서입니다.

Stage 4 방향이 확정되면, 해당 옵션을 기준으로 바로 구현 및 Kaggle 실험을 진행하고,
일주일 이내에 실험 결과 보고와 논문 초안 작성까지 진행할 계획입니다.

## 2. 현재까지의 진행 상황

### Stage 1. Re-image 논문 CNN 파이프라인 재현

목적:
- *Re-Imag(in)ing Price Trends* 논문의 chart-image CNN 파이프라인을 재현합니다.
- `lich99/Stock_CNN`의 핵심 모델 구조를 최대한 따라 구현했습니다.

현재 상태:
- Stage 1 코드 구조, checklist, Kaggle runner, Grad-CAM 생성 구조를 작성했습니다.
- Kaggle full run은 시간이 오래 걸려 일부 fast diagnostic 결과만 확보했습니다.
- 현재 보존된 usable full artifact는 `I20/R60`, seed `42`, fast Kaggle diagnostic입니다.

보존된 결과:
- `I20/R60`, seed `42`
- Accuracy: `0.5312`
- Majority-class accuracy: `0.5408`
- ROC-AUC: `0.5298`
- Test rows: `1,376,215`

제한:
- `I20/R20` local archive는 full result가 아니라 validation-smoke output입니다.
- `I20/R5`는 아직 로컬에 최종 보존되어 있지 않습니다.
- Strict batch size `128`, five-seed reproduction, 최종 Figure-13-style Grad-CAM은
  later work로 남겨두었습니다.

### Stage 2. BTC 자산군 확장

목적:
- Re-image 방식의 chart-image CNN을 BTC OHLCV 데이터에 적용합니다.
- BTC에서는 I5/I20/I60 image window와 R5/R20/R60 return horizon을 모두 생성했습니다.

현재 상태:
- BTC OHLCV audit, image generation, label/split/normalization, CNN training,
  prediction, trading metrics, Grad-CAM까지 구현했습니다.
- Single-seed 36-run grid를 실행했습니다:
  `I5/I20/I60 x R5/R20/R60 x 4 image specs`, seed `42`.

대표 결과:
- Best single-seed configuration: `I60/R20/ohlc_ma_vb`
- Accuracy: `0.603053`
- Majority-class accuracy: `0.541291`
- ROC-AUC: `0.616950`

남은 작업:
- Five-seed stability check는 later work로 남겨두었습니다.

### Stage 3. Re-image + Linear

목적:
- FiLM 이전에 단순한 Linear adapter를 붙인 비교군을 만듭니다.
- FiLM이 좋아졌을 때 단순히 parameter 증가 때문인지 확인하기 위한 bridge/ablation입니다.

구조:
- Stage 2 BTC CNN pipeline은 그대로 유지합니다.
- CNN feature 뒤에 bias-free Linear adapter/head를 추가했습니다.
- Naive `Linear(feature_dim, feature_dim)`는 I60에서 계산량이 너무 커서 제외하고,
  `adapter_dim=128` 방식으로 구현했습니다.

현재 상태:
- Stage 3 코드, Kaggle runner, result viewer를 구현했습니다.
- Stage 2 best configuration 하나에 대해 preliminary test를 완료했습니다.

대표 결과:
- Configuration: `I60/R20/ohlc_ma_vb`, seed `42`, adapter dim `128`
- Stage 2 baseline accuracy: `0.603053`
- Stage 3 Linear accuracy: `0.541291`
- Stage 3 Linear majority accuracy: `0.541291`
- Stage 3 Linear ROC-AUC: `0.522101`

현재 해석:
- 첫 diagnostic 기준으로 Linear는 Stage 2 best model을 개선하지 못했고,
  majority-class-level accuracy로 하락했습니다.
- 나머지 Stage 3 grid와 five-seed stability check는 later work입니다.

### Stage 4. FiLM

현재 상태:
- Stage 4 folder, checklist, workflow diagram, source map scaffold를 만들었습니다.
- FiLM reference는 `ethanjperez/film`을 기준으로 확인했습니다.
- 현재 설계상 FiLM 삽입 위치는 다음과 같습니다:

```text
Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d
```

핵심 미확정 사항:
- FiLM에 어떤 condition source를 넣을 것인지가 아직 확정되지 않았습니다.
- 이 부분을 교수님께 확인받은 뒤 main experiment를 진행하고자 합니다.

## 3. 제가 이해한 교수님 방향성

교수님께서 말씀해주신 방향성은 다음처럼 이해했습니다.

1. Re-image 기반 CNN baseline은 이미 강한 baseline입니다. 따라서 “차트 이미지가
   예측력이 있는가?” 자체가 핵심 질문은 아닙니다.

2. 핵심 질문은 같은 차트 패턴이라도 시장 맥락에 따라 의미가 달라질 수 있으므로,
   시장 맥락 정보를 condition으로 넣으면 예측 성능이나 robustness가 좋아지는지
   확인하는 것입니다.

3. FiLM을 쓰는 이유는 단순히 성능 향상만이 아니라, market context가 CNN feature를
   어떻게 조절했는지 `gamma/beta`를 통해 분석할 수 있다는 해석력 때문입니다.

4. 다만 FiLM만 사용해서 성능이 좋아졌다고 주장하면 부족하므로, 일반 concat 방식과
   attention/gating 방식과 비교하는 ablation이 필요합니다. 그래야 “왜 FiLM인가?”에
   대한 타당성을 확보할 수 있습니다.

5. 원래 FiLM 논문은 image + text question 구조였지만, 현재 금융 예측에서는 질문이
   사실상 “이후 n일이 Up인가 Down인가?”로 고정되어 있으므로, 반드시 RNN/LLM 질문
   encoder가 필요한 것은 아닙니다. 대신 F&G, Bollinger Band, MFI, volatility 같은
   structured market context를 condition으로 사용할 수 있습니다.

## 4. Stage 4 실험 옵션

아래 옵션 중 어떤 방향이 교수님께서 생각하신 Stage 4 main experiment에 가장 가까운지
확인받고 싶습니다.

### Option A. FiLM-only control

연구 질문:
- CNN에 FiLM layer를 붙였을 때 구조적으로 정상 작동하는가?
- `gamma/beta` 저장, Grad-CAM 연결, output export가 잘 되는가?

구조:

```text
Chart image -> Stock CNN + FiLM -> Up/Down
```

장점:
- 빠르게 구현할 수 있습니다.
- FiLM layer insertion과 logging을 확인하는 control로 적합합니다.

한계:
- 외부 시장 맥락 condition이 없으므로, 교수님께서 말씀하신 “시장 맥락을 넣는 FiLM”과는
  다소 거리가 있습니다.
- 성능이 좋아져도 market context 효과라고 주장하기 어렵습니다.

용도:
- Main experiment라기보다는 implementation sanity check/control로 적절합니다.

### Option B. Structured market context + FiLM

연구 질문:
- F&G, Bollinger Band, MFI, volatility 같은 구조화된 시장 맥락을 condition으로
  넣으면 chart-image CNN의 예측 성능과 해석력이 좋아지는가?

구조:

```text
Chart image
    -> Stock CNN feature

Market context
    -> F&G / Bollinger Band / MFI / volatility / horizon
    -> MLP condition encoder
    -> gamma/beta
    -> CNN feature modulation
    -> Up/Down
```

비교 실험:

```text
1. CNN-only baseline
2. CNN + context concat
3. CNN + context attention/gating
4. CNN + context FiLM gamma-only
5. CNN + context FiLM full gamma/beta
```

장점:
- 교수님께서 말씀하신 “시장 맥락을 조건으로 넣는 FiLM”과 가장 잘 맞습니다.
- concat/attention과 비교할 수 있어 FiLM의 타당성을 실험적으로 방어할 수 있습니다.
- `gamma/beta`를 시장 regime별로 분석할 수 있어 해석력 논리를 만들기 좋습니다.
- 6월 전까지 실험 가능한 범위로 보입니다.

주의점:
- F&G 데이터 source와 BTC 날짜 정렬을 확인해야 합니다.
- Bollinger Band, MFI는 BTC OHLCV에서 계산 가능하지만, 차트 이미지에도 일부 반영된
  정보를 수치 context로 다시 넣는 것이므로 “추가 외부 정보”라기보다는
  “명시적 market context”로 설명해야 합니다.

제 추천:
- Stage 4 main experiment로 가장 적합하다고 판단합니다.

### Option C. Horizon-aware FiLM

연구 질문:
- 같은 차트라도 R5/R20/R60 예측 horizon에 따라 봐야 하는 feature가 달라지는가?
- Horizon `n`을 condition으로 넣으면 하나의 FiLM model이 horizon별로 feature를
  다르게 조절할 수 있는가?

구조:

```text
Chart image + horizon n
    -> horizon embedding / MLP
    -> gamma/beta
    -> CNN feature modulation
    -> n일 뒤 Up/Down
```

장점:
- 교수님께서 말씀하신 “질문은 하나, 이후 n일은 Up or Down?”이라는 표현과 연결됩니다.
- 같은 chart image라도 예측 기간에 따라 다르게 해석한다는 논리가 가능합니다.

한계:
- 기존 Stage 1/2/3은 horizon별 모델을 따로 두는 구조였으므로, 실험 구조 변화가 큽니다.
- 6월 전까지 main으로 진행하기에는 Option B보다 위험합니다.

용도:
- Option B 이후 확장 실험으로 적합해 보입니다.

### Option D. News + non-LLM encoder + FiLM

연구 질문:
- BTC news text를 LLM이 아닌 text encoder로 condition vector로 바꾼 뒤 FiLM에 넣으면
  chart-image CNN이 개선되는가?

구조:

```text
Chart image
News text
    -> non-LLM text encoder
    -> gamma/beta
    -> CNN feature modulation
    -> Up/Down
```

장점:
- 원래 FiLM 논문의 image + language condition 구조에 더 가깝습니다.
- News condition이 들어가기 때문에 multimodal extension으로 설명하기 좋습니다.

한계:
- News publication time, date alignment, 중복 기사, missing day, leakage 문제가 큽니다.
- 6월 전까지 안정적인 실험 결과를 만들기에는 부담이 큽니다.

용도:
- 후속 확장 또는 second-stage experiment로 적합합니다.

### Option E. News + LLM encoder + FiLM

연구 질문:
- LLM이 만든 news representation을 condition으로 넣으면 FiLM-based chart model이
  개선되는가?

구조:

```text
News text
    -> LLM encoder / LLM embedding / LLM summary
    -> condition vector
    -> gamma/beta
    -> CNN feature modulation
```

장점:
- 최신 multimodal/LLM extension으로 확장성이 있습니다.

한계:
- LLM model version, prompt, cache, reproducibility, cost/runtime 문제가 큽니다.
- 논점이 “FiLM의 타당성”보다 “LLM 사용”으로 이동할 위험이 있습니다.
- 6월 전 main experiment로는 가장 위험합니다.

용도:
- Future work 또는 마지막 확장 실험으로 두는 것이 안전해 보입니다.

## 5. 제가 제안드리는 Stage 4 확정안

제가 이해한 교수님 방향성에 가장 가까운 main experiment는 Option B라고 판단합니다.

제안:

```text
Stage 4 main:
BTC CNN + structured market context conditioning
```

Condition candidates:

```text
F&G Index
Bollinger %B
Bollinger Bandwidth
MFI
realized volatility
volume z-score
possibly return horizon
```

Model comparison:

```text
1. CNN-only baseline
2. CNN + context concat
3. CNN + context attention/gating
4. CNN + context FiLM gamma-only
5. CNN + context FiLM full gamma/beta
```

Interpretability analysis:

```text
1. market context별 gamma/beta 변화
2. Up 예측과 Down 예측의 gamma/beta 차이
3. 맞춘 예측과 틀린 예측의 gamma/beta 차이
4. F&G/Bollinger/MFI regime별로 어떤 CNN block/channel이 강조되는지
5. Grad-CAM과 gamma/beta를 같은 sample/date 기준으로 연결한 분석
```

이 방향은 다음 세 가지를 동시에 만족합니다.

1. 기존 Re-image/BTC CNN baseline을 유지합니다.
2. 시장 맥락을 condition으로 넣습니다.
3. FiLM의 장점인 `gamma/beta` 기반 해석력을 실험 결과에 포함할 수 있습니다.

## 6. 교수님께 확인드리고 싶은 질문

Stage 4 main experiment를 아래 중 어떤 방향으로 확정하면 좋을지 확인 부탁드립니다.

1. **Option A: FiLM-only control**
   - FiLM layer 자체를 CNN에 붙이고 작동 여부와 gamma/beta export를 확인하는 방향

2. **Option B: Structured market context + FiLM**
   - F&G, Bollinger Band, MFI, volatility 같은 구조화된 시장 맥락을 condition으로 넣고,
     concat/attention/FiLM ablation까지 비교하는 방향

3. **Option C: Horizon-aware FiLM**
   - 예측 horizon `n`을 condition으로 넣어 R5/R20/R60에 따라 feature를 다르게 조절하는 방향

4. **Option D: News + non-LLM encoder + FiLM**
   - News text를 non-LLM encoder로 condition vector화한 뒤 FiLM에 넣는 방향

5. **Option E: News + LLM encoder + FiLM**
   - News text를 LLM encoder로 condition vector화한 뒤 FiLM에 넣는 방향

제가 이해한 바로는 **Option B가 교수님께서 말씀하신 방향성에 가장 가깝다**고 생각합니다.
확정해주시면 해당 방향으로 바로 실험을 진행하겠습니다.

## 7. 확정 후 진행 계획

교수님께서 Stage 4 방향을 확정해주시면 다음 일정으로 진행하겠습니다.

1. Day 1-2:
   - 선택된 condition source 확정
   - data alignment 및 leakage check
   - Stage 4 model/code 구현

2. Day 3-4:
   - Kaggle smoke test 및 single-seed 실험
   - baseline / concat / attention or gating / FiLM 비교 실행

3. Day 5-6:
   - metrics, trading metrics, Grad-CAM, gamma/beta export 정리
   - 결과 table과 figure 생성

4. Day 7:
   - 결과 보고서 작성
   - 논문 초안에 들어갈 Stage 4 method/result 문단 작성
   - 다음 미팅에서 보고

목표:
- 방향 확정 후 일주일 이내 실험 완료
- 결과 보고서 작성
- 논문 초안에 반영 가능한 method/result draft 작성

## 8. Brief English Summary

The project has completed the main pipeline organization through Stage 3:
Re-image CNN reproduction, BTC extension, and a Linear adapter comparison.
The next decision is how to define Stage 4 FiLM conditioning.

Based on the discussion, the most defensible Stage 4 direction appears to be
structured market-context conditioning:

```text
BTC chart image CNN + structured market context -> FiLM gamma/beta -> Up/Down
```

The proposed main comparison is:

```text
CNN-only
CNN + context concat
CNN + context attention/gating
CNN + context FiLM gamma-only
CNN + context FiLM full gamma/beta
```

This design directly tests whether FiLM is preferable to simpler fusion methods
and whether gamma/beta provide interpretable evidence of context-dependent chart
feature modulation.

## 9. 교수님께 보낼 메시지 초안

교수님 안녕하세요.

지난 미팅에서 말씀해주신 FiLM 방향성을 기준으로 현재까지 진행한 Stage 1-3 결과와
Stage 4 실험 선택지를 정리했습니다. GitHub에는 현재까지의 코드, 체크리스트, 실행
구조, 결과 요약을 계속 업데이트해두었습니다.

현재까지는 Re-image CNN 재현 구조, BTC 자산군 확장, Linear adapter 비교까지 진행했고,
다음 단계인 Stage 4에서 FiLM을 어떤 condition source와 결합할지 확정이 필요합니다.

제가 이해한 바로는 교수님께서 말씀하신 핵심은 단순히 FiLM layer를 붙이는 것이 아니라,
F&G, Bollinger Band, MFI, volatility 같은 시장 맥락 정보를 condition으로 넣고,
그 condition이 CNN feature를 `gamma/beta`로 어떻게 조절하는지 보는 방향이라고
이해했습니다. 또한 FiLM의 타당성을 확보하기 위해 일반 concat 방식과
attention/gating 방식도 ablation으로 비교해야 한다고 이해했습니다.

그래서 Stage 4 후보를 다음과 같이 정리했습니다.

1. FiLM-only control
2. Structured market context + FiLM
3. Horizon-aware FiLM
4. News + non-LLM encoder + FiLM
5. News + LLM encoder + FiLM

제가 보기에는 2번 Structured market context + FiLM이 교수님께서 말씀하신 방향에
가장 가까운 것으로 보입니다. 이 방향으로 확정해주시면 바로 구현과 Kaggle 실험을
진행해서, 일주일 이내에 실험 결과 보고와 논문 초안에 들어갈 method/result draft까지
작성해 다음 미팅에서 보고드리겠습니다.

어떤 Stage 4 방향이 교수님 의도에 가장 가까운지 확인 부탁드립니다.
