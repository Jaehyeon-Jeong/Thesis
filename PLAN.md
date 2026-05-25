# 확정 파이프라인 계획: 모델 핵심 구현은 GitHub를 따르고, 연구 설계는 유지

## Summary
- 전체 연구 설계는 바꾸지 않는다.
- 단계 순서는 고정한다:
  1. Re-image 논문 파이프라인 재현
  2. BTC 자산군 확장
  3. Linear 추가
  4. FiLM + News/LLM conditioning
- GitHub를 따른다는 뜻은 “프로젝트 목적을 바꾼다”가 아니라, **모델의 핵심 구현 디테일을 해당 repo와 최대한 동일하게 맞춘다**는 뜻이다.
- 따라서 1단계 CNN 모델은 `lich99/Stock_CNN`의 중요한 구현을 최대한 그대로 따른다.
- 4단계 FiLM 모델은 `ethanjperez/film`의 중요한 구현을 최대한 그대로 따른다.
- 단, 연구 목적, 실험 단계, BTC 확장, Linear 비교, News/LLM-FiLM 비교는 사용자가 정한 논문 계획 그대로 유지한다.

## 모든 작업 시작 전 고정 확인 규칙
- 새 작업을 시작하기 전에 반드시 이 `PLAN.md`를 먼저 확인한다.
- 현재 작업이 어느 단계에 속하는지 먼저 명시한다.
- 해당 단계에 연결된 로컬 논문 요약, PDF, GitHub reference, 데이터 제한사항을 확인한 뒤에만 세부계획이나 구현으로 넘어간다.
- 논문 PDF/요약과 GitHub 구현이 다르면 임의로 고치지 않는다.
  - 차이를 문서화한다.
  - 어느 쪽을 따를지 사용자에게 먼저 보여준다.
- 모델 구조, label, loss, optimizer, threshold, evaluation 같은 핵심 결정은
  논문에 있는 정보인지 구현상 선택인지 반드시 구분해서 적는다.
  - 논문에 있으면 paper page/section/figure/table 위치를 함께 남긴다.
  - 현재는 로컬 요약의 page mapping을 먼저 쓰되, 코드 주석을 달기 전 PDF viewer에서 다시 확인한다.
  - 논문에 없으면 `Paper: not reported` 또는 `Implementation choice`라고 명시한다.
- 현재 로컬 데이터로 가능한 범위와 불가능한 범위를 항상 먼저 확인한다.
- Grad-CAM, FiLM, BTC 확장처럼 별도 방법론 논문이 붙는 작업은 해당 방법론 논문 요약/PDF를 다시 확인한 뒤 구현한다.
- 한 번에 전체 파이프라인을 구현하지 않는다. `세부계획 -> 확인 -> 구현 -> 보고` 순서로 한 항목씩 진행한다.

## 실행 환경 원칙
- 기본 full experiment 실행 환경은 로컬이 아니라 **Kaggle Notebook**으로 둔다.
- 이유:
  - CNN 학습과 큰 image shard 처리는 로컬보다 Kaggle GPU/Notebook 환경이 현실적이다.
  - Kaggle dataset은 Kaggle Notebook에서 attach하기 쉽다.
  - Kaggle notebook version과 dataset version을 남기면 실행 기록을 관리하기 좋다.
- 단, 코드를 Kaggle 전용으로 두 벌 만들지 않는다.
  - GitHub/repo의 `src/`가 단일 핵심 코드베이스다.
  - Kaggle Notebook은 그 코드를 실행하는 runner/wrapper다.
  - Stage 1 Kaggle 실행 표준은 `notebooks/kaggle_stage1_single_horizon_one_cell.md`
    방식이다.
    - Kaggle에는 한 셀을 복붙해 실행한다.
    - 그 셀은 code snapshot과 data를 working/local disk로 복사한다.
    - 실제 학습/평가/Grad-CAM 구현은 기존 `src/`와 `scripts/`를 호출한다.
    - horizon은 한 번에 하나씩 실행한다.
    - 1-I10 Kaggle full single-seed debugging에서는 fast option을 사용할 수 있다:
      train/validation RAM pre-load, larger batch, mixed precision, optional
      DataParallel. 이 option은 실행 속도를 위한 것이며 model/label/split 구조를
      바꾸지 않는다.
  - Colab Notebook은 필요할 경우 optional runner다.
  - 로컬은 작은 subset smoke test와 문서/구조 확인 용도다.
- 환경 차이는 코드 분기가 아니라 config로 관리한다.
  - 예: `env_local.yaml`, `env_kaggle.yaml`, 필요하면 `env_colab.yaml`
- 재현성 기록에는 반드시 다음을 남긴다.
  - GitHub commit hash
  - Kaggle notebook version 또는 run id
  - Kaggle dataset name/version
  - Hugging Face dataset name/revision
  - config 파일
  - random seed
  - package/environment 정보
  - `metrics.json`, `predictions.csv`, 주요 figure outputs
- Kaggle에서 긴 모델 학습을 실행하는 runner는 반드시 output backup을 자동 생성한다.
  - 학습 직후 checkpoint가 생기면 즉시 backup한다.
  - evaluation 직후 prediction/metric이 생기면 다시 backup한다.
  - Grad-CAM/figure 생성 직후 다시 backup한다.
  - backup zip은 `PROJECT_ROOT` 밖, 예를 들어 `/kaggle/working/stage1_saved_outputs/`
    또는 `/kaggle/working/stage2_saved_outputs/`,
    `/kaggle/working/stage4_saved_outputs/`에 저장한다.
  - 이유: 다음 horizon/model 실행 때 `PROJECT_ROOT`를 새 code snapshot으로 다시 만들면
    이전 run의 checkpoint, prediction, metric, Grad-CAM이 삭제될 수 있기 때문이다.
  - 이 backup은 실험 로직 변경이 아니라 output 보존을 위한 실행 안정성 장치다.
- 4단계 News/LLM처럼 Hugging Face/LLM cache가 중요한 작업은 Colab도 후보가 될 수 있지만, 기본 원칙은 여전히 단일 코드베이스 + 환경별 runner다.

## 0단계: 자료/데이터/참고 구현 확인
- Re-image 근거:
  - `자료조사/Re-image 요약.md`
  - `자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
  - GitHub: `https://github.com/lich99/Stock_CNN`
- Grad-CAM 근거:
  - `자료조사/Grad-CAM요약.md`
  - `자료조사/Grad-CAM.pdf`
  - Re-image 논문 Figure 13을 재현할 때도 이 Grad-CAM 원전 자료를 함께 확인한다.
- FiLM 근거:
  - `자료조사/FiLM요약.md`
  - `자료조사/FiLM.pdf`
  - GitHub: `https://github.com/ethanjperez/film`
- 먼저 `docs/source_map.md`를 만든다.
  - 논문 page/section/figure/table
  - GitHub repo/file/commit hash
  - 코드에서 해당 근거가 쓰인 위치
- `monthly_20d` 데이터 확인:
  - 파일 수, sample 수, label columns
  - `has_vb`, `20_ma`가 실제로 volume/MA 포함인지 sample image로 확인
  - 공개 데이터로 가능한 재현 범위 명시

## 1단계: Re-image 논문 파이프라인 재현
- 목적:
  - 원 논문 pipeline을 정확히 구현하고 재현성 확인.
- 실행 환경:
  - full training/evaluation은 Kaggle Notebook 기본으로 설계한다.
  - 로컬에서는 subset smoke test만 통과하도록 설계한다.
  - Kaggle 전용 코드와 로컬 전용 코드를 따로 만들지 않고, 공통 `src/`와 환경별 config/runner를 사용한다.
  - Stage 1 Kaggle에서는 one-cell single-horizon runner를 표준으로 사용한다.
    - `notebooks/kaggle_stage1_single_horizon_one_cell.md`
    - `I20/R20`, `I20/R5`, `I20/R60`을 horizon 하나씩 실행한다.
    - 이전 all-in-one shell wrapper 방식은 표준 실행 경로로 사용하지 않는다.
- 모델 구현:
  - `lich99/Stock_CNN/models/baseline.py`의 핵심 모델 구현을 최대한 동일하게 따른다.
  - layer 구성, Conv/BN/LeakyReLU/MaxPool 순서, classifier 흐름, forward 구조를 repo 기준으로 맞춘다.
  - 다만 논문 PDF와 불일치하는 부분이 발견되면, 수정 전 반드시 불일치 내용을 기록하고 사용자에게 보여준다.
- 구현 항목:
  - OHLC image generator
  - MA line
  - volume bars
  - label construction
  - split
  - CNN baseline
  - training/evaluation
  - result table
  - Grad-CAM
- 고정 연구 디테일:
  - 5-day `32 x 15`
  - 20-day `64 x 60`
  - 60-day `96 x 180`
  - 하루 3px
  - filter `5 x 3`
  - pool `2 x 1`
  - channels `64 -> 128 -> 256 -> 512`
- 공개 `monthly_20d`로 먼저 `I20/R5`, `I20/R20`, `I20/R60` 재현을 확인한다.
- 5/60 stock image shard가 없으면 generator는 구현하되, 실증 재현 제한을 보고서에 명시한다.

## 1단계 Grad-CAM
- 원논문 Figure 13 재현은 필수다.
- Grad-CAM 구현 전 반드시 확인할 근거:
  - `자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
  - `자료조사/Re-image 요약.md`
  - `자료조사/Grad-CAM.pdf`
  - `자료조사/Grad-CAM요약.md`
- 2019년 sample 중:
  - `Up` 예측 10개
  - `Down` 예측 10개
- 원본 이미지 + 각 CNN block/layer Grad-CAM heatmap grid 저장.
- 이 그림은 feature map 자체가 아니라 feature map과 gradient로 만든 class-discriminative heatmap이라고 명시한다.
- 구현 원칙:
  - target class score는 softmax 이후 probability가 아니라 softmax 이전 logit을 기본으로 사용한다.
  - 선택한 convolution layer의 activation `A^k`와 target score에 대한 gradient를 저장한다.
  - channel weight는 gradient의 spatial average로 계산한다.
  - heatmap은 `ReLU(sum_k alpha_k^c A^k)`로 만든다.
  - heatmap은 입력 이미지 크기로 bilinear upsampling한다.
  - 논문 Figure 13처럼 원본 이미지 row와 layer/block별 Grad-CAM row를 함께 저장한다.
  - 밝은 영역은 해당 target class 결정에 더 크게 기여한 영역으로 해석한다.
  - raw feature map, saliency map, Guided Backprop, Guided Grad-CAM을 혼동하지 않는다.
- 제한사항:
  - Grad-CAM은 모델이 실제로 학습된 뒤의 post-hoc explanation이다.
  - 학습 전 sample image만으로는 Figure 13과 같은 class-discriminative heatmap을 만들 수 없다.
  - BTC/Linear/FiLM 단계에서도 같은 원칙으로 Grad-CAM을 생성하되, target class와 적용 layer를 명시한다.

## 2단계: BTC 자산군 확장
- 목적:
  - 1단계에서 재현한 Re-image CNN pipeline을 BTC 단일 자산에 적용.
- 진행 경계:
  - Stage 1 Kaggle full run이 도는 동안 Stage 2의 데이터 audit, checklist,
    scaffold, image-generation plan은 진행할 수 있다.
  - Stage 1과의 최종 비교표와 결론은 Stage 1 full output이 나온 뒤 확정한다.
- 데이터:
  - `kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024`
- 구현:
  - BTC OHLCV에서 5/20/60-day 이미지를 직접 생성.
  - CNN 모델 핵심은 1단계에서 확정한 `Stock_CNN`식 구현을 유지.
  - BTC dataset은 public stock shard보다 작으므로 기본 batch size는 논문값
    `128`을 유지한다. Stage 1 fast diagnostic의 `1024` batch를 Stage 2 기본값으로
    가져오지 않는다.
- 평가:
  - classification metrics
  - long/flat
  - long/short
  - Sharpe, MDD, turnover, transaction-cost-adjusted return
- BTC에서도 Grad-CAM 필수.
  - BTC baseline이 어떤 chart region을 강조하는지 확인한다.
  - Grad-CAM 생성 전 `자료조사/Grad-CAM요약.md`와 `자료조사/Grad-CAM.pdf`를 다시 확인한다.

## 3단계: Linear 추가
- 목적:
  - FiLM 전 단계로 단순한 Linear 변환을 붙인 비교 모델 생성.
- 구현:
  - 1/2단계에서 확정된 CNN 구조를 바꾸지 않는다.
  - CNN feature 뒤에 `torch.nn.Linear(..., bias=False)` adapter를 추가한다.
- 해석:
  - `bias=False`는 additive shift를 제거한다.
  - FiLM beta 제거와 유사한 비교축으로 설명하되, Dense Linear와 channel-wise FiLM은 동일하지 않다고 명시한다.
- 비교:
  - BTC CNN baseline
  - BTC CNN + Linear(bias=False)
- Linear 단계에서도 BTC Grad-CAM 필수.
  - Linear adapter가 붙어도 Grad-CAM은 CNN feature와 target class score의 gradient 관계로 계산한다.
  - baseline CNN과 Linear model의 heatmap 차이를 같은 sample/date 기준으로 비교한다.

## 4단계: Market Context Fusion + FiLM Conditioning
- 목적:
  - BTC price image CNN feature를 structured market context로 modulation.
  - Stage 2에서 확인한 강한 chart-image baseline 위에서, 시장 맥락이 같은 차트
    패턴의 해석을 바꿀 수 있는지 확인한다.
- 현재 진행 원칙:
  - Stage 4의 우선 실험은 아래 네 가지 context fusion 방식 비교로 고정한다.
    1. `4-A CNN + context concat`
    2. `4-B CNN + context gating`
    3. `4-C CNN + context FiLM gamma-only`
    4. `4-D CNN + context FiLM full gamma/beta`
  - Image input은 Stage 2 selected five-seed best인
    `I60/R20/ohlc_ma_vb`를 primary baseline으로 고정한다.
  - Market context는 이미지 위에 추가로 그리지 않는다. 별도 numeric vector로 구성한다.
    - 후보: F&G score, Bollinger %B, Bollinger bandwidth, MFI, realized volatility.
    - 모든 context feature는 image end date `t`까지의 정보만 사용한다.
    - context feature scaling/normalization은 train split에서만 fit한다.
    - 4-5에서 첫 model input은 아래 8개 matched-window feature로 고정했다:
      `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`,
      `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60`.
    - Preprocessing은 feature-specific transform 이후 train-only median imputation,
      train-only 1/99% clipping, train-only z-score normalization을 사용한다.
    - 4-I2 local build 결과, primary `I60/R20/ohlc_ma_vb` context table은
      2,399 rows(train 671, validation 287, test 1,441)이며 8개 primary feature의
      missing-rate warning은 없었다.
    - Primary `I60/R20/ohlc_ma_vb`의 첫 valid sample end date는 `2018-04-29`다.
      BTC OHLCV는 `2018-01-01`부터 시작하지만, MA60 첫 유효일과 I60 window가 모두
      inclusive라 정확한 offset은 `(60 - 1) + (60 - 1) = 118`일이다.
      F&G는 `2018-02-01`부터 시작하므로 valid primary sample을 제거하지 않으며,
      primary sample end date와 직접 겹치는 F&G 원본 missing date는 `2024-10-26`
      하루뿐이다.
  - News context는 버리지 않는다. 다만 바로 LLM/VLM prompt로 처리하지 않고,
    별도 `4-N` track에서 dataset audit, publication-time alignment, daily aggregation,
    leakage check를 먼저 수행한다.
  - 뉴스 dataset 후보는 `huggingface.co/datasets/edaschau/bitcoin_news`다.
  - 뉴스는 원문 그대로 모델에 "질문"으로 넣는 것이 아니라, 일별 headline/article를
    요약하거나 non-LLM/LLM encoder로 embedding해서 context vector로 사용한다.
  - 현재 Stage 4 main은 structured numeric market context를 사용한
    FiLM/concat/gating 비교이고, news context는 같은 fusion 방식에 붙일 2차 확장이다.
- 구현 dependency:
  - 4-I0 기준: Stage 4는 Stage 2 BTC pipeline을 재작성하지 않는다.
  - Stage 4 config는 `stage2_dependency` section을 갖고, Stage 2 project root와
    Stage 2 `src` path를 명시한다.
  - Stage 4 script는 Stage 4 `src`와 Stage 2 `src`를 모두 import path에 추가한다.
  - Kaggle runner는 Stage 4 code snapshot만이 아니라 Stage 2 code snapshot도
    attach/copy하거나, Stage 2와 Stage 4가 같이 들어 있는 larger snapshot을 사용한다.
  - 4-I0 data update: local F&G data가
    `stage4_film_conditioning/F&G_data/fear_greed_index.csv`에 추가됐다.
    Raw F&G CSV는 GitHub에 올리지 않고, Kaggle run에서는 public F&G dataset을
    attach해서 재현성을 유지한다.
  - 4-I1 기준: Stage 4 local/Kaggle config, path/runtime/seed helper,
    Stage 2 dependency import utility, scaffold checker가 추가됐고 local check가
    통과했다. 다음 구현은 `4-I2` context source audit/feature builder다.
- 모델 구현:
  - `ethanjperez/film`의 핵심 구조를 최대한 동일하게 따른다.
  - context encoder와 fusion/modulation head를 분리한다.
  - context encoder는 numeric market context vector를 MLP로 condition embedding으로 바꾼다.
  - 4-5에서 shared context encoder를
    `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`로 고정했다.
  - 4-I3에서 이 shared context encoder를 구현했고, dummy tensor와 local 4-I2
    normalized context row 모두에서 `(B, 8) -> (B, 32)` shape check를 통과했다.
  - concat model은 CNN feature와 context embedding을 classifier 직전에 연결한다.
    - 4-6 기준: I60 flatten feature `(B, 184320)` 뒤에 context embedding `(B, 32)`를
      붙여 `(B, 184352)` classifier input을 만든다.
    - 4-I4에서 `CNN + context concat` model을 구현했다. Stage 2 I60 Stock_CNN
      convolution blocks는 그대로 재사용하고, final classifier만
      `Dropout(0.5) -> Linear(184352, 2)`로 교체했다.
    - 4-I4 local shape check 결과:
      image `(B, 1, 96, 180) -> flatten (B, 184320)`,
      context `(B, 8) -> embedding (B, 32)`,
      concat `(B, 184352) -> logits (B, 2)`.
    - 4-I4 parameter count는 `2,954,370`이며 Stage 2 I60 baseline 대비
      `+1,408`이다.
  - gating model은 context embedding으로 channel gate를 만들고 CNN feature를 곱해서 조절한다.
    - 4-6 기준: 첫 run에서는 final block feature map `(B, 512, 2, 180)`에만
      channel-wise gate를 적용하고 `gate = 2 * sigmoid(raw_gate)`를 사용한다.
    - 4-I5에서 `CNN + context gating` model을 구현했다. Context embedding
      `(B, 32)`이 raw gate `(B, 512)`를 만들고, `gate = 2 * sigmoid(raw_gate)`를
      마지막 I60 feature map `(B, 512, 2, 180)`에 곱한다.
    - 4-I5 gate head는 zero-initialized라 초기 gate는 정확히 `1.0`이고,
      Stage 2 feature path와 같은 identity modulation에서 시작한다.
    - 4-I5 parameter count는 `2,971,202`이며 Stage 2 I60 baseline 대비
      `+18,240`이다.
  - gamma-only FiLM은 block별 gamma만 만들어 `F' = gamma * F`를 적용한다.
  - full FiLM은 block별 gamma/beta를 만들어 `F' = gamma * F + beta`를 적용한다.
  - FiLM layer는 feature map에 channel-wise modulation을 적용한다.
    - 4-I6에서 `FeatureWiseAffineModulation` layer를 구현했다.
    - 4-I6에서 context embedding `(B, 32)`을 받아 I60 block channel
      `[64, 128, 256, 512]`별 gamma/beta를 만드는 `FilmParameterGenerator`를
      구현했다.
    - Gamma는 `1 + delta_gamma`로 만들고, gamma/beta head는 zero-initialize한다.
      따라서 초기값은 gamma `1.0`, beta `0.0`이고 Stage 2 feature path를
      identity로 보존한다.
	    - 4-I6 local check 결과 gamma-only generator parameter count는 `31,680`,
	      full FiLM generator parameter count는 `63,360`이었다.
	    - Dummy I60 feature map과 local 4-I2 normalized context row 모두에서
	      FiLM identity initialization check를 통과했다.
    - 4-I7에서 `FilmContextStockCNN`을 구현했다.
    - Stage 2 I60 Stock_CNN convolution block을 재사용하되, 각 block을
      `Conv -> BN -> FiLM -> LeakyReLU -> MaxPool` 순서로 실행한다.
    - `film_gamma`는 parameter count `2,985,986`, Stage 2 I60 baseline 대비
      `+33,024`이며 all-block identity initialization check를 통과했다.
    - `film_full`은 parameter count `3,017,666`, Stage 2 I60 baseline 대비
      `+64,704`이며 all-block identity initialization check를 통과했다.
    - Stage 2 block의 `LeakyReLU(inplace=True)` 때문에 해석 export용
      post-FiLM feature map은 activation 전에 복사해서 보관한다.
  - 4-I8에서 Stage 4 context runner를 구현했다.
    - `scripts/run_stage4_context_model.py`는 단일 Stage 4 context-conditioned
      training job을 실행한다.
    - `src/stage4_film/runners/context_experiment.py`는 Stage 2 BTC data loading,
      sample generation, chart image generation, split, train-only pixel normalization을
      재사용하고 normalized context tensor를 각 batch에 붙인다.
    - `src/stage4_film/training/loop.py`는 `model(image, context)` 형태로 학습한다.
    - Stage 2 일반 weight initialization 이후 gate/FiLM output head를 identity로
      다시 reset한다. 따라서 gating은 `gate=1`, FiLM은 `gamma=1`, `beta=0`에서
      시작한다.
    - Local smoke training은 `concat`과 `film_gamma`에서 통과했다.
  - 4-I9에서 Stage 4 prediction/classification/trading export를 구현했다.
    - `scripts/evaluate_stage4_predictions.py`는 Stage 4 checkpoint를 다시 로드하고
      `model(image, context)`로 prediction CSV와 classification metrics를 저장한다.
    - `scripts/evaluate_stage4_trading.py`는 prediction CSV를 읽고 BTC long/flat,
      long/short trading metrics를 저장한다.
    - Metric 공식은 Stage 2와 동일하게 재사용한다.
    - Prediction CSV에는 `context_method`, `stage4_experiment_name`, normalized
      context columns도 함께 남긴다.
    - Local smoke checkpoint 기준 `concat`, `film_gamma` export가 통과했다.
  - 4-I10에서 Stage 4 Grad-CAM plus context/modulation export를 구현했다.
    - `scripts/generate_stage4_gradcam_context.py`는 checkpoint와 prediction CSV를
      다시 로드하고 `model(image, context)` 기준 predicted-class Grad-CAM을 만든다.
    - `src/stage4_film/interpretability/gradcam_context.py`는 Stage 2 sample
      selection을 재사용하되, 선택 sample마다 context vector와 modulation 값을
      같이 export한다.
    - 출력은 figure, `samples.csv`, `modulation_summary.csv`,
      `modulation_values.json`이다.
    - `concat`은 context/context embedding을, `gating`은 raw gate/final gate를,
      `film_gamma`와 `film_full`은 block-wise gamma/beta를 저장한다.
    - FiLM Grad-CAM hook은 post-FiLM module에 걸며, PyTorch inplace activation
      충돌을 피하기 위해 tensor-level gradient hook을 사용한다.
    - Local smoke checkpoint 기준 `concat`, `film_gamma` Grad-CAM export가 통과했다.
  - 4-I11에서 Stage 4 smoke output checker를 구현했다.
    - `scripts/check_stage4_outputs.py`는 checkpoint, train history/metadata,
      predictions, classification metrics, trading metrics, Grad-CAM, selected
      samples, modulation summary/value export, context artifacts, run manifest를
      한 번에 확인한다.
    - Checker는 파일 존재뿐 아니라 CSV parse/row count와 JSON parse도 확인한다.
    - Local smoke output check는 `concat`, `film_gamma` 모두 통과했다.
    - Compact summary는 `reports/smoke_tests/stage4_smoke_summary.json`에 저장했다.
  - 4-I12에서 Kaggle four-ablation single-seed runner를 준비했다.
    - `notebooks/kaggle_stage4_four_ablation_single_seed_one_cell.md`는
      `I60/R20/ohlc_ma_vb`, context window `60`, seed `42`를 고정하고
      `concat`, `gating`, `film_gamma`, `film_full`을 순서대로 실행한다.
    - Cell은 Stage 4/Stage 2 code snapshot copy, Kaggle config patch,
      BTC/F&G source audit, context feature build, method별 train/evaluation/
      trading/Grad-CAM/output-check, backup zip, summary table까지 수행한다.
    - 아직 실제 Kaggle full run 결과는 아니므로 output check와 metric이
      보고될 때까지 4-I12는 open 상태로 유지한다.
- Re-image CNN에 이식:
  - 기존 CNN 구조는 유지하고, block 내부에 FiLM만 삽입한다.
  - 기본 위치: `Conv -> BN -> FiLM -> LeakyReLU -> MaxPool`
  - 4-6 기준: gate/FiLM output head는 zero-initialize해서 gate와 gamma는 `1`,
    beta는 `0`에서 시작하게 한다.
- 비교:
  - BTC baseline
  - BTC Linear
  - BTC CNN + context concat
  - BTC CNN + context gating
  - BTC CNN + context FiLM gamma-only
  - BTC CNN + context FiLM full gamma/beta
  - later: BTC CNN + news-context concat/gating/FiLM, news audit 통과 후 실행
- FiLM 단계에서도 BTC Grad-CAM 필수.
  - FiLM이 삽입된 block의 modulation 이후 feature와 기존 CNN block feature를 구분해서 기록한다.
  - Grad-CAM heatmap과 gate/gamma/beta를 같은 date/sample/layer 기준으로 연결해 해석한다.
  - 4-7 기준: primary Grad-CAM target은 predicted-class pre-softmax logit이다.
  - 최종 figure는 test split에서 Predicted Up 10개, Predicted Down 10개를 사용한다.
  - 4-C/4-D는 post-FiLM conditioned feature map을 primary Grad-CAM target layer로 사용한다.
- Kaggle 실행:
  - 4-8 기준: 첫 Stage 4 Kaggle runner는 structured numeric context 네 ablation
    `concat`, `gating`, `film_gamma`, `film_full`만 실행한다.
  - 4-I12 기준: 실제 실행 cell은
    `stage4_film_conditioning/notebooks/kaggle_stage4_four_ablation_single_seed_one_cell.md`
    에 준비되어 있다.
  - 첫 full sanity run은 seed `42`, 이후 robustness run은 seed
    `42, 43, 44, 45, 46`으로 실행한다.
  - Backup root는 `/kaggle/working/stage4_saved_outputs/`로 고정한다.
  - Runner는 context build, training, prediction evaluation, trading evaluation,
    Grad-CAM/export, output check, summary 뒤에 backup zip과 receipt JSON을 만든다.
  - 완료 판정은 output checker 통과 기준이다. Checkpoint만 있고 predictions,
    metrics, trading metrics, Grad-CAM, context/modulation export, manifests가 없으면
    incomplete run으로 본다.
- 추가 해석:
  - context feature 저장
  - concat/gate/gamma/beta 저장
  - gamma/beta 저장
  - layer/channel/date별 분석
  - regime/confidence/correctness별 분석

## 코드 작성 원칙
- 모델 핵심 구현은 GitHub를 최대한 동일하게 따른다.
- 연구 설계, 단계 순서, 비교 구조는 임의로 바꾸지 않는다.
- 논문 PDF와 GitHub 구현이 다를 경우:
  - 차이를 문서화한다.
  - 어떤 쪽을 따를지 사용자에게 확인한다.
- 코드는 읽기 쉽게 작성한다.
  - 함수명 직관적으로
  - tensor shape docstring
  - source 주석
  - leakage 방지 주석
  - 모든 코드에는 자세한 설명 주석을 남긴다.
    - 설명 주석과 docstring은 기본적으로 한국어로 작성한다.
    - 각 함수가 무엇을 입력받고 무엇을 반환하는지 적는다.
    - 중요한 중간 객체의 의미와 tensor/dataframe shape를 적는다.
    - 값이 다음 단계의 어느 함수나 파일로 넘어가는지 적는다.
    - 단순 문법 설명보다 실험 파이프라인에서의 역할을 설명한다.
- Grad-CAM 코드는 원전 수식과 Re-image Figure 13 재현 목적을 주석에 남긴다.
- 예시 주석:
```python
# Reference implementation:
#   lich99/Stock_CNN/models/baseline.py, commit <hash>
#
# Paper source:
#   Xiu et al., Re-Imagining Price Trends, Figure 7.
#
# This block intentionally follows the reference CNN implementation.
# Input:
#   x: (batch_size, 1, 64, 60)
# Output:
#   logits: (batch_size, 2)
```

Grad-CAM 예시 주석:
```python
# Grad-CAM source:
#   Selvaraju et al., Grad-CAM, method section, Eq. (1)-(2).
#   Local files: 자료조사/Grad-CAM요약.md, 자료조사/Grad-CAM.pdf.
#
# Re-image target figure:
#   Jiang, Kelly, and Xiu, Re-Imagining Price Trends, Figure 13.
#
# This is not a raw feature map. It is a class-discriminative heatmap
# computed from activations and gradients for a target class logit.
```

## Required Outputs
- `docs/source_map.md`
- `docs/stage_checklist.md`
- `reports/data_audit.md`
- `reports/tables/*.csv`
- `reports/figures/sample_images/`
- `reports/figures/gradcam/`
- `reports/figures/film_gamma_beta/`
- `outputs/*/predictions.csv`
- `outputs/*/metrics.json`
- `outputs/*/film_params.parquet`

## Assumptions
- GitHub를 따른다는 것은 모델 구현 핵심을 맞춘다는 뜻이지, 연구 설계를 바꾸는 것이 아니다.
- 논문 전체 계획은 사용자가 정한 1-4단계 구조 그대로 유지한다.
- Grad-CAM은 stock/BTC/baseline/Linear/FiLM 전 단계에서 필수 해석 산출물이다.
