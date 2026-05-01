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
- 데이터:
  - `kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024`
- 구현:
  - BTC OHLCV에서 5/20/60-day 이미지를 직접 생성.
  - CNN 모델 핵심은 1단계에서 확정한 `Stock_CNN`식 구현을 유지.
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

## 4단계: FiLM + News/LLM Conditioning
- 목적:
  - BTC price image CNN feature를 news/LLM condition으로 modulation.
- 모델 구현:
  - `ethanjperez/film`의 핵심 구조를 최대한 동일하게 따른다.
  - conditioning encoder와 FiLM generator를 분리한다.
  - FiLM generator가 block별 gamma/beta를 만든다.
  - FiLM layer는 feature map에 channel-wise affine modulation을 적용한다.
- Re-image CNN에 이식:
  - 기존 CNN 구조는 유지하고, block 내부에 FiLM만 삽입한다.
  - 기본 위치: `Conv -> BN -> FiLM -> LeakyReLU -> MaxPool`
- News/LLM:
  - BTC News: `huggingface.co/datasets/edaschau/bitcoin_news`
  - 1차 encoder: FiLM 원형에 가까운 word embedding 200d + GRU
  - 2차 encoder: Llama 계열 local/open LLM encoder
- 비교:
  - BTC baseline
  - BTC Linear
  - BTC Gamma-only FiLM
  - BTC Full FiLM
  - GRU encoder
  - LLM encoder
- FiLM 단계에서도 BTC Grad-CAM 필수.
  - FiLM이 삽입된 block의 modulation 이후 feature와 기존 CNN block feature를 구분해서 기록한다.
  - Grad-CAM heatmap과 FiLM gamma/beta를 같은 date/sample/layer 기준으로 연결해 해석한다.
- 추가 해석:
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
