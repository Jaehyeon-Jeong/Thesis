# Thesis Plan

## English

## Fixed Research Pipeline

The research design is fixed as four major stages after the initial data/source
audit:

1. **Stage 1: Re-image paper pipeline reproduction**
   - Reproduce the image-based CNN pipeline from *Re-Imag(in)ing Price Trends*.
   - Follow `lich99/Stock_CNN` for the core Stage 1 CNN implementation.
   - Current public-data scope: `I20/R5`, `I20/R20`, `I20/R60`.
2. **Stage 2: BTC asset-class extension**
   - Apply the confirmed Stage 1 image-CNN pipeline to BTC OHLCV.
   - Use time-series classification and trading metrics, not stock H-L deciles.
3. **Stage 3: Linear addition**
   - Add a simple Linear comparison model after the confirmed CNN baseline.
   - Keep the CNN backbone fixed.
4. **Stage 4: FiLM + News/LLM conditioning**
   - Add Gamma-only FiLM and Full FiLM.
   - Add BTC news conditioning, first with GRU-style text encoding and later
     with local/open LLM encoders.
   - Follow `ethanjperez/film` for the core FiLM design where applicable.

## Work Rules

- Every stage starts with a workflow diagram.
- Every stage has a checklist index and a `checklist_results/` folder.
- Checklist items must link to their result reports.
- Major Markdown files use both `## English` and `## 한국어`.
- Before implementation, check the relevant local summaries, PDFs, GitHub
  reference implementation, and data limitations.
- If paper and GitHub differ, document the mismatch before coding.
- Local execution is for smoke tests by default.
- Full training/evaluation runs target Kaggle Notebook.
- One shared codebase is used across local and Kaggle; runtime differences are
  handled by config.
- All code should include explanatory comments/docstrings:
  - explain each important function's input, output, and role in the pipeline;
  - document important tensor/DataFrame shapes;
  - explain where important values move next;
  - explain leakage-sensitive fields and why they are not model inputs.

## Upload Policy

Track:
- code
- configs
- checklists
- checklist result reports
- source maps
- workflow diagrams
- small summary tables
- small sample figures

Do not track:
- paper PDFs
- raw `.dat` image shards
- source `.feather` labels
- checkpoints
- large predictions
- old scratch/test code
- `.DS_Store`

## 한국어

## 고정 연구 파이프라인

자료/근거 확인 이후 연구 설계는 다음 네 단계로 고정합니다.

1. **Stage 1: Re-image 논문 파이프라인 재현**
   - *Re-Imag(in)ing Price Trends*의 이미지 기반 CNN pipeline을 재현합니다.
   - Stage 1 CNN 핵심 구현은 `lich99/Stock_CNN`을 따릅니다.
   - 현재 public-data 범위는 `I20/R5`, `I20/R20`, `I20/R60`입니다.
2. **Stage 2: BTC 자산군 확장**
   - 확인된 Stage 1 image-CNN pipeline을 BTC OHLCV에 적용합니다.
   - BTC는 단일 자산이므로 stock H-L decile이 아니라 time-series
     classification/trading metric을 사용합니다.
3. **Stage 3: Linear 추가**
   - 확정된 CNN baseline 뒤에 단순 Linear 비교 모델을 추가합니다.
   - CNN backbone은 고정합니다.
4. **Stage 4: FiLM + News/LLM conditioning**
   - Gamma-only FiLM과 Full FiLM을 추가합니다.
   - BTC news conditioning을 먼저 GRU-style text encoder로 구현하고,
     이후 local/open LLM encoder로 확장합니다.
   - FiLM 핵심 설계는 가능한 경우 `ethanjperez/film`을 따릅니다.

## 작업 규칙

- 모든 stage는 workflow diagram을 먼저 만듭니다.
- 모든 stage는 checklist index와 `checklist_results/` 폴더를 가집니다.
- checklist 항목은 해당 result report로 링크합니다.
- 주요 Markdown 문서는 `## English`와 `## 한국어`를 함께 둡니다.
- 구현 전에는 관련 로컬 요약, PDF, GitHub reference, 데이터 제한사항을 확인합니다.
- 논문과 GitHub 구현이 다르면 코딩 전에 mismatch를 문서화합니다.
- local 실행은 기본적으로 smoke test입니다.
- full training/evaluation은 Kaggle Notebook을 기준으로 합니다.
- local/Kaggle은 하나의 shared codebase를 쓰고, runtime 차이는 config로 처리합니다.
- 모든 코드에는 자세한 설명 주석/docstring을 남깁니다.
  - 중요한 함수가 무엇을 입력받고 무엇을 반환하는지 설명합니다.
  - 중요한 tensor/DataFrame shape를 적습니다.
  - 중요한 값이 다음 단계의 어느 함수나 파일로 넘어가는지 설명합니다.
  - leakage에 민감한 field는 왜 model input이 아닌지 설명합니다.

## 업로드 정책

추적:
- code
- config
- checklist
- checklist result report
- source map
- workflow diagram
- 작은 summary table
- 작은 sample figure

추적하지 않음:
- 논문 PDF
- raw `.dat` image shard
- source `.feather` label
- checkpoint
- 대용량 prediction
- 이전 scratch/test code
- `.DS_Store`
