# Stage 1: Re-image Reproduction

## English

This folder is reserved for Stage 1 of the thesis pipeline.

Stage 1 objective:
- Reproduce the Re-image paper pipeline before moving to BTC.
- Start with the author-provided public `monthly_20d` rendered images.
- Use the confirmed feasible experiments: `I20/R5`, `I20/R20`, and `I20/R60`.
- Follow the core I20 CNN implementation from `lich99/Stock_CNN` as closely as possible.
- Produce classification outputs and Grad-CAM figures in the style of Re-image Figure 13.

Execution environment:
- Full Stage 1 training/evaluation should be designed for Kaggle Notebook.
- Local execution is for small smoke tests and structure checks.
- Do not create separate Kaggle-only and local-only codebases.
- Keep one shared `src/` codebase and use environment-specific configs/runners.
- Colab may be added later as an optional runner, especially for Hugging Face or LLM-heavy stages.

Current status:
- Folder and planning documents are created.
- Shared code/config scaffold is implemented.
- Data loading for public `monthly_20d` image/label shards is implemented.
- Label, split, and train-only normalization metadata code is implemented.
- No model, training, evaluation, or Grad-CAM code has been implemented yet.

Required pre-work before every Stage 1 task:
- Read the root `../PLAN.md`.
- Check `../stage0_data_check/docs/monthly20_data_check.md`.
- Check `../stage0_data_check/docs/source_reference_check.md`.
- Check `docs/stage1_checklist.md`.
- Confirm whether the next action is still within the feasible Stage 1 boundary.

Primary limitation:
- Current local stock data supports only public 20-day full-spec images:
  `OHLC + 20-day MA + volume`.
- It does not directly support `I5`, `I60`, or A/B/C/D image-spec ablations.

## 한국어

이 폴더는 논문 파이프라인의 1단계 작업 공간입니다.

1단계 목표:
- BTC로 넘어가기 전에 Re-image 논문 파이프라인을 먼저 재현합니다.
- 저자가 공개한 `monthly_20d` 렌더링 이미지부터 시작합니다.
- 현재 가능한 실험은 `I20/R5`, `I20/R20`, `I20/R60`입니다.
- I20 CNN 핵심 구현은 `lich99/Stock_CNN`을 최대한 그대로 따릅니다.
- classification 결과와 Re-image Figure 13 스타일 Grad-CAM 그림을 생성합니다.

실행 환경:
- 1단계 full training/evaluation은 Kaggle Notebook 기준으로 설계합니다.
- 로컬 실행은 작은 smoke test와 구조 확인 용도입니다.
- Kaggle 전용 코드와 local 전용 코드를 따로 만들지 않습니다.
- 하나의 공통 `src/` 코드베이스를 두고, 환경별 config/runner로 실행합니다.
- Colab은 나중에 Hugging Face나 LLM-heavy 단계에서 optional runner로 추가할 수 있습니다.

현재 상태:
- 폴더와 계획 문서를 만들었습니다.
- 공통 code/config scaffold를 구현했습니다.
- public `monthly_20d` image/label shard data loading을 구현했습니다.
- label, split, train-only normalization metadata 코드를 구현했습니다.
- model, training, evaluation, Grad-CAM 코드는 아직 구현하지 않았습니다.

1단계의 모든 작업 전에 반드시 확인할 것:
- 루트 `../PLAN.md`
- `../stage0_data_check/docs/monthly20_data_check.md`
- `../stage0_data_check/docs/source_reference_check.md`
- `docs/stage1_checklist.md`
- 다음 작업이 현재 1단계 가능 범위 안에 있는지 확인합니다.

핵심 제한사항:
- 현재 로컬 stock 데이터는 public 20-day full-spec image만 지원합니다:
  `OHLC + 20-day MA + volume`.
- `I5`, `I60`, A/B/C/D image-spec ablation은 현재 데이터만으로 직접 수행할 수 없습니다.
