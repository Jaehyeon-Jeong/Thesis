# Stage 3: Linear Adapter Comparison

## English

This folder is reserved for Stage 3 of the thesis pipeline.

Stage 3 objective:
- Keep the Stage 2 BTC image-generation, label, split, normalization,
  evaluation, trading metric, and Grad-CAM pipeline fixed.
- Add a simple Linear adapter after the CNN feature extractor.
- Compare the BTC CNN baseline against `BTC CNN + Linear(bias=False)`.
- Use this as an intermediate comparison before Stage 4 FiLM conditioning.

Key modeling rule:
- Do not change the Stage 1/2 CNN core.
- Insert the Linear adapter after CNN feature extraction and before the final
  classification logits.
- Use `torch.nn.Linear(..., bias=False)` for the first Stage 3 comparison.
- `bias=False` removes additive shift from the adapter, but this is not the same
  as channel-wise Gamma-only FiLM.

Current boundary:
- Stage 3 starts with planning and scaffold only.
- Stage 2 five-seed reruns are still pending, so Stage 3 final claims should be
  treated as provisional until Stage 2 stability is checked.
- Full training/evaluation remains Kaggle-first, with local smoke tests only for
  code and shape checks.

Main documents:
- [Checklist](checklist.md)
- [Workflow diagram](workflow_diagram.md)
- [Stage 3 pipeline](docs/stage3_pipeline.md)
- [Source map](docs/source_map.md)
- [Stage 2 dependency review](docs/stage2_dependency_baseline_review.md)
- [Linear adapter design](docs/linear_adapter_design.md)
- [Training/evaluation comparison plan](docs/training_evaluation_comparison_plan.md)
- [Grad-CAM comparison plan](docs/gradcam_comparison_plan.md)
- [Kaggle runner and output plan](docs/kaggle_runner_output_plan.md)

Current planning status:
- Planning steps `3-1` through `3-5` are complete.
- First implementation gate is `3-I0` implementation readiness review.
- First planned Linear run is `I60/R20/ohlc_ma_vb`, seed `42`, then the
  single-seed `36`-run grid.
- Default Linear adapter dimension is `128`.
- Five-seed stability checks are deferred.

## 한국어

이 폴더는 논문 파이프라인의 3단계 작업 공간입니다.

3단계 목표:
- Stage 2의 BTC image generation, label, split, normalization, evaluation,
  trading metric, Grad-CAM 파이프라인은 고정합니다.
- CNN feature extractor 뒤에 단순 Linear adapter를 추가합니다.
- BTC CNN baseline과 `BTC CNN + Linear(bias=False)`를 비교합니다.
- 이 단계는 Stage 4 FiLM conditioning 전에 두는 중간 비교축입니다.

핵심 모델링 규칙:
- Stage 1/2에서 확정한 CNN core는 바꾸지 않습니다.
- Linear adapter는 CNN feature extraction 뒤, final classification logits 전에
  삽입합니다.
- 첫 Stage 3 비교에서는 `torch.nn.Linear(..., bias=False)`를 사용합니다.
- `bias=False`는 additive shift를 제거하지만, 이것이 channel-wise Gamma-only
  FiLM과 같다는 뜻은 아닙니다.

현재 경계:
- Stage 3는 계획과 scaffold부터 시작합니다.
- Stage 2의 5-seed rerun은 아직 예정이므로, Stage 3 최종 결론도 Stage 2 안정성
  확인 전까지는 provisional result로 봅니다.
- full training/evaluation은 Kaggle 우선이며, local은 code/shape smoke test 용도입니다.

주요 문서:
- [Checklist](checklist.md)
- [Workflow diagram](workflow_diagram.md)
- [Stage 3 pipeline](docs/stage3_pipeline.md)
- [Source map](docs/source_map.md)
- [Stage 2 dependency review](docs/stage2_dependency_baseline_review.md)
- [Linear adapter design](docs/linear_adapter_design.md)
- [Training/evaluation comparison plan](docs/training_evaluation_comparison_plan.md)
- [Grad-CAM comparison plan](docs/gradcam_comparison_plan.md)
- [Kaggle runner and output plan](docs/kaggle_runner_output_plan.md)

현재 계획 상태:
- Planning step `3-1`부터 `3-5`까지 완료했습니다.
- 다음 구현 gate는 `3-I0` implementation readiness review입니다.
- 첫 Linear 실행은 `I60/R20/ohlc_ma_vb`, seed `42`이고, 이후 single-seed
  `36`-run grid로 갑니다.
- 기본 Linear adapter dimension은 `128`입니다.
- Five-seed 안정성 확인은 later입니다.
