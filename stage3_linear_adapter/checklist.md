# Stage 3 Checklist

## English

Proceed one item at a time. Stage 3 must keep the Stage 2 BTC baseline pipeline
fixed and add only the Linear adapter comparison.

Planning phase:
- [x] 3-0. Stage 3 folder and planning documents
  - Result: [3-0 Stage 3 scaffold](checklist_results/3-0_stage3_scaffold.md)
- [x] 3-1. Stage 2 dependency and baseline-output review
  - Result: [3-1 dependency review](checklist_results/3-1_stage2_dependency_baseline_review.md)
- [x] 3-2. Linear adapter design and insertion-point review
  - Result: [3-2 adapter design](checklist_results/3-2_linear_adapter_design.md)
- [x] 3-3. Training/evaluation comparison plan
  - Result: [3-3 training/evaluation plan](checklist_results/3-3_training_evaluation_comparison_plan.md)
- [x] 3-4. Grad-CAM comparison plan
  - Result: [3-4 Grad-CAM plan](checklist_results/3-4_gradcam_comparison_plan.md)
- [x] 3-5. Kaggle runner and output plan
  - Result: [3-5 Kaggle/output plan](checklist_results/3-5_kaggle_runner_output_plan.md)

Implementation phase:
- [ ] 3-I0. Implementation readiness review
- [ ] 3-I1. Shared Stage 3 config/code scaffold
- [ ] 3-I2. Linear adapter model code
- [ ] 3-I3. BTC Linear runner using fixed Stage 2 data pipeline
- [ ] 3-I4. Prediction, classification metric, and trading metric export
- [ ] 3-I5. Baseline-vs-Linear Grad-CAM export
- [ ] 3-I6. Local or small Kaggle smoke test
- [ ] 3-I7. Kaggle full Linear baseline runs
- [ ] 3-I8. Stage 3 result report

Important:
- Do not modify Stage 2 image generation, label construction, split,
  normalization, or trading metric logic unless a checklist item explicitly
  records the reason.
- Do not treat Dense Linear and Gamma-only FiLM as identical. Stage 3 is a
  simple comparison bridge before FiLM.
- Grad-CAM remains required for Stage 3. Compare baseline and Linear heatmaps on
  the same sample/date whenever possible.

## 한국어

한 항목씩 진행합니다. Stage 3는 Stage 2 BTC baseline pipeline을 고정하고 Linear
adapter 비교만 추가해야 합니다.

계획 단계:
- [x] 3-0. Stage 3 폴더와 planning 문서
  - 결과: [3-0 Stage 3 scaffold](checklist_results/3-0_stage3_scaffold.md)
- [x] 3-1. Stage 2 dependency와 baseline output 확인
  - 결과: [3-1 dependency review](checklist_results/3-1_stage2_dependency_baseline_review.md)
- [x] 3-2. Linear adapter 설계와 삽입 위치 확인
  - 결과: [3-2 adapter design](checklist_results/3-2_linear_adapter_design.md)
- [x] 3-3. training/evaluation 비교 계획
  - 결과: [3-3 training/evaluation plan](checklist_results/3-3_training_evaluation_comparison_plan.md)
- [x] 3-4. Grad-CAM 비교 계획
  - 결과: [3-4 Grad-CAM plan](checklist_results/3-4_gradcam_comparison_plan.md)
- [x] 3-5. Kaggle runner와 output 계획
  - 결과: [3-5 Kaggle/output plan](checklist_results/3-5_kaggle_runner_output_plan.md)

구현 단계:
- [ ] 3-I0. 구현 readiness review
- [ ] 3-I1. Stage 3 공통 config/code scaffold
- [ ] 3-I2. Linear adapter model code
- [ ] 3-I3. 고정된 Stage 2 data pipeline을 사용하는 BTC Linear runner
- [ ] 3-I4. prediction, classification metric, trading metric export
- [ ] 3-I5. Baseline-vs-Linear Grad-CAM export
- [ ] 3-I6. local 또는 작은 Kaggle smoke test
- [ ] 3-I7. Kaggle full Linear baseline run
- [ ] 3-I8. Stage 3 결과 보고

중요:
- Stage 2 image generation, label construction, split, normalization, trading
  metric logic은 임의로 바꾸지 않습니다. 바꿀 필요가 있으면 checklist 항목에서 이유를
  먼저 기록합니다.
- Dense Linear와 Gamma-only FiLM을 동일한 것으로 취급하지 않습니다. Stage 3는
  FiLM 전의 단순 비교 bridge입니다.
- Stage 3에서도 Grad-CAM은 필수입니다. 가능하면 같은 sample/date 기준으로 baseline과
  Linear heatmap을 비교합니다.
