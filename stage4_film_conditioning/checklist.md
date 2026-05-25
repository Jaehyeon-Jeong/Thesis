# Stage 4 Checklist

## English

Stage 4 now tests **how market context should be attached to the fixed BTC
chart-image CNN**. The image pipeline stays fixed; the comparison is between
context fusion/modulation methods.

Fixed baseline:
- Image/model family: Stage 2 `I60/R20/ohlc_ma_vb`.
- Reason: selected five-seed Stage 2 best configuration.
- Baseline metrics: accuracy mean `0.5793`, ROC-AUC mean `0.5849`.
- Stage 3 Linear result is kept as a negative/simple-parameter ablation.

Main Stage 4 ablation:
- [ ] 4-A. `CNN + context concat`
  - Context is encoded by MLP and appended to the CNN feature before the
    classifier.
  - Question: is simple side-information fusion enough?
- [ ] 4-B. `CNN + context gating`
  - Context creates channel/feature gates that multiply CNN features.
  - Question: is simple multiplicative modulation enough?
- [ ] 4-C. `CNN + context FiLM gamma-only`
  - Context creates block-wise `gamma`; apply `F' = gamma * F`.
  - Question: is FiLM-style scaling enough without additive shift?
- [ ] 4-D. `CNN + context FiLM full`
  - Context creates block-wise `gamma` and `beta`; apply
    `F' = gamma * F + beta`.
  - Question: does full FiLM give the best conditional adaptation and
    interpretability?

Planning phase:
- [x] 4-0. Stage 4 folder, checklist, and workflow scaffold
  - Result: [4-0 Stage 4 scaffold](checklist_results/4-0_stage4_scaffold.md)
- [x] 4-1. Context fusion and news-context plan
  - Result: [4-1 Context fusion and news plan](checklist_results/4-1_context_fusion_and_news_plan.md)
- [x] 4-2. Structured numeric context audit and leakage policy
  - F&G, Bollinger %B, Bollinger bandwidth, MFI, realized volatility.
  - Primary decision: `context_window = image_window`.
  - For the selected `I60/R20/ohlc_ma_vb` baseline, use matched 60-day
    context first: `F&G60`, `BB60`, `MFI60`, `RV60`.
  - Keep `BB20`, `MFI14`, and short F&G summaries only as later
    `standard_window` or `multi_scale` diagnostics.
  - Result: [4-2 Structured context audit and leakage policy](checklist_results/4-2_structured_context_audit_and_leakage_policy.md)
- [x] 4-3. News dataset audit and news-context feasibility decision
  - Candidate: `edaschau/bitcoin_news`.
  - Decision: feasible as a second-phase context source.
  - First news version: headline-only, strict `t-1` alignment, train-fit
    non-LLM encoder.
  - Defer article summaries and LLM embeddings until leakage-safe headline
    context is stable.
  - Result: [4-3 News dataset audit and feasibility decision](checklist_results/4-3_news_dataset_audit_and_feasibility.md)
- [x] 4-4. Stage 2/Stage 3 dependency and baseline-output review
  - Locked primary baseline: Stage 2 `I60/R20/ohlc_ma_vb`.
  - Primary comparison target: five-seed accuracy mean `0.579320`,
    ROC-AUC mean `0.584862`.
  - Stage 3 Linear is a negative/simple-parameter ablation, not a Stage 4 code
    dependency.
  - Result: [4-4 Stage 2/Stage 3 dependency and baseline output review](checklist_results/4-4_stage2_stage3_dependency_and_baseline_output_review.md)
- [x] 4-5. Context encoder and normalization plan
  - Primary context vector: 8 matched-window features.
  - Preprocessing: feature transform, train-only median imputation,
    train-only 1/99% clipping, train-only z-score normalization.
  - Shared encoder: `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
  - Result: [4-5 Context encoder and normalization plan](checklist_results/4-5_context_encoder_and_normalization_plan.md)
- [x] 4-6. Concat/gating/FiLM insertion design
  - 4-A concat attaches the 32-dim context embedding after CNN flatten:
    `184320 + 32 -> Linear(..., 2)`.
  - 4-B gating applies a final-block channel gate to `(B, 512, 2, 180)`.
  - 4-C/4-D FiLM is inserted after BatchNorm and before LeakyReLU in every I60
    block.
  - Result: [4-6 Concat/gating/FiLM insertion design](checklist_results/4-6_concat_gating_film_insertion_design.md)
- [x] 4-7. Grad-CAM plus context/gate/gamma/beta export plan
  - Primary target is the predicted-class pre-softmax logit.
  - Final report figure uses 10 Predicted Up and 10 Predicted Down test samples.
  - Export context values and gate/gamma/beta values beside Grad-CAM samples.
  - Result: [4-7 Grad-CAM plus context/gate/gamma/beta export plan](checklist_results/4-7_gradcam_context_modulation_export_plan.md)
- [x] 4-8. Kaggle runner and output backup plan
  - Runner stages: context build, training, prediction evaluation, trading
    evaluation, Grad-CAM/export, output check, and summary.
  - Backup root: `/kaggle/working/stage4_saved_outputs`.
  - Completion requires the output checker to pass; checkpoint existence alone
    is not enough.
  - Result: [4-8 Kaggle runner and output backup plan](checklist_results/4-8_kaggle_runner_and_output_backup_plan.md)

Advisor confirmation/reporting:
- [x] 4-R1. Professor meeting direction brief
  - Purpose: summarize Stage 1-4 progress, explain why Stage 4 is
    market-context-conditioned feature modulation, and map 4-A/B/C/D to the
    advisor direction file excerpts.
  - Result: [Professor meeting direction brief](docs/professor_meeting_stage4_direction_brief.md)
- [ ] 4-R2. Advisor feedback and final Stage 4 scope lock
  - Confirm whether Stage 4 should proceed with `matched_window` numeric
    context first.
  - Confirm whether news/LLM remains second-phase after numeric context.
  - Confirm whether 4-A concat, 4-B gating, 4-C gamma-only FiLM, and 4-D full
    FiLM are the intended ablation set.

Implementation phase:
- [x] 4-I0. Implementation readiness review
  - Decision: implementation can proceed to `4-I1`.
  - Stage 4 will reuse Stage 2 BTC data/image/split/evaluation helpers through
    a configurable Stage 2 `src` dependency.
  - Local BTC data is available; full context feature construction needs Kaggle
    F&G attachment or a supplied local F&G CSV.
  - Result: [4-I0 Implementation readiness review](checklist_results/4-I0_implementation_readiness_review.md)
- [ ] 4-I1. Shared Stage 4 config/code scaffold
- [ ] 4-I2. Structured context feature builder
- [ ] 4-I3. Context MLP encoder
- [ ] 4-I4. `CNN + context concat` model
- [ ] 4-I5. `CNN + context gating` model
- [ ] 4-I6. FiLM layer and FiLM generator modules
- [ ] 4-I7. `CNN + FiLM gamma-only` and `CNN + FiLM full` models
- [ ] 4-I8. BTC Stage 4 runner using fixed Stage 2 data pipeline
- [ ] 4-I9. Prediction, classification metric, and trading metric export
- [ ] 4-I10. Grad-CAM plus context/gate/gamma/beta export
- [ ] 4-I11. Local or small Kaggle smoke test
- [ ] 4-I12. Kaggle single-config run for the four main ablations
- [ ] 4-I13. Kaggle selected grid/five-seed runner
- [ ] 4-I14. Stage 4 result report

News-context extension:
- [ ] 4-N1. `edaschau/bitcoin_news` source audit
- [ ] 4-N2. Publication-time alignment and no-future-leakage check
- [ ] 4-N3. Daily aggregation policy
  - Options: latest headline, top-k headlines, concatenated headlines,
    article summary, or daily embedding average.
- [ ] 4-N4. Non-LLM news encoder
  - Options: TF-IDF/SVD, trainable embedding + GRU, sentence-transformer style
    embedding if reproducibility is fixed.
- [ ] 4-N5. LLM-summary or LLM-embedding plan
  - Deferred until data alignment and caching/reproducibility are fixed.
- [ ] 4-N6. News-context concat/gating/FiLM comparison

Important:
- Do not draw the context values into the chart image for the main Stage 4
  experiment.
- The context enters as a separate vector.
- All context features must be available at or before image end date `t`.
- Train-only statistics must be used for context normalization.
- News is not removed from the thesis. It is a second-phase context track after
  the structured numeric-context ablation is stable.

## 한국어

Stage 4는 이제 **market context를 고정된 BTC chart-image CNN에 어떻게 붙일지**를
검증하는 단계입니다. 이미지 파이프라인은 고정하고, context fusion/modulation 방식을
비교합니다.

고정 baseline:
- Image/model family: Stage 2 `I60/R20/ohlc_ma_vb`.
- 이유: Stage 2 selected five-seed best configuration.
- Baseline metrics: accuracy mean `0.5793`, ROC-AUC mean `0.5849`.
- Stage 3 Linear 결과는 단순 parameter 증가 비교의 negative ablation으로 둡니다.

Stage 4 main ablation:
- [ ] 4-A. `CNN + context concat`
  - context를 MLP로 encoding한 뒤 classifier 직전 CNN feature에 붙입니다.
  - 질문: 단순 side information 추가만으로 충분한가?
- [ ] 4-B. `CNN + context gating`
  - context가 channel/feature gate를 만들고 CNN feature에 곱합니다.
  - 질문: 단순 multiplicative modulation만으로 충분한가?
- [ ] 4-C. `CNN + context FiLM gamma-only`
  - context가 block별 `gamma`를 만들고 `F' = gamma * F`를 적용합니다.
  - 질문: additive shift 없이 FiLM-style scaling만으로 충분한가?
- [ ] 4-D. `CNN + context FiLM full`
  - context가 block별 `gamma`, `beta`를 만들고 `F' = gamma * F + beta`를 적용합니다.
  - 질문: full FiLM이 conditional adaptation과 해석력에서 가장 좋은가?

계획 단계:
- [x] 4-0. Stage 4 폴더, checklist, workflow scaffold
  - 결과: [4-0 Stage 4 scaffold](checklist_results/4-0_stage4_scaffold.md)
- [x] 4-1. Context fusion과 news-context 계획
  - 결과: [4-1 Context fusion and news plan](checklist_results/4-1_context_fusion_and_news_plan.md)
- [x] 4-2. Structured numeric context audit와 leakage policy
  - F&G, Bollinger %B, Bollinger bandwidth, MFI, realized volatility.
  - Primary decision: `context_window = image_window`.
  - 선택된 `I60/R20/ohlc_ma_vb` baseline에서는 60일 matched context를 먼저 사용:
    `F&G60`, `BB60`, `MFI60`, `RV60`.
  - `BB20`, `MFI14`, short F&G summary는 나중의 `standard_window` 또는
    `multi_scale` diagnostic으로만 유지합니다.
  - 결과: [4-2 Structured context audit and leakage policy](checklist_results/4-2_structured_context_audit_and_leakage_policy.md)
- [x] 4-3. News dataset audit와 news-context 사용 가능성 결정
  - 후보: `edaschau/bitcoin_news`.
  - 결정: second-phase context source로 사용 가능.
  - 첫 news version: headline-only, strict `t-1` alignment, train-fit
    non-LLM encoder.
  - Article summary와 LLM embedding은 leakage-safe headline context가 안정화된
    뒤로 미룹니다.
  - 결과: [4-3 News dataset audit and feasibility decision](checklist_results/4-3_news_dataset_audit_and_feasibility.md)
- [x] 4-4. Stage 2/Stage 3 dependency와 baseline output 확인
  - Primary baseline 고정: Stage 2 `I60/R20/ohlc_ma_vb`.
  - Primary comparison target: five-seed accuracy mean `0.579320`,
    ROC-AUC mean `0.584862`.
  - Stage 3 Linear는 negative/simple-parameter ablation이며, Stage 4 code
    dependency가 아닙니다.
  - 결과: [4-4 Stage 2/Stage 3 dependency and baseline output review](checklist_results/4-4_stage2_stage3_dependency_and_baseline_output_review.md)
- [x] 4-5. Context encoder와 normalization 계획
  - Primary context vector: matched-window 8개 feature.
  - Preprocessing: feature transform, train-only median imputation,
    train-only 1/99% clipping, train-only z-score normalization.
  - Shared encoder: `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
  - 결과: [4-5 Context encoder and normalization plan](checklist_results/4-5_context_encoder_and_normalization_plan.md)
- [x] 4-6. Concat/gating/FiLM 삽입 설계
  - 4-A concat은 CNN flatten 뒤 32차원 context embedding을 붙입니다:
    `184320 + 32 -> Linear(..., 2)`.
  - 4-B gating은 final block feature map `(B, 512, 2, 180)`에 channel gate를
    적용합니다.
  - 4-C/4-D FiLM은 모든 I60 block에서 BatchNorm 뒤, LeakyReLU 전에 삽입합니다.
  - 결과: [4-6 Concat/gating/FiLM insertion design](checklist_results/4-6_concat_gating_film_insertion_design.md)
- [x] 4-7. Grad-CAM plus context/gate/gamma/beta export 계획
  - Primary target은 predicted-class pre-softmax logit입니다.
  - 최종 보고 figure는 test sample에서 Predicted Up 10개, Predicted Down 10개를
    사용합니다.
  - Grad-CAM sample 옆에 context 값과 gate/gamma/beta 값을 같이 export합니다.
  - 결과: [4-7 Grad-CAM plus context/gate/gamma/beta export plan](checklist_results/4-7_gradcam_context_modulation_export_plan.md)
- [x] 4-8. Kaggle runner와 output backup 계획
  - Runner 단계: context build, training, prediction evaluation, trading
    evaluation, Grad-CAM/export, output check, summary.
  - Backup root: `/kaggle/working/stage4_saved_outputs`.
  - 완료 판정은 output checker 통과 기준입니다. Checkpoint 존재만으로는 완료가
    아닙니다.
  - 결과: [4-8 Kaggle runner and output backup plan](checklist_results/4-8_kaggle_runner_and_output_backup_plan.md)

교수님 확인/보고:
- [x] 4-R1. 교수님 미팅용 방향성 brief
  - 목적: Stage 1-4 진행 현황, Stage 4를 market-context-conditioned feature
    modulation으로 해석한 이유, 4-A/B/C/D가 교수님 방향성 파일의 어떤 발췌와
    연결되는지 정리합니다.
  - 결과: [Professor meeting direction brief](docs/professor_meeting_stage4_direction_brief.md)
- [ ] 4-R2. 교수님 피드백 반영과 Stage 4 최종 scope lock
  - Stage 4를 `matched_window` numeric context부터 진행하는 것이 맞는지 확인합니다.
  - news/LLM을 numeric context 이후 second-phase로 두는 것이 맞는지 확인합니다.
  - 4-A concat, 4-B gating, 4-C gamma-only FiLM, 4-D full FiLM이 의도한
    ablation set인지 확인합니다.

구현 단계:
- [x] 4-I0. 구현 readiness review
  - 결정: `4-I1` 구현으로 진행할 수 있습니다.
  - Stage 4는 configurable Stage 2 `src` dependency를 통해 Stage 2 BTC
    data/image/split/evaluation helper를 재사용합니다.
  - 로컬 BTC data는 있지만 full context feature construction에는 Kaggle F&G
    attach 또는 별도 local F&G CSV가 필요합니다.
  - 결과: [4-I0 Implementation readiness review](checklist_results/4-I0_implementation_readiness_review.md)
- [ ] 4-I1. Stage 4 공통 config/code scaffold
- [ ] 4-I2. Structured context feature builder
- [ ] 4-I3. Context MLP encoder
- [ ] 4-I4. `CNN + context concat` model
- [ ] 4-I5. `CNN + context gating` model
- [ ] 4-I6. FiLM layer와 FiLM generator module
- [ ] 4-I7. `CNN + FiLM gamma-only`와 `CNN + FiLM full` model
- [ ] 4-I8. 고정된 Stage 2 data pipeline을 쓰는 BTC Stage 4 runner
- [ ] 4-I9. prediction, classification metric, trading metric export
- [ ] 4-I10. Grad-CAM plus context/gate/gamma/beta export
- [ ] 4-I11. local 또는 작은 Kaggle smoke test
- [ ] 4-I12. 네 가지 main ablation의 Kaggle single-config run
- [ ] 4-I13. Kaggle selected grid/five-seed runner
- [ ] 4-I14. Stage 4 결과 보고

News-context 확장:
- [ ] 4-N1. `edaschau/bitcoin_news` source audit
- [ ] 4-N2. Publication-time alignment와 no-future-leakage check
- [ ] 4-N3. Daily aggregation policy
  - 선택지: latest headline, top-k headline, headline concat, article summary,
    daily embedding average.
- [ ] 4-N4. Non-LLM news encoder
  - 선택지: TF-IDF/SVD, trainable embedding + GRU, 재현성 고정된
    sentence-transformer style embedding.
- [ ] 4-N5. LLM-summary 또는 LLM-embedding plan
  - 데이터 정렬과 cache/reproducibility가 고정된 뒤로 미룹니다.
- [ ] 4-N6. News-context concat/gating/FiLM comparison

중요:
- Main Stage 4 실험에서 context 값을 chart image 위에 추가로 그리지 않습니다.
- context는 별도 vector로 들어갑니다.
- 모든 context feature는 image end date `t` 또는 그 이전에 알 수 있어야 합니다.
- context normalization은 train split 통계로만 fit합니다.
- 뉴스는 논문에서 제거하지 않습니다. structured numeric-context ablation이 안정화된 뒤
  second-phase context track으로 사용합니다.
