# Stage 4: Market Context Fusion + FiLM Conditioning

## English

This folder is the Stage 4 workspace for testing whether structured market
context can improve the BTC chart-image CNN through conditional feature
modulation.

Stage 4 objective:
- Keep the Stage 2 BTC image-generation, label, split, normalization,
  evaluation, trading metric, and Grad-CAM pipeline fixed.
- Use the selected five-seed Stage 2 baseline as the primary image model:
  `I60/R20/ohlc_ma_vb`.
- Build a structured market context vector from information available at or
  before the image end date `t`.
- Compare simple context fusion against FiLM-style conditional modulation.
- Use the comparison to answer why FiLM is needed rather than only adding more
  parameters or appending side information.

Primary Stage 4 image input:
- `I60/R20/ohlc_ma_vb`
- This is selected because the Stage 2 selected five-seed check found it to be
  the strongest candidate:
  - accuracy mean `0.5793`
  - accuracy std `0.0182`
  - ROC-AUC mean `0.5849`
- The seed-42 Stage 2 run is useful for metadata and Grad-CAM inspection, but
  the Stage 4 baseline claim should use the five-seed mean.

Stage 3 dependency position:
- Stage 3 Linear is not a Stage 4 architecture dependency.
- It is kept as a negative/simple-parameter ablation.
- The preliminary matching seed-42 test dropped from Stage 2 accuracy
  `0.603053` and ROC-AUC `0.616950` to Stage 3 Linear accuracy `0.541291` and
  ROC-AUC `0.522101`.

Structured market context candidates:
- Fear & Greed score.
- Bollinger %B.
- Bollinger bandwidth.
- Money Flow Index.
- Realized volatility.

Important modeling rule:
- These context values are not drawn into the chart image in the main Stage 4
  experiment.
- The chart image remains the Stage 2 `ohlc_ma_vb` image.
- The context values are fed as a separate numeric vector.
- The numeric context vector is normalized with train-only statistics, encoded
  by an MLP, and then used by the fusion/modulation model.

Context encoder and normalization decision:
- Primary model input uses 8 matched-window features:
  `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`,
  `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, and `rv_60`.
- F&G and MFI-style 0-100 values are scaled by `100`; positive skewed
  volatility/range values use `log1p`.
- All splits use train-only median imputation, train-only 1/99% clipping, and
  train-only z-score normalization.
- The shared context encoder is intentionally small:
  `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
- Primary sample timing note:
  - `I60/R20/ohlc_ma_vb` requires both a 60-day image and a valid 60-day MA
    line inside that image.
  - With BTC OHLCV starting on `2018-01-01`, the first valid primary sample end
    date is `2018-04-29`.
  - The offset is `118` days because the MA60 and image windows are both
    inclusive: `(60 - 1) + (60 - 1) = 118`.
  - F&G starts on `2018-02-01`, so the F&G start date does not remove any valid
    primary samples.
  - Only one raw F&G missing date directly overlaps a primary sample end date:
    `2024-10-26`; it is handled with the previous available F&G value.

Main Stage 4 ablation models:

| Track | Model | What changes | Interpretation |
|:---|:---|:---|:---|
| `4-A` | CNN + context concat | Append context embedding to CNN feature before classifier | Tests whether simple side-information fusion is enough |
| `4-B` | CNN + context gating | Use context to multiply CNN channels/features by learned gates | Tests a simpler modulation alternative |
| `4-C` | CNN + context FiLM gamma-only | Use context to generate block-wise `gamma`; apply `F' = gamma * F` | Tests FiLM scaling without additive shift |
| `4-D` | CNN + context FiLM full | Use context to generate block-wise `gamma` and `beta`; apply `F' = gamma * F + beta` | Main FiLM model and interpretability target |

Insertion decision:
- `4-A` concatenates the context embedding after the I60 CNN flatten feature:
  `(B, 184320) + (B, 32) -> (B, 184352)`.
- `4-B` gates the final I60 feature map only:
  `(B, 512, 2, 180)` with `gate = 2 * sigmoid(raw_gate)`.
- `4-C` and `4-D` insert FiLM after BatchNorm and before LeakyReLU in every
  I60 convolution block.
- Gate, gamma, and beta heads are zero-initialized so gating starts at `1`,
  gamma starts at `1`, and beta starts at `0`.

Explanation/export decision:
- Grad-CAM target is the predicted-class pre-softmax logit.
- Final Stage 4 figures use `10` Predicted Up and `10` Predicted Down test
  samples; smoke runs may use `2` per predicted class.
- 4-A exports Grad-CAM plus context values.
- 4-B exports Grad-CAM plus context and final-layer gate values.
- 4-C exports Grad-CAM on post-gamma feature maps plus context and gamma values.
- 4-D exports Grad-CAM on post-gamma/beta feature maps plus context, gamma, and
  beta values.

Kaggle runner and backup decision:
- The first implementation runner should execute the numeric-context four-way
  ablation only: `concat`, `gating`, `film_gamma`, and `film_full`.
- The runner needs access to the Stage 2 `src` package because Stage 4 reuses
  the fixed BTC data/image/split/evaluation pipeline instead of rewriting it.
- First full sanity run: all four ablations with seed `42`.
- Later robustness run: the same four ablations with seeds
  `42, 43, 44, 45, 46`.
- Backup root is fixed to `/kaggle/working/stage4_saved_outputs`.
- The runner must backup after context build, training, prediction evaluation,
  trading evaluation, Grad-CAM/export, output check, and summary.
- An experiment is complete only when the output checker confirms checkpoint,
  predictions, metrics, trading metrics, Grad-CAM, context exports, modulation
  exports where applicable, and manifests. Checkpoint existence alone is not a
  completed Stage 4 result.

Why this matches the advisor's direction:
- The chart-image CNN baseline is already strong.
- The research question is not simply whether chart images work.
- The research question is whether market context can make the visual feature
  extractor more adaptive to regime/state.
- FiLM is useful because the path from context to feature modulation is explicit:
  the model produces gates/gamma/beta values that can be analyzed by date,
  market regime, layer, channel, confidence, and correctness.

Advisor direction file mapping:
- Source: `/Users/jaehyeonjeong/Desktop/film_chart_research_summary.md`.
- It frames FiLM as conditional modulation for chart-image features, not as a
  full visual-question-answering task.
- It states that the prediction query is effectively fixed, so an RNN question
  encoder is not essential.
- It recommends compact structured metadata and an MLP/embedding-based condition
  encoder.
- It recommends comparing CNN-only, naive condition concatenation, FiLM, and an
  optional attention-based fusion alternative.
- This is why Stage 4 uses structured numeric context first and compares
  concat, gating, gamma-only FiLM, and full FiLM.

News-context position:
- News is not removed from the thesis.
- Candidate source: Hugging Face `edaschau/bitcoin_news`.
- Public metadata checked on 2026-05-25 shows `210,832` rows, date range
  2011-2025, and columns such as `date_time`, `title`, `source`, `url`, and
  `article_text`.
- News requires source audit, publication-time alignment, daily aggregation, and
  encoder/cache rules before model training.
- Therefore it is kept as a second-phase context track after the structured
  numeric-context ablation is stable.
- 4-3 decision: first news version should be headline-only, strict `t-1`, and
  encoded with train-fit non-LLM text features.

Implementation readiness decision:
- `4-I0` is complete.
- Stage 4 code should add a `stage2_dependency` config section and import Stage
  2 helpers for BTC loading, sample generation, image generation, split,
  normalization, evaluation, and trading metrics.
- Local BTC OHLCV data exists.
- Local F&G data is now available at
  `stage4_film_conditioning/F&G_data/fear_greed_index.csv`.
  The raw CSV is not tracked in GitHub; only the data availability note and
  audit summary are tracked.
- The first full execution target remains Kaggle, but local context feature
  development can now use the supplied F&G CSV.
- The detailed task map is stored in
  `reports/tables/stage4_implementation_task_map.csv`.

Implementation status:
- `4-I1` is complete.
- Added local/Kaggle configs, Stage 4 config/path/runtime/seed helpers, script
  path utilities, and `check_stage4_scaffold.py`.
- Local scaffold check passed and confirmed BTC, F&G, and Stage 2 `src`
  availability.
- `4-I2` is complete.
- Added F&G source audit, OHLCV-derived context features, and train-only context
  preprocessing.
- Local I60/R20/ohlc_ma_vb context build produced `2,399` rows:
  train `671`, validation `287`, test `1,441`.
- Primary context feature missing-rate warnings: none.
- `4-I3` is complete.
- Added the shared context MLP encoder:
  `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
- Local check passed on dummy context tensors and real normalized rows from the
  `4-I2` context table.
- `4-I4` is complete.
- Added the `CNN + context concat` model:
  - Stage 2 I60 Stock_CNN convolution blocks are reused unchanged.
  - The Stage 2 final classifier is replaced with
    `Dropout(0.5) -> Linear(184352, 2)`.
  - Tensor path:
    `(B, 1, 96, 180) -> CNN -> (B, 512, 2, 180) -> flatten (B, 184320)`;
    context `(B, 8) -> MLP -> (B, 32)`;
    concat `(B, 184352) -> logits (B, 2)`.
  - Parameter count check passed: `2,954,370`, which is `+1,408` vs the Stage
    2 I60 baseline.
- `4-I5` is complete.
- Added the `CNN + context gating` model:
  - Stage 2 I60 Stock_CNN convolution blocks are reused unchanged.
  - Context embedding `(B, 32)` generates a channel gate `(B, 512)`.
  - Gate formula: `gate = 2 * sigmoid(raw_gate)`.
  - The gate is applied to the final feature map `(B, 512, 2, 180)`.
  - The classifier input remains `(B, 184320)`.
  - Gate head is zero-initialized, so the model starts from identity
    modulation with gate min/max `1.0 / 1.0`.
  - Parameter count check passed: `2,971,202`, which is `+18,240` vs the Stage
    2 I60 baseline.
- `4-I6` is complete.
- Added the reusable FiLM building blocks:
  - `FeatureWiseAffineModulation` applies channel-wise `F' = gamma * F` or
    `F' = gamma * F + beta` to a CNN feature map.
  - `FilmParameterGenerator` maps the 32-dim context embedding to block-wise
    gamma/beta tensors for I60 channels `[64, 128, 256, 512]`.
  - Gamma-only generator parameter count: `31,680`.
  - Full gamma/beta generator parameter count: `63,360`.
  - Heads are zero-initialized so gamma starts at `1.0` and beta starts at
    `0.0`, preserving the Stage 2 CNN feature path at initialization.
  - Local FiLM layer/generator check passed on dummy feature maps and real
    normalized context rows.
- `4-I7` is complete.
- Added the gamma-only and full FiLM Stock_CNN models:
  - `FilmContextStockCNN` reuses the Stage 2 Stock_CNN convolution blocks.
  - FiLM is inserted in every I60 block as
    `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`.
  - `film_gamma`: `F' = gamma * F`; parameter count `2,985,986`, which is
    `+33,024` vs the Stage 2 I60 baseline.
  - `film_full`: `F' = gamma * F + beta`; parameter count `3,017,666`, which is
    `+64,704` vs the Stage 2 I60 baseline.
  - Local model shape checks passed for both models, including dummy tensors,
    real normalized context rows, and all-block identity initialization.
- `4-I8` is complete.
- Added the Stage 4 context runner:
  - `scripts/run_stage4_context_model.py` runs one context-conditioned training
    job.
  - `src/stage4_film/runners/context_experiment.py` reuses Stage 2 BTC
    data/image/split/pixel-normalization and attaches normalized context
    tensors.
  - `src/stage4_film/training/loop.py` trains with `model(image, context)`.
  - Gate/FiLM heads are reset to identity after generic weight initialization.
  - Local smoke training passed for `concat` and `film_gamma`.
- `4-I9` is complete.
- Added Stage 4 prediction and trading exports:
  - `scripts/evaluate_stage4_predictions.py` reloads a Stage 4 checkpoint and
    writes prediction CSV plus classification metrics.
  - `scripts/evaluate_stage4_trading.py` reads the prediction CSV and writes
    BTC long/flat and long/short trading metrics.
  - Classification/trading metric formulas reuse Stage 2; model inference uses
    `model(image, context)`.
  - Local export checks passed for `concat` and `film_gamma` smoke checkpoints.
- `4-I10` is complete.
- Added Stage 4 Grad-CAM plus context/modulation export:
  - `scripts/generate_stage4_gradcam_context.py` reloads a Stage 4 checkpoint
    and prediction CSV, then computes predicted-class Grad-CAM through
    `model(image, context)`.
  - The figure is saved with `samples.csv`, `modulation_summary.csv`, and
    `modulation_values.json`.
  - `concat` exports context values and context embedding summaries.
  - `gating` additionally exports raw gate and final gate values.
  - `film_gamma`/`film_full` export block-wise gamma and beta values.
  - Local Grad-CAM export checks passed for `concat` and `film_gamma` smoke
    checkpoints.
- Next step: `4-I11` local or small Kaggle smoke test.

Main documents:
- [Checklist](checklist.md)
- [Workflow diagram](workflow_diagram.md)
- [Stage 4 pipeline](docs/stage4_pipeline.md)
- [Context fusion ablation plan](docs/condition_track_plan.md)
- [Professor meeting direction brief](docs/professor_meeting_stage4_direction_brief.md)
- [FiLM insertion design](docs/film_insertion_design.md)
- [Source map](docs/source_map.md)
- [Planning report](checklist_results/4-1_context_fusion_and_news_plan.md)
- [News dataset audit](checklist_results/4-3_news_dataset_audit_and_feasibility.md)
- [Stage 2/Stage 3 dependency review](checklist_results/4-4_stage2_stage3_dependency_and_baseline_output_review.md)
- [Context encoder and normalization plan](checklist_results/4-5_context_encoder_and_normalization_plan.md)
- [Concat/gating/FiLM insertion design](checklist_results/4-6_concat_gating_film_insertion_design.md)
- [Grad-CAM plus context/gate/gamma/beta export plan](checklist_results/4-7_gradcam_context_modulation_export_plan.md)
- [Kaggle runner and output backup plan](checklist_results/4-8_kaggle_runner_and_output_backup_plan.md)
- [Implementation readiness review](checklist_results/4-I0_implementation_readiness_review.md)
- [Shared config/code scaffold](checklist_results/4-I1_shared_code_config_scaffold.md)
- [Structured context feature builder](checklist_results/4-I2_structured_context_feature_builder.md)
- [Context MLP encoder](checklist_results/4-I3_context_mlp_encoder.md)
- [Context concat model](checklist_results/4-I4_context_concat_model.md)
- [Context gating model](checklist_results/4-I5_context_gating_model.md)
- [FiLM layer and generator](checklist_results/4-I6_film_layer_generator.md)
- [FiLM context models](checklist_results/4-I7_film_context_models.md)
- [Stage 4 context runner](checklist_results/4-I8_stage4_context_runner.md)
- [Prediction and trading exports](checklist_results/4-I9_prediction_trading_exports.md)
- [Grad-CAM context/modulation export](checklist_results/4-I10_gradcam_context_modulation_export.md)

## 한국어

이 폴더는 structured market context가 BTC chart-image CNN을 개선할 수 있는지
확인하는 Stage 4 작업 공간입니다. 핵심은 단순히 이미지를 더 복잡하게 그리는 것이
아니라, 시장 맥락이 CNN feature를 조건부로 조절하게 만드는 것입니다.

Stage 4 목표:
- Stage 2의 BTC image generation, label, split, normalization, evaluation,
  trading metric, Grad-CAM 파이프라인은 고정합니다.
- primary image model은 Stage 2 selected five-seed best인
  `I60/R20/ohlc_ma_vb`로 둡니다.
- image end date `t`까지 이용 가능한 정보만 사용해서 structured market context
  vector를 만듭니다.
- 단순 context fusion과 FiLM-style conditional modulation을 비교합니다.
- FiLM이 단순 parameter 증가나 side information 추가보다 왜 필요한지 실험적으로
  방어합니다.

Stage 4 primary image input:
- `I60/R20/ohlc_ma_vb`
- Stage 2 selected five-seed check에서 가장 강한 후보였기 때문에 사용합니다:
  - accuracy mean `0.5793`
  - accuracy std `0.0182`
  - ROC-AUC mean `0.5849`
- seed 42 Stage 2 run은 metadata와 Grad-CAM 확인에 유용하지만, Stage 4 baseline
  claim은 five-seed mean을 기준으로 둡니다.

Stage 3 dependency 위치:
- Stage 3 Linear는 Stage 4 architecture dependency가 아닙니다.
- negative/simple-parameter ablation으로만 둡니다.
- 같은 조합의 seed 42 preliminary test에서 Stage 2 accuracy `0.603053`,
  ROC-AUC `0.616950`이 Stage 3 Linear accuracy `0.541291`, ROC-AUC `0.522101`로
  하락했습니다.

Structured market context 후보:
- Fear & Greed score.
- Bollinger %B.
- Bollinger bandwidth.
- Money Flow Index.
- Realized volatility.

중요한 모델링 규칙:
- 이 context 값들을 main Stage 4 실험에서 chart image 위에 추가로 그리지 않습니다.
- chart image는 Stage 2의 `ohlc_ma_vb` 이미지를 그대로 사용합니다.
- context 값은 별도 numeric vector로 모델에 들어갑니다.
- numeric context vector는 train split 통계로만 normalize하고, MLP로 encoding한 뒤
  fusion/modulation model에서 사용합니다.

Context encoder와 normalization 결정:
- Primary model input은 matched-window 8개 feature를 사용합니다:
  `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`,
  `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60`.
- F&G와 MFI처럼 0-100 scale인 값은 `100`으로 나누고, 양수 skew가 큰 volatility/range
  값은 `log1p`를 적용합니다.
- 모든 split에는 train-only median imputation, train-only 1/99% clipping,
  train-only z-score normalization을 적용합니다.
- Shared context encoder는 작게 고정합니다:
  `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
- Primary sample timing note:
  - `I60/R20/ohlc_ma_vb`는 60일 image와 이미지 내부의 유효한 60일 MA line이
    모두 필요합니다.
  - BTC OHLCV가 `2018-01-01`부터 시작하므로 첫 primary sample end date는
    `2018-04-29`입니다.
  - Offset은 `118`일입니다. MA60과 image window가 모두 inclusive라서
    `(60 - 1) + (60 - 1) = 118`입니다.
  - F&G는 `2018-02-01`부터 시작하므로 F&G 시작일 때문에 valid primary sample이
    제거되지는 않습니다.
  - Primary sample end date와 직접 겹치는 F&G 원본 missing date는
    `2024-10-26` 하루뿐이며, 직전 이용 가능 F&G 값으로 처리합니다.

Stage 4 주요 ablation model:

| Track | Model | 바뀌는 부분 | 해석 |
|:---|:---|:---|:---|
| `4-A` | CNN + context concat | classifier 직전에 CNN feature와 context embedding을 붙임 | 단순 side information 추가만으로 충분한지 확인 |
| `4-B` | CNN + context gating | context가 CNN channel/feature gate를 만들어 곱함 | 더 단순한 modulation 대안 |
| `4-C` | CNN + context FiLM gamma-only | context가 block별 `gamma`를 만들고 `F' = gamma * F` 적용 | additive shift 없는 FiLM scaling |
| `4-D` | CNN + context FiLM full | context가 block별 `gamma`, `beta`를 만들고 `F' = gamma * F + beta` 적용 | main FiLM model과 해석력 대상 |

삽입 위치 결정:
- `4-A`는 I60 CNN flatten feature 뒤에 context embedding을 붙입니다:
  `(B, 184320) + (B, 32) -> (B, 184352)`.
- `4-B`는 final I60 feature map 하나만 gate합니다:
  `(B, 512, 2, 180)`, `gate = 2 * sigmoid(raw_gate)`.
- `4-C`와 `4-D`는 모든 I60 convolution block에서 BatchNorm 뒤, LeakyReLU 전에
  FiLM을 삽입합니다.
- Gate/gamma/beta head는 zero-initialize해서 gate는 `1`, gamma는 `1`, beta는
  `0`에서 시작하게 합니다.

Explanation/export 결정:
- Grad-CAM target은 predicted-class pre-softmax logit입니다.
- 최종 Stage 4 figure는 test sample에서 Predicted Up 10개, Predicted Down 10개를
  사용합니다. Smoke run에서는 predicted class별 2개를 허용합니다.
- 4-A는 Grad-CAM과 context 값을 export합니다.
- 4-B는 Grad-CAM, context, final-layer gate 값을 export합니다.
- 4-C는 post-gamma feature map 기준 Grad-CAM과 context/gamma 값을 export합니다.
- 4-D는 post-gamma/beta feature map 기준 Grad-CAM과 context/gamma/beta 값을
  export합니다.

Kaggle runner와 backup 결정:
- 첫 구현 runner는 numeric-context 네 가지 ablation만 실행합니다:
  `concat`, `gating`, `film_gamma`, `film_full`.
- Stage 4는 고정된 BTC data/image/split/evaluation pipeline을 재작성하지 않고
  Stage 2 `src` package를 재사용하므로 runner는 Stage 2 `src`에도 접근해야 합니다.
- 첫 full sanity run은 네 ablation을 seed `42`로 실행합니다.
- 이후 robustness run은 같은 네 ablation을 seed `42, 43, 44, 45, 46`으로
  실행합니다.
- Backup root는 `/kaggle/working/stage4_saved_outputs`로 고정합니다.
- Runner는 context build, training, prediction evaluation, trading evaluation,
  Grad-CAM/export, output check, summary 뒤에 backup을 만들어야 합니다.
- Experiment는 output checker가 checkpoint, predictions, metrics, trading
  metrics, Grad-CAM, context exports, 해당 model의 modulation exports, manifest를
  확인해야 완료입니다. Checkpoint만 존재하는 것은 Stage 4 완료 결과가 아닙니다.

교수님 방향성과 맞는 이유:
- chart-image CNN baseline은 이미 강합니다.
- 핵심 질문은 chart image 자체가 예측력이 있는지가 아닙니다.
- 핵심 질문은 market context가 visual feature extractor를 regime/state에 따라
  더 적응적으로 만들 수 있는지입니다.
- FiLM은 context에서 feature modulation으로 가는 경로가 명시적입니다. 따라서
  gate/gamma/beta 값을 date, regime, layer, channel, confidence, correctness별로
  분석할 수 있습니다.

교수님 방향성 파일과의 연결:
- Source: `/Users/jaehyeonjeong/Desktop/film_chart_research_summary.md`.
- 해당 note는 FiLM을 full VQA가 아니라 chart-image feature의 conditional
  modulation으로 재정의합니다.
- 금융 예측 질문은 사실상 고정되어 있으므로 RNN question encoder가 필수는 아니라고
  정리합니다.
- compact structured metadata와 MLP/embedding-based condition encoder를 권장합니다.
- CNN-only, naive condition concatenation, FiLM, optional attention-based fusion
  비교를 권장합니다.
- 그래서 Stage 4는 structured numeric context를 먼저 사용하고 concat, gating,
  gamma-only FiLM, full FiLM을 비교합니다.

뉴스 context 위치:
- 뉴스는 논문에서 제거하지 않습니다.
- 후보 source: Hugging Face `edaschau/bitcoin_news`.
- 2026-05-25 기준 공개 metadata 확인 결과 `210,832` rows, 2011-2025 date range,
  `date_time`, `title`, `source`, `url`, `article_text` columns를 포함합니다.
- 뉴스는 source audit, publication-time alignment, daily aggregation,
  encoder/cache rule이 필요합니다.
- 따라서 structured numeric-context ablation이 안정화된 뒤 second-phase context
  track으로 유지합니다.
- 4-3 결정: 첫 news version은 headline-only, strict `t-1`, train-fit non-LLM
  text feature encoder로 시작합니다.

Implementation readiness 결정:
- `4-I0`은 완료됐습니다.
- Stage 4 code는 `stage2_dependency` config section을 추가하고, BTC loading,
  sample generation, image generation, split, normalization, evaluation,
  trading metric은 Stage 2 helper를 import해서 사용해야 합니다.
- 로컬 BTC OHLCV data가 있습니다.
- 로컬 F&G data도
  `stage4_film_conditioning/F&G_data/fear_greed_index.csv`에 추가됐습니다.
  Raw CSV는 GitHub에 올리지 않고, data availability note와 audit summary만
  추적합니다.
- 첫 full 실행 target은 여전히 Kaggle이지만, local context feature 개발은 이제
  제공된 F&G CSV로 진행할 수 있습니다.
- 상세 task map은 `reports/tables/stage4_implementation_task_map.csv`에 저장했습니다.

Implementation status:
- `4-I1`을 완료했습니다.
- Local/Kaggle config, Stage 4 config/path/runtime/seed helper, script path
  utility, `check_stage4_scaffold.py`를 추가했습니다.
- Local scaffold check를 통과했고 BTC, F&G, Stage 2 `src` 사용 가능성을
  확인했습니다.
- `4-I2`를 완료했습니다.
- F&G source audit, OHLCV-derived context feature, train-only context
  preprocessing을 추가했습니다.
- Local I60/R20/ohlc_ma_vb context build에서 `2,399` row가 생성됐습니다:
  train `671`, validation `287`, test `1,441`.
- Primary context feature missing-rate warning은 없습니다.
- `4-I3`을 완료했습니다.
- Shared context MLP encoder를 추가했습니다:
  `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
- Dummy context tensor와 `4-I2` context table의 실제 normalized row 모두에서
  local check를 통과했습니다.
- `4-I4`를 완료했습니다.
- `CNN + context concat` model을 추가했습니다:
  - Stage 2 I60 Stock_CNN convolution block은 그대로 재사용합니다.
  - Stage 2 final classifier를 `Dropout(0.5) -> Linear(184352, 2)`로
    교체합니다.
  - Tensor path:
    `(B, 1, 96, 180) -> CNN -> (B, 512, 2, 180) -> flatten (B, 184320)`;
    context `(B, 8) -> MLP -> (B, 32)`;
    concat `(B, 184352) -> logits (B, 2)`.
  - Parameter count check 통과: `2,954,370`, Stage 2 I60 baseline 대비
    `+1,408`.
- `4-I5`를 완료했습니다.
- `CNN + context gating` model을 추가했습니다:
  - Stage 2 I60 Stock_CNN convolution block은 그대로 재사용합니다.
  - Context embedding `(B, 32)`이 channel gate `(B, 512)`를 만듭니다.
  - Gate formula: `gate = 2 * sigmoid(raw_gate)`.
  - Gate는 마지막 feature map `(B, 512, 2, 180)`에 적용됩니다.
  - Classifier input은 `(B, 184320)` 그대로 유지됩니다.
  - Gate head는 zero-initialized라서 gate min/max `1.0 / 1.0`의 identity
    modulation에서 시작합니다.
  - Parameter count check 통과: `2,971,202`, Stage 2 I60 baseline 대비
    `+18,240`.
- `4-I6`을 완료했습니다.
- 재사용 가능한 FiLM building block을 추가했습니다:
  - `FeatureWiseAffineModulation`은 CNN feature map에 channel-wise
    `F' = gamma * F` 또는 `F' = gamma * F + beta`를 적용합니다.
  - `FilmParameterGenerator`는 32차원 context embedding에서 I60 channel
    `[64, 128, 256, 512]`에 맞는 block별 gamma/beta tensor를 만듭니다.
  - Gamma-only generator parameter count: `31,680`.
  - Full gamma/beta generator parameter count: `63,360`.
  - Head는 zero-initialized라서 gamma는 `1.0`, beta는 `0.0`에서 시작하고,
    initialization 시 Stage 2 CNN feature path를 보존합니다.
  - Local FiLM layer/generator check가 dummy feature map과 실제 normalized
    context row 모두에서 통과했습니다.
- `4-I7`을 완료했습니다.
- Gamma-only/full FiLM Stock_CNN model을 추가했습니다:
  - `FilmContextStockCNN`은 Stage 2 Stock_CNN convolution block을 재사용합니다.
  - 모든 I60 block에
    `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d` 순서로 FiLM을
    삽입합니다.
  - `film_gamma`: `F' = gamma * F`; parameter count `2,985,986`, Stage 2 I60
    baseline 대비 `+33,024`.
  - `film_full`: `F' = gamma * F + beta`; parameter count `3,017,666`, Stage 2
    I60 baseline 대비 `+64,704`.
  - 두 model 모두 dummy tensor, 실제 normalized context row, all-block identity
    initialization을 포함한 local model shape check를 통과했습니다.
- `4-I8`을 완료했습니다.
- Stage 4 context runner를 추가했습니다:
  - `scripts/run_stage4_context_model.py`는 context-conditioned training job
    하나를 실행합니다.
  - `src/stage4_film/runners/context_experiment.py`는 Stage 2 BTC
    data/image/split/pixel-normalization을 재사용하고 normalized context tensor를
    붙입니다.
  - `src/stage4_film/training/loop.py`는 `model(image, context)` 형태로
    학습합니다.
  - 일반 weight initialization 뒤 gate/FiLM head를 identity로 다시 reset합니다.
  - `concat`, `film_gamma` local smoke training을 통과했습니다.
- `4-I9`를 완료했습니다.
- Stage 4 prediction/trading export를 추가했습니다:
  - `scripts/evaluate_stage4_predictions.py`는 Stage 4 checkpoint를 다시 로드하고
    prediction CSV와 classification metrics를 저장합니다.
  - `scripts/evaluate_stage4_trading.py`는 prediction CSV를 읽고 BTC long/flat,
    long/short trading metrics를 저장합니다.
  - Classification/trading metric 공식은 Stage 2를 재사용하고, inference만
    `model(image, context)`를 사용합니다.
  - `concat`, `film_gamma` smoke checkpoint에서 local export check를 통과했습니다.
- `4-I10`을 완료했습니다.
- Stage 4 Grad-CAM plus context/modulation export를 추가했습니다:
  - `scripts/generate_stage4_gradcam_context.py`는 Stage 4 checkpoint와
    prediction CSV를 다시 로드하고 `model(image, context)` 경로로
    predicted-class Grad-CAM을 계산합니다.
  - Figure와 함께 `samples.csv`, `modulation_summary.csv`,
    `modulation_values.json`를 저장합니다.
  - `concat`은 context 값과 context embedding summary를 export합니다.
  - `gating`은 raw gate와 최종 gate 값까지 export합니다.
  - `film_gamma`/`film_full`은 block별 gamma/beta 값을 export합니다.
  - `concat`, `film_gamma` smoke checkpoint에서 local Grad-CAM export check를
    통과했습니다.
- 다음 단계는 `4-I11` local 또는 작은 Kaggle smoke test입니다.

주요 문서:
- [Checklist](checklist.md)
- [Workflow diagram](workflow_diagram.md)
- [Stage 4 pipeline](docs/stage4_pipeline.md)
- [Context fusion ablation plan](docs/condition_track_plan.md)
- [Professor meeting direction brief](docs/professor_meeting_stage4_direction_brief.md)
- [FiLM insertion design](docs/film_insertion_design.md)
- [Source map](docs/source_map.md)
- [Planning report](checklist_results/4-1_context_fusion_and_news_plan.md)
- [News dataset audit](checklist_results/4-3_news_dataset_audit_and_feasibility.md)
- [Stage 2/Stage 3 dependency review](checklist_results/4-4_stage2_stage3_dependency_and_baseline_output_review.md)
- [Context encoder and normalization plan](checklist_results/4-5_context_encoder_and_normalization_plan.md)
- [Concat/gating/FiLM insertion design](checklist_results/4-6_concat_gating_film_insertion_design.md)
- [Grad-CAM plus context/gate/gamma/beta export plan](checklist_results/4-7_gradcam_context_modulation_export_plan.md)
- [Kaggle runner and output backup plan](checklist_results/4-8_kaggle_runner_and_output_backup_plan.md)
- [Implementation readiness review](checklist_results/4-I0_implementation_readiness_review.md)
- [Shared config/code scaffold](checklist_results/4-I1_shared_code_config_scaffold.md)
- [Structured context feature builder](checklist_results/4-I2_structured_context_feature_builder.md)
- [Context MLP encoder](checklist_results/4-I3_context_mlp_encoder.md)
- [Context concat model](checklist_results/4-I4_context_concat_model.md)
- [Context gating model](checklist_results/4-I5_context_gating_model.md)
- [FiLM layer and generator](checklist_results/4-I6_film_layer_generator.md)
- [FiLM context models](checklist_results/4-I7_film_context_models.md)
- [Stage 4 context runner](checklist_results/4-I8_stage4_context_runner.md)
- [Prediction and trading exports](checklist_results/4-I9_prediction_trading_exports.md)
- [Grad-CAM context/modulation export](checklist_results/4-I10_gradcam_context_modulation_export.md)
