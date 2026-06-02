# Stage 4 Source Map

## English

| Source | Local/reference path | Stage 4 use |
| --- | --- | --- |
| FiLM paper summary | `자료조사/FiLM요약.md` | FiLM equation, gamma/beta interpretation, BN-FiLM placement |
| FiLM paper PDF | `자료조사/FiLM.pdf` | Exact paper citation check before final code comments |
| Advisor direction note | `/Users/jaehyeonjeong/Desktop/film_chart_research_summary.md` | Stage 4 research framing: fixed chart-image baseline, structured context, MLP encoder, concat/gating/FiLM comparison, interpretability |
| FiLM reference repository | `https://github.com/ethanjperez/film`, commit `fe43ddf8a22b339dcca2efa07091ce9d498955cf` | Reference implementation for FiLM generator/layer conventions |
| Stage 2 BTC model | `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py` | Base CNN block variants for I5/I20/I60 |
| Stage 2 BTC pipeline | `stage2_btc_extension/src/stage2_btc/` | Fixed BTC data, labels, split, normalization, evaluation |
| Stage 3 Linear model | `stage3_linear_adapter/src/stage3_linear/models/linear_stock_cnn.py` | Comparison model only |
| Grad-CAM local summary | `자료조사/Grad-CAM요약.md` | Stage 4 Grad-CAM rule and interpretation |
| Fear & Greed candidate dataset | `https://www.kaggle.com/datasets/ashishpatel8736/historical-and-fear-greed-index-datasets` | Candidate external sentiment/regime context source; must be audited for crypto-vs-equity meaning, date coverage, scale, and missing dates |
| BTC news candidate dataset | `https://huggingface.co/datasets/edaschau/bitcoin_news` | 4-3 audit source; feasible second-phase news context with strict `t-1` headline-only first policy |

Implementation-source distinction:
- Paper/reference reported: FiLM is feature-wise affine modulation with gamma
  and beta generated from a condition.
- Paper/reference reported: applying FiLM after BatchNorm is a known setting in
  the FiLM implementation notes.
- Implementation choice for this thesis: in Stock_CNN blocks, insert FiLM as
  `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`.
- Advisor-direction mapping:
  - The advisor note says the task should not be treated as full VQA; the
    financial prediction query is effectively fixed.
  - It says structured metadata can replace the original FiLM question encoder.
  - It recommends comparing CNN-only, naive condition concatenation, FiLM, and
    optional attention-based fusion.
  - Therefore Stage 4 uses numeric context + MLP encoder and compares concat,
    gating, gamma-only FiLM, and full FiLM.
- Short excerpts from the advisor note that determine the model set:
  - "strong no-conditioning baseline"
  - "the prediction query is effectively fixed"
  - "compact structured metadata"
  - "an MLP or embedding-based condition encoder is a cleaner design than an RNN"
  - "CNN + naive condition concatenation"
  - "CNN + FiLM"
  - "Optional attention-based fusion"
- Implementation choice for the first Stage 4 main run: structured numeric
  context first; news context remains a second-phase track after audit.
- 4-4 baseline dependency decision:
  - Stage 4 primary image/model baseline is Stage 2 `I60/R20/ohlc_ma_vb`.
  - Primary comparison target is the Stage 2 selected five-seed mean:
    accuracy `0.579320`, ROC-AUC `0.584862`.
  - Stage 3 Linear is a negative/simple-parameter ablation only; it is not a
    Stage 4 architecture dependency.
- 4-3 news-context decision:
  - `edaschau/bitcoin_news` is feasible because it overlaps the Stage 2 BTC
    period and has timestamped title/article/source fields.
  - First news version is headline-only, strict `t-1`, train-fit non-LLM
    encoder.
  - Full article summaries and LLM embeddings are deferred until leakage-safe
    headline context is stable.
- Structured context source split:
  - F&G requires an external dataset.
  - Bollinger %B, Bollinger bandwidth, MFI, and realized volatility are derived
    from BTC OHLCV and do not require additional data.
- 4-5 context encoder and normalization decision:
  - Primary first-run model input has 8 features:
    `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`,
    `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60`.
  - Preprocessing uses train-only median imputation, train-only 1/99% clipping,
    and train-only z-score normalization after feature-specific transforms.
  - Shared encoder is
    `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
- 4-6 insertion decision:
  - Concat attaches the `(B, 32)` context embedding after I60 flatten:
    `(B, 184320) -> (B, 184352)`.
  - Gating uses only a final-block channel gate on `(B, 512, 2, 180)`.
  - Gamma-only and full FiLM are inserted after BatchNorm and before LeakyReLU
    in every I60 block.
  - Gate/FiLM output heads are zero-initialized so gate/gamma/beta start as
    identity modulation.
- 4-7 Grad-CAM/export decision:
  - Primary Grad-CAM target is the predicted-class pre-softmax logit.
  - Final report selection is 10 Predicted Up and 10 Predicted Down test
    samples.
  - 4-B/4-C/4-D export context plus gate/gamma/beta values beside the selected
    Grad-CAM samples.
  - FiLM heatmaps use post-FiLM conditioned feature maps as the primary target
    layers.
- 4-8 Kaggle runner/output backup decision:
  - The first Stage 4 Kaggle runner executes the four numeric-context
    ablations: `concat`, `gating`, `film_gamma`, and `film_full`.
  - First full sanity run uses seed `42`; later robustness run uses seeds
    `42, 43, 44, 45, 46`.
  - Backup root is `/kaggle/working/stage4_saved_outputs`.
  - Completion is defined by output-check success, not checkpoint existence.
  - If checkpoint exists but prediction/metric/Grad-CAM exports are missing,
    the runner resumes evaluation/export instead of declaring the run complete.
- 4-I0 implementation readiness decision:
  - Stage 4 can proceed to implementation.
  - Stage 4 reuses Stage 2 BTC loading, sample/image generation,
    split/normalization, prediction metric, trading metric, and Grad-CAM helper
    code through a configurable Stage 2 `src` dependency.
  - Local BTC OHLCV data is available.
  - Local F&G data is now available at
    `stage4_film_conditioning/FG_data/fear_greed_index.csv`; raw F&G CSV files
    are not tracked in GitHub.
- 4-I1 scaffold decision:
  - Added `configs/env_local.yaml` and `configs/env_kaggle.yaml`.
  - Added Stage 4 config/path/runtime/seed helpers.
  - Added shared script path utility that exposes both Stage 4 `src` and
    Stage 2 `src`.
  - Added `scripts/check_stage4_scaffold.py`; local scaffold check passed for
    BTC, F&G, and Stage 2 dependency paths.
- 4-I2 structured context feature builder decision:
  - Added `src/stage4_film/context/sources.py`,
    `src/stage4_film/context/features.py`, and
    `src/stage4_film/context/normalization.py`.
  - Added `scripts/audit_stage4_context_sources.py` and
    `scripts/build_stage4_context_features.py`.
  - Local I60/R20/ohlc_ma_vb context build produced 2,399 rows:
    train 671, validation 287, test 1,441.
  - F&G as-of coverage is complete for the primary sample set; max as-of age is
    1 day because the F&G dataset has a missing calendar date on 2024-10-26.
  - All 8 primary context features have 0.0 missing rate in train,
    validation, and test after the matched-window sample restrictions.
  - Primary sample timing detail:
    - `I60/R20/ohlc_ma_vb` requires a 60-day image and valid MA60 values for
      every day inside the image.
    - With BTC OHLCV starting on 2018-01-01, the first valid sample end date is
      2018-04-29.
    - The exact offset is 118 days because both windows are inclusive:
      `(60 - 1) + (60 - 1) = 118`.
    - F&G starts on 2018-02-01, so it does not remove valid primary samples.
    - The only raw F&G missing date directly overlapping a primary sample end
      date is 2024-10-26, handled by previous-available-value as-of merge.
- 4-I3 context MLP encoder decision:
  - Added `src/stage4_film/conditions/context_encoder.py`.
  - Shared encoder architecture:
    `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
  - Parameter count is 1,344.
  - `scripts/check_stage4_context_encoder.py` passed on both dummy tensors and
    real normalized rows from the local `4-I2` context table.
- 4-I4 context concat model decision:
  - Added `src/stage4_film/models/context_stock_cnn.py`.
  - Added `scripts/check_stage4_model_shapes.py`.
  - Stage 2 I60 Stock_CNN convolution blocks are reused unchanged; only the
    final classifier is replaced.
  - Primary tensor path:
    image `(B, 1, 96, 180) -> flatten (B, 184320)`;
    context `(B, 8) -> embedding (B, 32)`;
    concat `(B, 184352) -> logits (B, 2)`.
  - Parameter count is `2,954,370`, which is `+1,408` vs Stage 2 I60 baseline.
  - The local model shape checker passed on dummy tensors and real normalized
    context rows from the local `4-I2` context table.
- 4-I5 context gating model decision:
  - Extended `src/stage4_film/models/context_stock_cnn.py` with
    `GatedContextStockCNN`.
  - Extended `scripts/check_stage4_model_shapes.py` with `--model gating`.
  - Stage 2 I60 Stock_CNN convolution blocks are reused unchanged.
  - Primary tensor path:
    image `(B, 1, 96, 180) -> final feature map (B, 512, 2, 180)`;
    context `(B, 8) -> embedding (B, 32) -> raw gate (B, 512)`;
    `gate = 2 * sigmoid(raw_gate)`;
    gated feature `(B, 512, 2, 180) -> flatten (B, 184320) -> logits (B, 2)`.
  - Gate head is zero-initialized, so initial gate min/max is `1.0 / 1.0`.
  - Parameter count is `2,971,202`, which is `+18,240` vs Stage 2 I60 baseline.
  - The local model shape checker passed on dummy tensors and real normalized
    context rows from the local `4-I2` context table.
- 4-I6 FiLM layer/generator decision:
  - Added `src/stage4_film/layers/film.py`.
  - Added `src/stage4_film/conditions/film_generator.py`.
  - Added `scripts/check_stage4_film_layers.py`.
  - FiLM applies channel-wise modulation to a feature map:
    gamma-only `F' = gamma * F`; full FiLM `F' = gamma * F + beta`.
  - The generator uses the shared context embedding `(B, 32)` and emits
    block-wise gamma/beta parameters for I60 channels `[64, 128, 256, 512]`.
  - Gamma is implemented as `1 + delta_gamma`, with zero-initialized heads.
    Full FiLM beta is also zero-initialized.
  - This gives identity initialization: gamma min/max `1.0 / 1.0`, beta
    min/max `0.0 / 0.0`, and the modulated feature maps equal the original
    feature maps at initialization.
  - Generator parameter counts:
    - gamma-only: `31,680`
    - full gamma/beta: `63,360`
  - The local FiLM checker passed on dummy I60 feature maps and real normalized
    context rows from the local `4-I2` context table.
- 4-I7 FiLM Stock_CNN model decision:
  - Added `src/stage4_film/models/film_stock_cnn.py`.
  - Extended `scripts/check_stage4_model_shapes.py` with `--model film_gamma`
    and `--model film_full`.
  - Stage 2 I60 Stock_CNN convolution blocks are reused, but each block is
    executed as `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`.
  - `film_gamma` tensor path:
    context `(B, 8) -> embedding (B, 32) -> gamma for channels
    `[64, 128, 256, 512]`; final flatten `(B, 184320) -> logits (B, 2)`.
  - `film_full` uses the same path and additionally emits beta for every block.
  - Parameter counts:
    - `film_gamma`: `2,985,986`, `+33,024` vs Stage 2 I60 baseline.
    - `film_full`: `3,017,666`, `+64,704` vs Stage 2 I60 baseline.
  - Both local model checks passed on dummy tensors and real normalized context
    rows from the local `4-I2` context table.
  - In-place `LeakyReLU` can mutate post-FiLM tensors; the model stores
    pre-FiLM/post-FiLM feature maps before activation for later interpretation
    export.
- 4-I8 Stage 4 context runner decision:
  - Added `src/stage4_film/runners/context_experiment.py`.
  - Added `src/stage4_film/training/loop.py`.
  - Added `scripts/run_stage4_context_model.py`.
  - The runner reuses Stage 2 BTC data loading, sample construction, chart
    image generation, split, and train-only pixel normalization.
  - Stage 4 context features are built from the same split sample universe,
    normalized with train-only context statistics, and attached as
    `batch["context"]`.
  - Training calls `model(image, context)`.
  - Generic Stage 2 weight initialization is followed by a Stage 4 identity
    reset for gate/FiLM output heads so modulation starts from the Stage 2
    visual path.
  - Local smoke training passed for `concat` and `film_gamma`.
- 4-I9 prediction/trading export decision:
  - Added `src/stage4_film/evaluation/prediction.py`.
  - Added `scripts/evaluate_stage4_predictions.py`.
  - Added `scripts/evaluate_stage4_trading.py`.
  - Classification metrics reuse Stage 2 `compute_classification_metrics`.
  - Trading metrics reuse Stage 2 `compute_trading_metrics`.
  - The prediction helper changes inference from `model(image)` to
    `model(image, context)` and appends normalized context columns to the
    prediction CSV.
  - Local export checks passed for `concat` and `film_gamma` smoke checkpoints.
- 4-I10 Grad-CAM/context/modulation export decision:
  - Added `src/stage4_film/interpretability/gradcam_context.py`.
  - Added `scripts/generate_stage4_gradcam_context.py`.
  - Grad-CAM target remains the predicted-class pre-softmax logit.
  - Stage 4 Grad-CAM calls `model(image, context)` instead of `model(image)`.
  - Concat exports context values and context embedding summaries.
  - Gating exports context values, raw gate, and final gate values.
  - FiLM exports context values plus block-wise gamma/beta values.
  - FiLM Grad-CAM uses post-FiLM target modules and tensor-level gradient hooks
    to avoid PyTorch full-backward-hook conflicts with in-place LeakyReLU.
  - Local Grad-CAM export checks passed for `concat` and `film_gamma` smoke
    checkpoints.
- 4-I11 smoke output check decision:
  - Added `scripts/check_stage4_outputs.py`.
  - The checker treats a Stage 4 run as complete only when checkpoint,
    train metadata, predictions, classification metrics, trading metrics,
    Grad-CAM, selected samples, modulation exports, context artifacts, and run
    manifest are all present and parseable.
  - It checks CSV row counts and JSON parseability, not only file existence.
  - Local output checks passed for `concat` and `film_gamma` smoke runs.
- 4-I12 Kaggle four-ablation run decision/result:
  - Added `notebooks/kaggle_stage4_four_ablation_single_seed_one_cell.md`.
  - The first real Kaggle run is fixed to `I60/R20/ohlc_ma_vb`,
    `context_window=60`, seed `42`, and methods `concat`, `gating`,
    `film_gamma`, `film_full`.
  - The cell copies Stage 4 and Stage 2 code snapshots, patches
    `configs/env_kaggle.yaml`, audits BTC/F&G sources, builds context features,
    runs train/evaluation/trading/Grad-CAM/output-check per method, and writes
    backup zips under `/kaggle/working/stage4_saved_outputs`.
  - Kaggle seed-42 result is complete. `film_full` is the best Stage 4 method:
    accuracy `0.584316`, ROC-AUC `0.596811`.
  - Result table:
    `reports/tables/stage4_four_ablation_seed42_run_summary.csv`.
- 4-I13 Kaggle five-seed runner decision:
  - Added `notebooks/kaggle_stage4_four_ablation_five_seed_one_cell.md`.
  - The robustness run keeps the same selected configuration and runs seeds
    `42, 43, 44, 45, 46` for all four context methods.
  - Both Kaggle cells are execution wrappers. They do not change model
    definitions, context feature definitions, split rules, or evaluation
    metrics.
- Stage 4 v1 five-seed diagnosis:
  - The five-seed run showed that `film_full` is the best Stage 4 v1 method,
    but its mean accuracy `0.5510` and ROC-AUC `0.5677` remain below the
    selected Stage 2 visual baseline.
  - Seed `42` was promising, while seed `45` collapsed to all-Down predictions.
  - The next stage is diagnostic separation, not immediate FiLM complexity.
- Stage 4 v2 priority map:
  - `4-V0`: `I60/R20/ohlc_ma_vb`, visual-only, no context.
  - `4-V1`: `I60/R20/ohlc`, visual-only, no context.
  - `4-V2`: `I60/R20/ohlc` + all structured context + `film_full`.
  - `4-V3`: `I60/R20/ohlc` + F&G-only + `film_full`.
  - `4-V4`: `I60/R20/ohlc` + technical-only context + `film_full`.
  - `4-V5`: `I60/R20/ohlc` + all structured context + `film_full`, five
    seeds.
  - `4-V6`: `I60/R20/ohlc_ma_vb` + F&G-only + `film_full`, five seeds.
  - `4-V7`: bounded/residual last-block FiLM.
- 4-V0 runner decision:
  - Added `notebooks/kaggle_stage4_v2_p1_visual_only_same_split_one_cell.md`.
  - It intentionally uses the Stage 2 visual-only runner because the first v2
    control has no context branch.
  - It writes result summaries under `/kaggle/working/stage4_v2_visual_only_reports`.
- 4-V1 runner decision:
  - Added `notebooks/kaggle_stage4_v2_p2_ohlc_visual_only_one_cell.md`.
  - It also uses the Stage 2 visual-only runner because this is still a
    no-context control.
  - It removes MA/VB from the image to measure the plain-OHLC visual baseline
    before re-adding structured context in `4-V2`.
- 4-V2 runner decision:
  - Added `notebooks/kaggle_stage4_v2_p3_ohlc_all_context_film_full_one_cell.md`.
  - It uses the Stage 4 context runner with `IMAGE_SPEC=ohlc` and
    `CONTEXT_METHODS=["film_full"]`.
  - It is judged primarily against `4-V1`, because the image no longer contains
    MA/VB visual overlays.
- 4-V3 runner decision:
  - Added `notebooks/kaggle_stage4_v2_p4_ohlc_fg_only_film_full_five_seed_one_cell.md`.
  - It patches `context.primary_features` to the four F&G features and
    `stage4_model.context_dim` to `4`.
  - It uses `feature_set_name=fg_only` and `experiment_suffix=fg_only` to avoid
    overwriting all-context outputs.
  - From this point, Stage 4 v2 diagnostic runs default to five seeds.
- 4-V4 runner decision:
  - Added `notebooks/kaggle_stage4_v2_p5_ohlc_technical_only_film_full_five_seed_one_cell.md`.
  - It patches `context.primary_features` to
    `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, and `rv_60`.
  - It uses `feature_set_name=technical_only` and
    `experiment_suffix=technical_only` to keep outputs separate from all-context
    and F&G-only runs.
- 4-V5/4-V6 sequencing decision:
  - Before changing FiLM architecture, finish two remaining context-signal
    checks under the same FiLM-full structure.
  - `4-V5` repeats all-context as five seeds to decide whether the earlier
    seed-42 improvement was robust.
  - Added `notebooks/kaggle_stage4_v2_p6_ohlc_all_context_film_full_five_seed_one_cell.md`.
  - `4-V5` uses `feature_set_name=all_context` and
    `experiment_suffix=all_context` so it does not reuse or overwrite the
    earlier seed-42 `4-V2` output.
  - `4-V6` puts F&G-only on top of `ohlc_ma_vb` to test whether external
    sentiment/regime context adds signal to the strongest visual baseline.
  - Added `notebooks/kaggle_stage4_v2_p7_ohlc_ma_vb_fg_only_film_full_five_seed_one_cell.md`.
  - `4-V6` uses the same F&G-only feature suffix as `4-V3`, but the output names
    remain separated because `image_spec=ohlc_ma_vb`.
  - `4-V7` is now prepared as the bounded/residual last-block FiLM test:
    `notebooks/kaggle_stage4_v2_p8_ohlc_ma_vb_fg_only_bounded_last_block_film_five_seed_one_cell.md`.
  - It keeps `I60/R20/ohlc_ma_vb` and F&G-only context fixed, then changes only
    the FiLM architecture to `film_full_bounded_last_block`.

## 한국어

| Source | Local/reference path | Stage 4 사용 위치 |
| --- | --- | --- |
| FiLM 논문 요약 | `자료조사/FiLM요약.md` | FiLM 수식, gamma/beta 해석, BN-FiLM 위치 |
| FiLM 논문 PDF | `자료조사/FiLM.pdf` | 최종 코드 주석 전 정확한 paper citation 확인 |
| 교수님 방향성 note | `/Users/jaehyeonjeong/Desktop/film_chart_research_summary.md` | Stage 4 연구 framing: 고정 chart-image baseline, structured context, MLP encoder, concat/gating/FiLM 비교, 해석력 |
| FiLM reference repository | `https://github.com/ethanjperez/film`, commit `fe43ddf8a22b339dcca2efa07091ce9d498955cf` | FiLM generator/layer convention 참고 |
| Stage 2 BTC model | `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py` | I5/I20/I60 base CNN block variant |
| Stage 2 BTC pipeline | `stage2_btc_extension/src/stage2_btc/` | 고정된 BTC data, label, split, normalization, evaluation |
| Stage 3 Linear model | `stage3_linear_adapter/src/stage3_linear/models/linear_stock_cnn.py` | 비교 모델로만 사용 |
| Grad-CAM local summary | `자료조사/Grad-CAM요약.md` | Stage 4 Grad-CAM 규칙과 해석 |
| Fear & Greed 후보 dataset | `https://www.kaggle.com/datasets/ashishpatel8736/historical-and-fear-greed-index-datasets` | 외부 sentiment/regime context 후보. crypto-vs-equity 의미, date coverage, scale, missing date audit 필요 |
| BTC news 후보 dataset | `https://huggingface.co/datasets/edaschau/bitcoin_news` | 4-3 audit source. strict `t-1` headline-only first policy로 second-phase news context 사용 가능 |

구현 근거 구분:
- Paper/reference reported: FiLM은 condition에서 생성한 gamma와 beta로 feature-wise
  affine modulation을 적용합니다.
- Paper/reference reported: FiLM implementation note에서 BatchNorm 뒤 FiLM 적용은
  중요한 설정입니다.
- 이 논문 구현 선택: Stock_CNN block에서는
  `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`로 삽입합니다.
- 교수님 방향성 mapping:
  - 교수님 note는 이 task를 full VQA로 보지 말라고 정리합니다. 금융 예측 질문은
    사실상 고정되어 있습니다.
  - original FiLM의 question encoder 대신 structured metadata를 condition으로
    쓸 수 있다고 정리합니다.
  - CNN-only, naive condition concatenation, FiLM, optional attention-based
    fusion 비교를 권장합니다.
  - 따라서 Stage 4는 numeric context + MLP encoder를 사용하고 concat, gating,
    gamma-only FiLM, full FiLM을 비교합니다.
- 네 가지 model set을 결정한 교수님 note 원문 발췌:
  - "strong no-conditioning baseline"
  - "the prediction query is effectively fixed"
  - "compact structured metadata"
  - "an MLP or embedding-based condition encoder is a cleaner design than an RNN"
  - "CNN + naive condition concatenation"
  - "CNN + FiLM"
  - "Optional attention-based fusion"
- 첫 Stage 4 main run 구현 선택: structured numeric context를 먼저 사용합니다.
  news context는 audit 이후 second-phase track으로 유지합니다.
- 4-4 baseline dependency 결정:
  - Stage 4 primary image/model baseline은 Stage 2 `I60/R20/ohlc_ma_vb`입니다.
  - Primary comparison target은 Stage 2 selected five-seed mean입니다:
    accuracy `0.579320`, ROC-AUC `0.584862`.
  - Stage 3 Linear는 negative/simple-parameter ablation일 뿐이며 Stage 4
    architecture dependency가 아닙니다.
- 4-3 news-context 결정:
  - `edaschau/bitcoin_news`는 Stage 2 BTC 기간과 겹치고 timestamp/title/article/source
    field가 있으므로 사용 가능합니다.
  - 첫 news version은 headline-only, strict `t-1`, train-fit non-LLM encoder입니다.
  - Full article summary와 LLM embedding은 leakage-safe headline context가 안정화된
    뒤로 미룹니다.
- Structured context source 구분:
  - F&G는 외부 dataset이 필요합니다.
  - Bollinger %B, Bollinger bandwidth, MFI, realized volatility는 BTC OHLCV에서
    파생하므로 추가 dataset이 필요 없습니다.
- 4-5 context encoder와 normalization 결정:
  - 첫 run의 primary model input은 8개 feature입니다:
    `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`,
    `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60`.
  - Preprocessing은 feature-specific transform 이후 train-only median imputation,
    train-only 1/99% clipping, train-only z-score normalization을 사용합니다.
  - Shared encoder는
    `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`입니다.
- 4-6 insertion 결정:
  - Concat은 I60 flatten 뒤 `(B, 32)` context embedding을 붙입니다:
    `(B, 184320) -> (B, 184352)`.
  - Gating은 final block feature map `(B, 512, 2, 180)`에만 channel gate를
    적용합니다.
  - Gamma-only와 full FiLM은 모든 I60 block에서 BatchNorm 뒤, LeakyReLU 전에
    삽입합니다.
  - Gate/FiLM output head는 zero-initialize해서 gate/gamma/beta가 identity
    modulation에서 시작하게 합니다.
- 4-7 Grad-CAM/export 결정:
  - Primary Grad-CAM target은 predicted-class pre-softmax logit입니다.
  - 최종 보고 sample은 test split에서 Predicted Up 10개, Predicted Down 10개입니다.
  - 4-B/4-C/4-D는 선택된 Grad-CAM sample 옆에 context와 gate/gamma/beta 값을
    같이 export합니다.
  - FiLM heatmap은 post-FiLM conditioned feature map을 primary target layer로
    사용합니다.
- 4-8 Kaggle runner/output backup 결정:
  - 첫 Stage 4 Kaggle runner는 structured numeric context 기반 네 ablation
    `concat`, `gating`, `film_gamma`, `film_full`을 실행합니다.
  - 첫 full sanity run은 seed `42`, 이후 robustness run은 seed
    `42, 43, 44, 45, 46`으로 실행합니다.
  - Backup root는 `/kaggle/working/stage4_saved_outputs`입니다.
  - 완료 판정은 output check 통과 기준이며, checkpoint 존재만으로 완료 처리하지
    않습니다.
  - Checkpoint는 있지만 prediction/metric/Grad-CAM export가 없으면 완료가 아니라
    evaluation/export를 이어서 실행합니다.
- 4-I0 implementation readiness 결정:
  - Stage 4는 implementation으로 진행할 수 있습니다.
  - Stage 4는 configurable Stage 2 `src` dependency를 통해 Stage 2 BTC loading,
    sample/image generation, split/normalization, prediction metric, trading
    metric, Grad-CAM helper code를 재사용합니다.
  - 로컬 BTC OHLCV data가 있습니다.
  - 로컬 F&G data도 `stage4_film_conditioning/FG_data/fear_greed_index.csv`에
    추가됐습니다. Raw F&G CSV 파일은 GitHub에 track하지 않습니다.
- 4-I1 scaffold 결정:
  - `configs/env_local.yaml`, `configs/env_kaggle.yaml`을 추가했습니다.
  - Stage 4 config/path/runtime/seed helper를 추가했습니다.
  - Stage 4 `src`와 Stage 2 `src`를 함께 노출하는 shared script path utility를
    추가했습니다.
  - `scripts/check_stage4_scaffold.py`를 추가했고, local scaffold check에서 BTC,
    F&G, Stage 2 dependency path가 모두 통과했습니다.
- 4-I2 structured context feature builder 결정:
  - `src/stage4_film/context/sources.py`,
    `src/stage4_film/context/features.py`,
    `src/stage4_film/context/normalization.py`를 추가했습니다.
  - `scripts/audit_stage4_context_sources.py`,
    `scripts/build_stage4_context_features.py`를 추가했습니다.
  - Local I60/R20/ohlc_ma_vb context build에서 2,399 row가 생성됐습니다:
    train 671, validation 287, test 1,441.
  - Primary sample set에서 F&G as-of coverage는 완전합니다. F&G dataset에
    2024-10-26 missing date가 있어 max as-of age는 1일입니다.
  - Matched-window sample restriction 이후 8개 primary context feature는
    train/validation/test 모두 missing rate 0.0입니다.
  - Primary sample timing detail:
    - `I60/R20/ohlc_ma_vb`는 60일 image와 image 안 모든 날짜의 유효한 MA60이
      필요합니다.
    - BTC OHLCV가 2018-01-01부터 시작하므로 첫 valid sample end date는
      2018-04-29입니다.
    - 정확한 offset은 118일입니다. 두 window가 모두 inclusive라서
      `(60 - 1) + (60 - 1) = 118`입니다.
    - F&G는 2018-02-01부터 시작하므로 valid primary sample을 제거하지 않습니다.
    - Primary sample end date와 직접 겹치는 F&G 원본 missing date는
      2024-10-26 하루뿐이고, previous-available-value as-of merge로 처리합니다.
- 4-I3 context MLP encoder 결정:
  - `src/stage4_film/conditions/context_encoder.py`를 추가했습니다.
  - Shared encoder 구조:
    `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
  - Parameter count는 1,344입니다.
  - `scripts/check_stage4_context_encoder.py`가 dummy tensor와 local `4-I2`
    context table의 실제 normalized row 모두에서 통과했습니다.
- 4-I4 context concat model 결정:
  - `src/stage4_film/models/context_stock_cnn.py`를 추가했습니다.
  - `scripts/check_stage4_model_shapes.py`를 추가했습니다.
  - Stage 2 I60 Stock_CNN convolution block은 그대로 재사용하고, 마지막
    classifier만 교체합니다.
  - Primary tensor path:
    image `(B, 1, 96, 180) -> flatten (B, 184320)`;
    context `(B, 8) -> embedding (B, 32)`;
    concat `(B, 184352) -> logits (B, 2)`.
  - Parameter count는 `2,954,370`이며 Stage 2 I60 baseline 대비 `+1,408`입니다.
  - Local model shape checker가 dummy tensor와 local `4-I2` context table의
    실제 normalized context row 모두에서 통과했습니다.
- 4-I5 context gating model 결정:
  - `src/stage4_film/models/context_stock_cnn.py`에 `GatedContextStockCNN`을
    추가했습니다.
  - `scripts/check_stage4_model_shapes.py`에 `--model gating`을 추가했습니다.
  - Stage 2 I60 Stock_CNN convolution block은 그대로 재사용합니다.
  - Primary tensor path:
    image `(B, 1, 96, 180) -> final feature map (B, 512, 2, 180)`;
    context `(B, 8) -> embedding (B, 32) -> raw gate (B, 512)`;
    `gate = 2 * sigmoid(raw_gate)`;
    gated feature `(B, 512, 2, 180) -> flatten (B, 184320) -> logits (B, 2)`.
  - Gate head는 zero-initialized라서 initial gate min/max는 `1.0 / 1.0`입니다.
  - Parameter count는 `2,971,202`이며 Stage 2 I60 baseline 대비 `+18,240`입니다.
  - Local model shape checker가 dummy tensor와 local `4-I2` context table의
    실제 normalized context row 모두에서 통과했습니다.
- 4-I6 FiLM layer/generator 결정:
  - `src/stage4_film/layers/film.py`를 추가했습니다.
  - `src/stage4_film/conditions/film_generator.py`를 추가했습니다.
  - `scripts/check_stage4_film_layers.py`를 추가했습니다.
  - FiLM은 feature map에 channel-wise modulation을 적용합니다:
    gamma-only `F' = gamma * F`; full FiLM `F' = gamma * F + beta`.
  - Generator는 shared context embedding `(B, 32)`을 받아 I60 channel
    `[64, 128, 256, 512]`에 맞는 block별 gamma/beta parameter를 만듭니다.
  - Gamma는 `1 + delta_gamma`로 구현했고 head는 zero-initialized입니다.
    Full FiLM beta도 zero-initialized입니다.
  - 그래서 initialization에서 identity가 됩니다: gamma min/max `1.0 / 1.0`,
    beta min/max `0.0 / 0.0`, modulated feature map은 원래 feature map과 같습니다.
  - Generator parameter count:
    - gamma-only: `31,680`
    - full gamma/beta: `63,360`
  - Local FiLM checker가 dummy I60 feature map과 local `4-I2` context table의
    실제 normalized context row 모두에서 통과했습니다.
- 4-I7 FiLM Stock_CNN model 결정:
  - `src/stage4_film/models/film_stock_cnn.py`를 추가했습니다.
  - `scripts/check_stage4_model_shapes.py`에 `--model film_gamma`,
    `--model film_full`을 추가했습니다.
  - Stage 2 I60 Stock_CNN convolution block을 재사용하되, 각 block을
    `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d` 순서로 실행합니다.
  - `film_gamma` tensor path:
    context `(B, 8) -> embedding (B, 32) -> channel [64, 128, 256, 512]용
    gamma; final flatten `(B, 184320) -> logits (B, 2)`.
  - `film_full`은 같은 path에서 block별 beta도 함께 만듭니다.
  - Parameter count:
    - `film_gamma`: `2,985,986`, Stage 2 I60 baseline 대비 `+33,024`.
    - `film_full`: `3,017,666`, Stage 2 I60 baseline 대비 `+64,704`.
  - 두 local model check가 dummy tensor와 local `4-I2` context table의 실제
    normalized context row 모두에서 통과했습니다.
  - In-place `LeakyReLU`가 post-FiLM tensor를 바꿀 수 있으므로, model은
    해석 export용 pre-FiLM/post-FiLM feature map을 activation 전에 저장합니다.
- 4-I8 Stage 4 context runner 결정:
  - `src/stage4_film/runners/context_experiment.py`를 추가했습니다.
  - `src/stage4_film/training/loop.py`를 추가했습니다.
  - `scripts/run_stage4_context_model.py`를 추가했습니다.
  - Runner는 Stage 2 BTC data loading, sample 생성, chart image 생성, split,
    train-only pixel normalization을 그대로 재사용합니다.
  - Stage 4 context feature는 같은 split sample universe에서 만들고, train-only
    context 통계로 normalization한 뒤 `batch["context"]`로 붙입니다.
  - 학습은 `model(image, context)`를 호출합니다.
  - Stage 2 일반 weight initialization 뒤 gate/FiLM output head를 identity로 다시
    reset해서 modulation이 Stage 2 visual path에서 시작하게 합니다.
  - `concat`, `film_gamma` local smoke training을 통과했습니다.
- 4-I9 prediction/trading export 결정:
  - `src/stage4_film/evaluation/prediction.py`를 추가했습니다.
  - `scripts/evaluate_stage4_predictions.py`를 추가했습니다.
  - `scripts/evaluate_stage4_trading.py`를 추가했습니다.
  - Classification metric은 Stage 2 `compute_classification_metrics`를
    재사용합니다.
  - Trading metric은 Stage 2 `compute_trading_metrics`를 재사용합니다.
  - Prediction helper는 inference를 `model(image)`에서 `model(image, context)`로
    바꾸고 prediction CSV에 normalized context column을 같이 붙입니다.
  - `concat`, `film_gamma` smoke checkpoint에서 local export check를 통과했습니다.
- 4-I10 Grad-CAM/context/modulation export 결정:
  - `src/stage4_film/interpretability/gradcam_context.py`를 추가했습니다.
  - `scripts/generate_stage4_gradcam_context.py`를 추가했습니다.
  - Grad-CAM target은 계속 predicted-class pre-softmax logit입니다.
  - Stage 4 Grad-CAM은 `model(image)`가 아니라 `model(image, context)`를
    호출합니다.
  - Concat은 context 값과 context embedding summary를 export합니다.
  - Gating은 context 값, raw gate, final gate 값을 export합니다.
  - FiLM은 context 값과 block-wise gamma/beta 값을 export합니다.
  - FiLM Grad-CAM은 post-FiLM module을 target으로 쓰고, PyTorch
    full-backward-hook과 inplace LeakyReLU 충돌을 피하기 위해 tensor-level
    gradient hook을 사용합니다.
  - `concat`, `film_gamma` smoke checkpoint에서 local Grad-CAM export check를
    통과했습니다.
- 4-I11 smoke output check 결정:
  - `scripts/check_stage4_outputs.py`를 추가했습니다.
  - Stage 4 run 완료 기준은 checkpoint, train metadata, prediction,
    classification metric, trading metric, Grad-CAM, selected samples,
    modulation export, context artifact, run manifest가 모두 존재하고
    parse 가능한 상태입니다.
  - 단순 file existence뿐 아니라 CSV row count와 JSON parse 가능 여부도
    확인합니다.
  - `concat`, `film_gamma` smoke run에서 local output check를 통과했습니다.
- 4-I12 Kaggle four-ablation run 결정/결과:
  - `notebooks/kaggle_stage4_four_ablation_single_seed_one_cell.md`를 추가했습니다.
  - 첫 실제 Kaggle run은 `I60/R20/ohlc_ma_vb`, `context_window=60`,
    seed `42`, methods `concat`, `gating`, `film_gamma`, `film_full`로
    고정합니다.
  - Cell은 Stage 4/Stage 2 code snapshot을 복사하고,
    `configs/env_kaggle.yaml`을 patch한 뒤 BTC/F&G source audit, context
    feature build, method별 training/evaluation/trading/Grad-CAM/output-check를
    실행하고 `/kaggle/working/stage4_saved_outputs`에 backup zip을 저장합니다.
  - Kaggle seed-42 결과는 완료되었습니다. `film_full`이 Stage 4 방법 중
    최고였고 accuracy `0.584316`, ROC-AUC `0.596811`입니다.
  - Result table:
    `reports/tables/stage4_four_ablation_seed42_run_summary.csv`.
- 4-I13 Kaggle five-seed runner 결정:
  - `notebooks/kaggle_stage4_four_ablation_five_seed_one_cell.md`를 추가했습니다.
  - Robustness run은 같은 selected configuration을 고정하고 seeds
    `42, 43, 44, 45, 46`에 대해 네 context method를 실행합니다.
  - 두 Kaggle cell은 execution wrapper입니다. model 정의, context feature 정의,
    split rule, evaluation metric은 바꾸지 않습니다.
- Stage 4 v1 five-seed 진단:
  - five-seed run에서 `film_full`이 Stage 4 v1 best method였지만, mean
    accuracy `0.5510`, ROC-AUC `0.5677`은 선택된 Stage 2 visual baseline보다
    낮습니다.
  - Seed `42`는 promising했지만 seed `45`는 all-Down prediction으로
    collapse했습니다.
  - 다음 단계는 FiLM을 바로 복잡하게 만드는 것이 아니라 원인을 분리하는
    diagnostic입니다.
- Stage 4 v2 priority map:
  - `4-V0`: `I60/R20/ohlc_ma_vb`, visual-only, context 없음.
  - `4-V1`: `I60/R20/ohlc`, visual-only, context 없음.
  - `4-V2`: `I60/R20/ohlc` + all structured context + `film_full`.
  - `4-V3`: `I60/R20/ohlc` + F&G-only + `film_full`.
  - `4-V4`: `I60/R20/ohlc` + technical-only context + `film_full`.
  - `4-V5`: `I60/R20/ohlc` + all structured context + `film_full`, five
    seeds.
  - `4-V6`: `I60/R20/ohlc_ma_vb` + F&G-only + `film_full`, five seeds.
  - `4-V7`: bounded/residual last-block FiLM.
- 4-V0 runner 결정:
  - `notebooks/kaggle_stage4_v2_p1_visual_only_same_split_one_cell.md`를
    추가했습니다.
  - 첫 v2 control은 context branch가 없으므로 Stage 2 visual-only runner를
    의도적으로 사용합니다.
  - 결과 summary는 `/kaggle/working/stage4_v2_visual_only_reports`에 저장합니다.
- 4-V1 runner 결정:
  - `notebooks/kaggle_stage4_v2_p2_ohlc_visual_only_one_cell.md`를 추가했습니다.
  - 이 실험도 context가 없는 control이므로 Stage 2 visual-only runner를
    사용합니다.
  - `4-V2`에서 structured context를 다시 넣기 전에, 이미지에서 MA/VB를 제거한
    plain-OHLC visual baseline을 확인합니다.
- 4-V2 runner 결정:
  - `notebooks/kaggle_stage4_v2_p3_ohlc_all_context_film_full_one_cell.md`를
    추가했습니다.
  - Stage 4 context runner를 사용하며 `IMAGE_SPEC=ohlc`,
    `CONTEXT_METHODS=["film_full"]`로 고정합니다.
  - 이미지는 더 이상 MA/VB visual overlay를 포함하지 않으므로, 주 비교 대상은
    `4-V1`입니다.
- 4-V3 runner 결정:
  - `notebooks/kaggle_stage4_v2_p4_ohlc_fg_only_film_full_five_seed_one_cell.md`를
    추가했습니다.
  - `context.primary_features`를 F&G feature 4개로, `stage4_model.context_dim`을
    `4`로 patch합니다.
  - all-context output을 덮어쓰지 않도록 `feature_set_name=fg_only`,
    `experiment_suffix=fg_only`를 사용합니다.
  - 이 시점부터 Stage 4 v2 diagnostic run은 기본 five-seed입니다.
- 4-V4 runner 결정:
  - `notebooks/kaggle_stage4_v2_p5_ohlc_technical_only_film_full_five_seed_one_cell.md`를
    추가했습니다.
  - `context.primary_features`를 `bb_percent_b_60`, `bb_bandwidth_60`,
    `mfi_60`, `rv_60`으로 patch합니다.
  - all-context와 F&G-only output을 덮어쓰지 않도록
    `feature_set_name=technical_only`, `experiment_suffix=technical_only`를
    사용합니다.
- 4-V5/4-V6 순서 결정:
  - FiLM architecture를 바꾸기 전에, 같은 FiLM-full 구조에서 남은 context-signal
    확인을 먼저 끝냅니다.
  - `4-V5`는 이전 seed-42 all-context 개선이 robust한지 확인하기 위해
    all-context를 five-seed로 반복합니다.
  - `notebooks/kaggle_stage4_v2_p6_ohlc_all_context_film_full_five_seed_one_cell.md`를
    추가했습니다.
  - `4-V5`는 예전 seed-42 `4-V2` output을 재사용하거나 덮어쓰지 않도록
    `feature_set_name=all_context`, `experiment_suffix=all_context`를 사용합니다.
  - `4-V6`는 가장 강한 `ohlc_ma_vb` visual baseline 위에 F&G-only를 얹어,
    외부 sentiment/regime context가 incremental signal을 주는지 확인합니다.
  - `notebooks/kaggle_stage4_v2_p7_ohlc_ma_vb_fg_only_film_full_five_seed_one_cell.md`를
    추가했습니다.
  - `4-V6`는 `4-V3`와 같은 F&G-only suffix를 사용하지만,
    `image_spec=ohlc_ma_vb`가 output 이름에 포함되므로 결과는 분리됩니다.
  - `4-V7`은 bounded/residual last-block FiLM 실험으로 준비했습니다:
    `notebooks/kaggle_stage4_v2_p8_ohlc_ma_vb_fg_only_bounded_last_block_film_five_seed_one_cell.md`.
  - `I60/R20/ohlc_ma_vb`와 F&G-only context는 고정하고, FiLM architecture만
    `film_full_bounded_last_block`으로 바꿉니다.
