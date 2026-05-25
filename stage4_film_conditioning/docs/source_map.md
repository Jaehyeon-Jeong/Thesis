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
    `stage4_film_conditioning/F&G_data/fear_greed_index.csv`; raw F&G CSV files
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
  - Next implementation item is `4-I6`, FiLM layer and FiLM generator modules.

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
  - 로컬 F&G data도 `stage4_film_conditioning/F&G_data/fear_greed_index.csv`에
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
  - 다음 구현 항목은 `4-I6`, FiLM layer와 FiLM generator module입니다.
