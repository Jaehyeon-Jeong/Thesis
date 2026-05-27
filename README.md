# Thesis

## English

This repository organizes the thesis experiment pipeline for reproducing and
extending the image-based CNN framework from *Re-Imag(in)ing Price Trends*.

The current public GitHub contents are documentation, reproducibility metadata,
small audit outputs, diagrams, configs, and code scaffolds. Large data files,
paper PDFs, checkpoints, and large prediction outputs are intentionally not
tracked.

### Pipeline

| Stage | Purpose | Current status |
| --- | --- | --- |
| `stage0_data_check` | Audit data, papers, and reference implementations | Completed |
| `stage1_reimage_reproduction` | Reproduce the Re-image CNN pipeline on public I20 stock images | In progress: `I20/R60` seed-42 fast diagnostic archived; `I20/R20` archive is smoke-only; `I20/R5`, strict batch-128 run, and five-seed reproduction are later |
| `stage2_btc_extension` | Extend the confirmed pipeline to BTC OHLCV | Single-seed 36-run complete; selected `I20/R20` and `I60/R20` five-seed robustness check complete; full 180-run five-seed grid later |
| `stage3_linear_adapter` | Add a Linear comparison model | First test on Stage 2 best config completed; result dropped to majority level; remaining grid runs pending |
| `stage4_film_conditioning` | Compare market-context concat, gating, gamma-only FiLM, and full FiLM on the fixed BTC CNN | v1 four-ablation five-seed run complete; v2 priorities 1-2 reproduced Stage 2 controls; priority 3 seed-42 showed partial recovery; priority 4 F&G-only five-seed runner ready |

### Current Status

Stage 1:
- Current usable full test artifact: `I20/R60`, seed `42`, fast Kaggle
  diagnostic.
- `I20/R60` snapshot: accuracy `0.5312`, majority accuracy `0.5408`,
  ROC-AUC `0.5298`, test rows `1,376,215`.
- `I20/R20` is not ready as a full result in the local archive. The preserved
  metrics/Grad-CAM are validation-smoke outputs only.
- `I20/R5` is not archived locally yet.
- Later: strict paper-style batch size `128`, five independent runs/seeds, and
  final `10` up + `10` down Figure-13-style Grad-CAM.

Stage 2:
- Current result package: BTC single-seed grid, `36` experiments
  (`I5/I20/I60` x `R5/R20/R60` x four image specs), seed `42`.
- Best single-seed configuration: `I60/R20/ohlc_ma_vb`.
- Selected five-seed robustness check completed for `I20/R20` and `I60/R20`
  across four image specs and seeds `42, 43, 44, 45, 46` (`40/40` runs ok).
- Best selected five-seed configuration: `I60/R20/ohlc_ma_vb`, accuracy mean
  `0.5793`, accuracy std `0.0182`, majority accuracy `0.5413`, ROC-AUC mean
  `0.5849`.
- Interpretation: `I60/R20` survives seed variation; `I20/R20` does not beat
  the majority baseline on average.
- Remaining Stage 2 work is the full `180`-run five-seed grid if a final global
  Stage 2 stability claim is needed.

Stage 3:
- Stage 2 data/image/split/normalization/evaluation/Grad-CAM pipeline remains
  fixed.
- First Linear comparison uses a bias-free adapter/head with `adapter_dim=128`.
- Naive `Linear(feature_dim, feature_dim)` is explicitly rejected because it is
  infeasible for `I60`.
- Implemented Kaggle runner: one full run and single-seed `36`-run grid.
- Preliminary completed run: `I60/R20/ohlc_ma_vb`, seed `42`, adapter dim `128`.
  This was the best Stage 2 single-seed configuration.
- Stage 2 baseline for this configuration: accuracy `0.603053`, majority
  accuracy `0.541291`, ROC-AUC `0.616950`.
- Stage 3 Linear for the same configuration: accuracy `0.541291`, majority
  accuracy `0.541291`, ROC-AUC `0.522101`.
- Interpretation so far: Linear did not improve the best Stage 2 model in this
  first diagnostic; it dropped to majority-class-level accuracy.
- Remaining Stage 3 single-seed grid configurations are pending.
- Local smoke test passed for `I5/R5/ohlc`, seed `42`, one epoch, tiny rows.
- Later: Stage 3 result report after Kaggle outputs and five-seed stability
  checks.

Stage 4:
- Stage 4 is now defined as a market-context fusion/modulation comparison on
  the fixed Stage 2 BTC baseline `I60/R20/ohlc_ma_vb`.
- Main ablations: `4-A CNN + context concat`, `4-B CNN + context gating`,
  `4-C CNN + context FiLM gamma-only`, and `4-D CNN + context FiLM full`.
- First context source: structured numeric market context, including F&G,
  Bollinger %B, Bollinger bandwidth, MFI, and realized volatility.
- 4-5 fixed the first model input as 8 matched-window features:
  `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`,
  `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, and `rv_60`.
- Context preprocessing is train-only: feature transform, median imputation,
  1/99% clipping, and z-score normalization are fit on train only.
- Shared context encoder:
  `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
- Primary sample timing note:
  - `I60/R20/ohlc_ma_vb` requires both a 60-day image and valid MA60 values for
    every day inside the image.
  - BTC OHLCV starts on `2018-01-01`; the first valid primary sample end date
    is therefore `2018-04-29`.
  - The offset is `118` days because both windows are inclusive:
    `(60 - 1) + (60 - 1) = 118`.
  - F&G starts on `2018-02-01`, so the F&G start date does not remove any valid
    primary samples.
  - Only one raw F&G missing date directly overlaps a primary sample end date:
    `2024-10-26`; it is handled with previous-available-value as-of merge.
- 4-6 fixed the insertion design:
  - concat attaches after I60 flatten: `(B, 184320) + (B, 32) -> (B, 184352)`;
  - gating applies a final-block channel gate to `(B, 512, 2, 180)`;
  - FiLM is inserted after BatchNorm and before LeakyReLU in every I60 block.
- 4-7 fixed the explanation/export policy:
  - Grad-CAM target is the predicted-class pre-softmax logit;
  - final figures use 10 Predicted Up and 10 Predicted Down test samples;
  - context and gate/gamma/beta values are exported beside selected Grad-CAM
    samples.
- 4-8 fixed the Kaggle runner/output backup contract:
  - first full sanity run is the four numeric-context ablations with seed `42`;
  - later robustness run is the same four ablations with seeds
    `42, 43, 44, 45, 46`;
  - backup root is `/kaggle/working/stage4_saved_outputs`;
  - completion requires output-check success, not checkpoint existence alone.
- 4-I0 confirmed implementation readiness:
  - Stage 4 will reuse Stage 2 BTC data/image/split/evaluation helpers through
    a configurable Stage 2 `src` dependency;
  - local BTC OHLCV data exists, and local F&G data is now available at
    `stage4_film_conditioning/FG_data/fear_greed_index.csv`;
  - raw F&G CSV files are not tracked in GitHub. Kaggle runs should still
    attach the public F&G dataset for final reproducibility.
- 4-I1 completed the shared scaffold:
  - local/Kaggle configs;
  - Stage 4 config/path/runtime/seed helpers;
  - script path utility exposing Stage 4 `src` and Stage 2 `src`;
  - `check_stage4_scaffold.py`, which passed locally for BTC, F&G, and Stage 2
    dependency paths.
- 4-I2 completed the structured context feature builder:
  - F&G source audit;
  - OHLCV-derived Bollinger, MFI, and realized volatility features;
  - train-only context preprocessing and scaler export.
- Local I60/R20/ohlc_ma_vb context build produced `2,399` rows:
  train `671`, validation `287`, test `1,441`.
- Primary context feature missing-rate warnings: none.
- 4-I3 completed the shared context MLP encoder:
  - `(B, 8) -> (B, 32)`;
  - parameter count `1,344`;
  - local check passed on dummy tensors and real normalized rows from the 4-I2
    context table.
- 4-I4 completed the `CNN + context concat` model:
  - Stage 2 I60 Stock_CNN convolution blocks are reused unchanged;
  - final classifier is replaced with `Dropout(0.5) -> Linear(184352, 2)`;
  - tensor path is image `(B, 1, 96, 180) -> flatten (B, 184320)`, context
    `(B, 8) -> embedding (B, 32)`, concat `(B, 184352) -> logits (B, 2)`;
  - parameter count is `2,954,370`, only `+1,408` vs the Stage 2 I60 baseline;
  - local shape/parameter check passed on dummy tensors and real normalized
    context rows.
- 4-I5 completed the `CNN + context gating` model:
  - Stage 2 I60 Stock_CNN convolution blocks are reused unchanged;
  - context embedding `(B, 32)` generates a channel gate `(B, 512)`;
  - gate formula is `gate = 2 * sigmoid(raw_gate)`;
  - the gate is applied to the final feature map `(B, 512, 2, 180)`;
  - classifier input remains `(B, 184320)`;
  - gate head is zero-initialized, so initial gate min/max is `1.0 / 1.0`;
  - parameter count is `2,971,202`, `+18,240` vs the Stage 2 I60 baseline.
- 4-I6 completed the reusable FiLM layer/generator modules:
  - `FeatureWiseAffineModulation` applies channel-wise `F' = gamma * F` or
    `F' = gamma * F + beta`;
  - `FilmParameterGenerator` maps context embedding `(B, 32)` to block-wise
    gamma/beta for I60 channels `[64, 128, 256, 512]`;
  - gamma-only generator parameter count is `31,680`;
  - full gamma/beta generator parameter count is `63,360`;
  - identity initialization passed locally with gamma `1.0`, beta `0.0`, and
    unchanged feature maps.
- 4-I7 completed the gamma-only and full FiLM Stock_CNN models:
  - `FilmContextStockCNN` reuses the Stage 2 I60 Stock_CNN convolution blocks;
  - each block runs as `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`;
  - `film_gamma` parameter count is `2,985,986`, `+33,024` vs Stage 2 I60;
  - `film_full` parameter count is `3,017,666`, `+64,704` vs Stage 2 I60;
  - local checks passed for dummy tensors, real normalized context rows, and
    all-block identity initialization.
- 4-I8 completed the Stage 4 context training runner:
  - it reuses the fixed Stage 2 BTC data/image/split/pixel-normalization path;
  - it attaches normalized context tensors and trains with `model(image, context)`;
  - local smoke training passed for `concat` and `film_gamma`.
- 4-I9 completed prediction/classification/trading exports:
  - `evaluate_stage4_predictions.py` writes prediction CSV and classification metrics;
  - `evaluate_stage4_trading.py` writes BTC long/flat and long/short trading metrics;
  - local export checks passed for `concat` and `film_gamma` smoke checkpoints.
- 4-I10 completed Grad-CAM plus context/gate/gamma/beta exports:
  - `generate_stage4_gradcam_context.py` writes the Grad-CAM figure,
    `samples.csv`, `modulation_summary.csv`, and `modulation_values.json`;
  - concat exports context/context-embedding summaries;
  - gating exports raw gate and final gate values;
  - FiLM exports block-wise gamma/beta values;
  - local Grad-CAM export checks passed for `concat` and `film_gamma` smoke
    checkpoints.
- 4-I11 completed local smoke output checking:
  - `check_stage4_outputs.py` verifies checkpoint, training metadata,
    predictions, classification metrics, trading metrics, Grad-CAM, samples,
    modulation exports, context artifacts, and run manifest;
  - local checker passed for `concat` and `film_gamma` smoke runs;
  - compact summary is stored in
    `stage4_film_conditioning/reports/smoke_tests/stage4_smoke_summary.json`.
- 4-I12 Kaggle four-ablation seed-42 run is complete:
  - run: `I60/R20/ohlc_ma_vb`, context window `60`, seed `42`, methods
    `concat`, `gating`, `film_gamma`, `film_full`;
  - all four methods returned `status = ok` with 1,441 test predictions;
  - best Stage 4 method: `film_full`, accuracy `0.584316`, ROC-AUC `0.596811`;
  - this is promising versus the Stage 2 five-seed mean, but it is not yet
    better than the same Stage 2 seed-42 baseline run.
- 4-I13 Kaggle five-seed robustness run is complete for the same selected
  configuration and seeds `42, 43, 44, 45, 46`.
- Best Stage 4 v1 method: `film_full`, accuracy mean `0.5510`, ROC-AUC mean
  `0.5677`.
- The selected Stage 2 visual baseline remains stronger: accuracy mean
  `0.5793`, ROC-AUC mean `0.5849`.
- Diagnosis: `film_full` seed `42` was promising, but seed `45` collapsed to
  all-Down predictions. Stage 4 v1 is therefore not robust enough to claim an
  improvement.
- Stage 4 v2 priorities are now tracked in the Stage 4 checklist. Priority 1 is
  a visual-only same-split control:
  `I60/R20/ohlc_ma_vb`, no context.
- Priority 1 Kaggle runner is ready:
  `stage4_film_conditioning/notebooks/kaggle_stage4_v2_p1_visual_only_same_split_one_cell.md`.
- Priority 2 Kaggle runner is ready:
  `stage4_film_conditioning/notebooks/kaggle_stage4_v2_p2_ohlc_visual_only_one_cell.md`.
- Priority 3 Kaggle runner is ready:
  `stage4_film_conditioning/notebooks/kaggle_stage4_v2_p3_ohlc_all_context_film_full_one_cell.md`.
- Priority 4 five-seed Kaggle runner is ready:
  `stage4_film_conditioning/notebooks/kaggle_stage4_v2_p4_ohlc_fg_only_film_full_five_seed_one_cell.md`.
- Priority 5 five-seed Kaggle runner is ready:
  `stage4_film_conditioning/notebooks/kaggle_stage4_v2_p5_ohlc_technical_only_film_full_five_seed_one_cell.md`.
- Current Stage 4 v2 order after `4-V4`: run `4-V5` all-context five-seed,
  then `4-V6` `ohlc_ma_vb + F&G-only` five-seed, then bounded/last-block FiLM.
- News context is preserved as a second-phase track after source/date/leakage
  audit. Candidate source: Hugging Face `edaschau/bitcoin_news`.
- Advisor-direction mapping is documented in the Stage 4 README/source map and
  checklist result `4-1`.
- The planned FiLM insertion point is inside each Stock_CNN block:
  `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`.

### Key documents

- [PLAN.md](PLAN.md)
- [Overall pipeline diagram](docs/overall_pipeline_diagram.md)
- [Execution environment diagram](docs/execution_environment_diagram.md)
- [Professor Stage 4 decision report](reports/professor_stage4_decision_report_2026-05-21.md)
- [Stage 0 checklist](stage0_data_check/checklist.md)
- [Stage 1 checklist](stage1_reimage_reproduction/checklist.md)
- [Stage 2 checklist](stage2_btc_extension/checklist.md)
- [Stage 3 checklist](stage3_linear_adapter/checklist.md)
- [Stage 4 checklist](stage4_film_conditioning/checklist.md)

### Data policy

Tracked:
- Markdown plans and result reports
- Mermaid diagrams
- source maps
- configs
- small CSV summaries
- small sample figures

Not tracked:
- paper PDFs
- `.dat` image shards
- `.feather` source labels
- checkpoints
- large prediction CSVs
- old scratch/test code

## 한국어

이 저장소는 *Re-Imag(in)ing Price Trends*의 이미지 기반 CNN 파이프라인을
재현하고 BTC/Linear/FiLM으로 확장하기 위한 논문 실험 프로젝트입니다.

현재 GitHub에는 문서, 재현성 메타데이터, 작은 audit 산출물, 다이어그램,
config, 코드 scaffold만 올립니다. 대용량 데이터, 논문 PDF, checkpoint,
대용량 prediction output은 의도적으로 추적하지 않습니다.

### 파이프라인

| 단계 | 목적 | 현재 상태 |
| --- | --- | --- |
| `stage0_data_check` | 데이터, 논문, reference implementation 확인 | 완료 |
| `stage1_reimage_reproduction` | public I20 stock image로 Re-image CNN pipeline 재현 | 진행 중: `I20/R60` seed-42 fast diagnostic 보존; `I20/R20` archive는 smoke-only; `I20/R5`, strict batch-128 run, five-seed reproduction은 later |
| `stage2_btc_extension` | 확인된 pipeline을 BTC OHLCV로 확장 | single-seed 36-run 완료; `I20/R20`, `I60/R20` 선별 five-seed robustness check 완료; full 180-run five-seed grid는 later |
| `stage3_linear_adapter` | Linear 비교 모델 추가 | Stage 2 best config 1회 테스트 완료; majority 수준으로 하락; 나머지 grid run 예정 |
| `stage4_film_conditioning` | 고정 BTC CNN 위에서 market-context concat, gating, gamma-only FiLM, full FiLM 비교 | v1 four-ablation five-seed run 완료; v2 우선순위 1-2는 Stage 2 control 재현 완료; 우선순위 3 seed-42는 일부 회복 확인; 우선순위 4 F&G-only five-seed runner 준비 완료 |

### 현재 상태

Stage 1:
- 현재 full test artifact로 사용할 수 있는 것은 `I20/R60`, seed `42`, fast
  Kaggle diagnostic입니다.
- `I20/R60` snapshot: accuracy `0.5312`, majority accuracy `0.5408`,
  ROC-AUC `0.5298`, test rows `1,376,215`.
- `I20/R20`은 로컬 archive 기준 full 결과가 아닙니다. 현재 보존된
  metrics/Grad-CAM은 validation-smoke output입니다.
- `I20/R5`는 아직 로컬에 보존되어 있지 않습니다.
- Later: 논문식 strict batch size `128`, five independent runs/seeds, 최종
  `10` up + `10` down Figure-13-style Grad-CAM.

Stage 2:
- 현재 결과 패키지: BTC single-seed grid, `36`개 실험
  (`I5/I20/I60` x `R5/R20/R60` x image spec 4개), seed `42`.
- Single-seed best configuration: `I60/R20/ohlc_ma_vb`.
- `I20/R20`과 `I60/R20`을 대상으로 image spec 4개와 seed
  `42, 43, 44, 45, 46`을 돌린 선별 five-seed robustness check를 완료했습니다
  (`40/40` runs ok).
- 선별 five-seed best configuration: `I60/R20/ohlc_ma_vb`, accuracy mean
  `0.5793`, accuracy std `0.0182`, majority accuracy `0.5413`, ROC-AUC mean
  `0.5849`.
- 해석: `I60/R20` 우위는 seed 변화 후에도 유지되지만, `I20/R20`은 평균적으로
  majority baseline을 이기지 못했습니다.
- 남은 Stage 2 작업은 최종적인 전체 안정성 주장이 필요할 경우 full `180`-run
  five-seed grid를 수행하는 것입니다.

Stage 3:
- Stage 2 data/image/split/normalization/evaluation/Grad-CAM pipeline은
  고정합니다.
- 첫 Linear 비교는 `adapter_dim=128`의 bias-free adapter/head를 사용합니다.
- 단순 `Linear(feature_dim, feature_dim)`는 `I60`에서 계산상 불가능하므로
  명시적으로 제외했습니다.
- 구현된 Kaggle runner는 full run 1개와 single-seed `36`-run grid를 지원합니다.
- Preliminary 완료 run: `I60/R20/ohlc_ma_vb`, seed `42`, adapter dim `128`.
  이 조합은 Stage 2 single-seed best configuration입니다.
- 같은 조합의 Stage 2 baseline: accuracy `0.603053`, majority accuracy
  `0.541291`, ROC-AUC `0.616950`.
- 같은 조합의 Stage 3 Linear: accuracy `0.541291`, majority accuracy
  `0.541291`, ROC-AUC `0.522101`.
- 현재 해석: 첫 diagnostic 기준 Linear는 Stage 2 best model을 개선하지 못했고,
  majority-class-level accuracy로 하락했습니다.
- 나머지 Stage 3 single-seed grid configuration은 실행 예정입니다.
- Local smoke test는 `I5/R5/ohlc`, seed `42`, one epoch, tiny rows로 통과했습니다.
- Later: Kaggle output과 five-seed 안정성 확인 후 Stage 3 result report 작성.

Stage 4:
- Stage 4는 이제 고정된 Stage 2 BTC baseline `I60/R20/ohlc_ma_vb` 위에서
  market-context fusion/modulation을 비교하는 단계로 정리했습니다.
- Main ablation: `4-A CNN + context concat`, `4-B CNN + context gating`,
  `4-C CNN + context FiLM gamma-only`, `4-D CNN + context FiLM full`.
- 첫 context source는 structured numeric market context입니다: F&G, Bollinger %B,
  Bollinger bandwidth, MFI, realized volatility.
- 4-5에서 첫 model input을 matched-window 8개 feature로 고정했습니다:
  `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`,
  `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60`.
- Context preprocessing은 train-only입니다. Feature transform, median imputation,
  1/99% clipping, z-score normalization을 train split에서만 fit합니다.
- Shared context encoder:
  `Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU`.
- Primary sample timing note:
  - `I60/R20/ohlc_ma_vb`는 60일 image와 image 안 모든 날짜의 유효한 MA60이
    필요합니다.
  - BTC OHLCV는 `2018-01-01`부터 시작하므로 첫 primary sample end date는
    `2018-04-29`입니다.
  - Offset은 `118`일입니다. 두 window가 모두 inclusive라서
    `(60 - 1) + (60 - 1) = 118`입니다.
  - F&G는 `2018-02-01`부터 시작하므로 valid primary sample을 제거하지 않습니다.
  - Primary sample end date와 직접 겹치는 F&G 원본 missing date는
    `2024-10-26` 하루뿐이며, previous-available-value as-of merge로 처리합니다.
- 4-6에서 삽입 설계를 고정했습니다.
  - concat은 I60 flatten 뒤에 붙입니다: `(B, 184320) + (B, 32) -> (B, 184352)`.
  - gating은 final block feature map `(B, 512, 2, 180)`에 channel gate를 적용합니다.
  - FiLM은 모든 I60 block에서 BatchNorm 뒤, LeakyReLU 전에 삽입합니다.
- 4-7에서 explanation/export 정책을 고정했습니다.
  - Grad-CAM target은 predicted-class pre-softmax logit입니다.
  - 최종 figure는 test sample에서 Predicted Up 10개, Predicted Down 10개를 사용합니다.
  - 선택된 Grad-CAM sample 옆에 context와 gate/gamma/beta 값을 같이 export합니다.
- 4-8에서 Kaggle runner/output backup 계약을 고정했습니다.
  - 첫 full sanity run은 structured numeric context 네 ablation을 seed `42`로 실행합니다.
  - 이후 robustness run은 같은 네 ablation을 seed `42, 43, 44, 45, 46`으로 실행합니다.
  - Backup root는 `/kaggle/working/stage4_saved_outputs`입니다.
  - 완료 판정은 checkpoint 존재가 아니라 output check 통과 기준입니다.
- 4-I0에서 구현 준비를 확인했습니다.
  - Stage 4는 configurable Stage 2 `src` dependency를 통해 Stage 2 BTC
    data/image/split/evaluation helper를 재사용합니다.
  - 로컬 BTC OHLCV data가 있고, 로컬 F&G data도
    `stage4_film_conditioning/FG_data/fear_greed_index.csv`에 추가됐습니다.
  - Raw F&G CSV는 GitHub에 track하지 않습니다. 최종 재현성을 위해 Kaggle run에서는
    public F&G dataset을 계속 attach해야 합니다.
- 4-I1에서 공통 scaffold를 완료했습니다.
  - local/Kaggle config;
  - Stage 4 config/path/runtime/seed helper;
  - Stage 4 `src`와 Stage 2 `src`를 노출하는 script path utility;
  - `check_stage4_scaffold.py`를 추가했고 BTC, F&G, Stage 2 dependency path
    local check를 통과했습니다.
- 4-I2에서 structured context feature builder를 완료했습니다.
  - F&G source audit;
  - OHLCV-derived Bollinger, MFI, realized volatility feature;
  - train-only context preprocessing과 scaler export.
- Local I60/R20/ohlc_ma_vb context build에서 `2,399` row가 생성됐습니다:
  train `671`, validation `287`, test `1,441`.
- Primary context feature missing-rate warning은 없습니다.
- 4-I3에서 shared context MLP encoder를 완료했습니다.
  - `(B, 8) -> (B, 32)`;
  - parameter count `1,344`;
  - dummy tensor와 4-I2 context table의 실제 normalized row에서 local check를
    통과했습니다.
- 4-I4에서 `CNN + context concat` model을 완료했습니다.
  - Stage 2 I60 Stock_CNN convolution block은 그대로 재사용합니다.
  - final classifier를 `Dropout(0.5) -> Linear(184352, 2)`로 교체했습니다.
  - tensor path는 image `(B, 1, 96, 180) -> flatten (B, 184320)`, context
    `(B, 8) -> embedding (B, 32)`, concat `(B, 184352) -> logits (B, 2)`입니다.
  - parameter count는 `2,954,370`이며 Stage 2 I60 baseline 대비 `+1,408`입니다.
  - dummy tensor와 실제 normalized context row에서 local shape/parameter check를
    통과했습니다.
- 4-I5에서 `CNN + context gating` model을 완료했습니다.
  - Stage 2 I60 Stock_CNN convolution block은 그대로 재사용합니다.
  - context embedding `(B, 32)`이 channel gate `(B, 512)`를 만듭니다.
  - gate formula는 `gate = 2 * sigmoid(raw_gate)`입니다.
  - gate는 마지막 feature map `(B, 512, 2, 180)`에 적용됩니다.
  - classifier input은 `(B, 184320)` 그대로 유지됩니다.
  - gate head는 zero-initialized라서 initial gate min/max가 `1.0 / 1.0`입니다.
  - parameter count는 `2,971,202`이며 Stage 2 I60 baseline 대비 `+18,240`입니다.
- 4-I6에서 재사용 가능한 FiLM layer/generator module을 완료했습니다.
  - `FeatureWiseAffineModulation`은 channel-wise `F' = gamma * F` 또는
    `F' = gamma * F + beta`를 적용합니다.
  - `FilmParameterGenerator`는 context embedding `(B, 32)`을 I60 channel
    `[64, 128, 256, 512]`의 block별 gamma/beta로 바꿉니다.
  - Gamma-only generator parameter count는 `31,680`입니다.
  - Full gamma/beta generator parameter count는 `63,360`입니다.
  - Local identity initialization check에서 gamma `1.0`, beta `0.0`, feature
    map 보존을 확인했습니다.
- 4-I7에서 gamma-only/full FiLM Stock_CNN model을 완료했습니다.
  - `FilmContextStockCNN`은 Stage 2 I60 Stock_CNN convolution block을 재사용합니다.
  - 각 block은 `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`로
    실행됩니다.
  - `film_gamma` parameter count는 `2,985,986`, Stage 2 I60 대비 `+33,024`입니다.
  - `film_full` parameter count는 `3,017,666`, Stage 2 I60 대비 `+64,704`입니다.
  - Dummy tensor, 실제 normalized context row, all-block identity initialization
    local check를 통과했습니다.
- 4-I8에서 Stage 4 context training runner를 완료했습니다.
  - 고정된 Stage 2 BTC data/image/split/pixel-normalization 경로를 재사용합니다.
  - normalized context tensor를 붙이고 `model(image, context)`로 학습합니다.
  - `concat`, `film_gamma` local smoke training을 통과했습니다.
- 4-I9에서 prediction/classification/trading export를 완료했습니다.
  - `evaluate_stage4_predictions.py`가 prediction CSV와 classification metrics를 저장합니다.
  - `evaluate_stage4_trading.py`가 BTC long/flat, long/short trading metrics를 저장합니다.
  - `concat`, `film_gamma` smoke checkpoint에서 local export check를 통과했습니다.
- 4-I10에서 Grad-CAM plus context/gate/gamma/beta export를 완료했습니다.
  - `generate_stage4_gradcam_context.py`가 Grad-CAM figure, `samples.csv`,
    `modulation_summary.csv`, `modulation_values.json`를 저장합니다.
  - concat은 context/context-embedding summary를 export합니다.
  - gating은 raw gate와 final gate 값을 export합니다.
  - FiLM은 block-wise gamma/beta 값을 export합니다.
  - `concat`, `film_gamma` smoke checkpoint에서 local Grad-CAM export check를
    통과했습니다.
- 4-I11에서 local smoke output checking을 완료했습니다.
  - `check_stage4_outputs.py`는 checkpoint, training metadata, prediction,
    classification metric, trading metric, Grad-CAM, samples, modulation export,
    context artifact, run manifest를 확인합니다.
  - `concat`, `film_gamma` smoke run에서 local checker를 통과했습니다.
  - Compact summary는
    `stage4_film_conditioning/reports/smoke_tests/stage4_smoke_summary.json`에
    저장했습니다.
- 4-I12 Kaggle four-ablation seed-42 run을 완료했습니다.
  - 실행: `I60/R20/ohlc_ma_vb`, context window `60`, seed `42`, methods
    `concat`, `gating`, `film_gamma`, `film_full`.
  - 네 방법 모두 `status = ok`이며 test prediction 1,441개를 생성했습니다.
  - Stage 4 최고 방법: `film_full`, accuracy `0.584316`, ROC-AUC `0.596811`.
  - Stage 2 five-seed mean과 비교하면 promising하지만, 같은 Stage 2 seed-42
    baseline run보다 높지는 않아 robust improvement 주장은 보류합니다.
- 4-I13 Kaggle five-seed robustness run을 같은 selected configuration과
  seeds `42, 43, 44, 45, 46`으로 완료했습니다.
- Stage 4 v1 best method는 `film_full`이며 accuracy mean `0.5510`,
  ROC-AUC mean `0.5677`입니다.
- 선택된 Stage 2 visual baseline은 여전히 더 강합니다: accuracy mean
  `0.5793`, ROC-AUC mean `0.5849`.
- 진단: `film_full` seed `42`는 promising했지만, seed `45`는 all-Down
  prediction으로 collapse했습니다. 따라서 Stage 4 v1은 robust improvement로
  주장하기에는 부족합니다.
- Stage 4 v2 우선순위는 Stage 4 checklist에 추가했습니다. 우선순위 1은
  visual-only same-split control입니다: `I60/R20/ohlc_ma_vb`, context 없음.
- 우선순위 1 Kaggle runner 준비 완료:
  `stage4_film_conditioning/notebooks/kaggle_stage4_v2_p1_visual_only_same_split_one_cell.md`.
- 우선순위 2 Kaggle runner 준비 완료:
  `stage4_film_conditioning/notebooks/kaggle_stage4_v2_p2_ohlc_visual_only_one_cell.md`.
- 우선순위 3 Kaggle runner 준비 완료:
  `stage4_film_conditioning/notebooks/kaggle_stage4_v2_p3_ohlc_all_context_film_full_one_cell.md`.
- 우선순위 4 five-seed Kaggle runner 준비 완료:
  `stage4_film_conditioning/notebooks/kaggle_stage4_v2_p4_ohlc_fg_only_film_full_five_seed_one_cell.md`.
- 우선순위 5 five-seed Kaggle runner 준비 완료:
  `stage4_film_conditioning/notebooks/kaggle_stage4_v2_p5_ohlc_technical_only_film_full_five_seed_one_cell.md`.
- `4-V4` 이후 현재 Stage 4 v2 순서: `4-V5` all-context five-seed,
  `4-V6` `ohlc_ma_vb + F&G-only` five-seed, 그 다음 bounded/last-block FiLM.
- News context는 제거하지 않고 source/date/leakage audit 이후 second-phase track으로
  유지합니다. 후보 source는 Hugging Face `edaschau/bitcoin_news`입니다.
- 교수님 방향성 파일과 Stage 4 실험 결정의 연결은 Stage 4 README/source map과
  checklist result `4-1`에 문서화했습니다.
- 계획한 FiLM 삽입 위치는 각 Stock_CNN block 내부입니다:
  `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`.

### 주요 문서

- [PLAN.md](PLAN.md)
- [전체 파이프라인 다이어그램](docs/overall_pipeline_diagram.md)
- [실행환경 다이어그램](docs/execution_environment_diagram.md)
- [교수님 Stage 4 방향 확정 요청 보고서](reports/professor_stage4_decision_report_2026-05-21.md)
- [Stage 0 체크리스트](stage0_data_check/checklist.md)
- [Stage 1 체크리스트](stage1_reimage_reproduction/checklist.md)
- [Stage 2 체크리스트](stage2_btc_extension/checklist.md)
- [Stage 3 체크리스트](stage3_linear_adapter/checklist.md)
- [Stage 4 체크리스트](stage4_film_conditioning/checklist.md)

### 데이터 정책

GitHub에 올리는 것:
- Markdown 계획과 결과 보고
- Mermaid diagram
- source map
- config
- 작은 CSV summary
- 작은 sample figure

GitHub에 올리지 않는 것:
- 논문 PDF
- `.dat` image shard
- `.feather` source label
- checkpoint
- 대용량 prediction CSV
- 이전 scratch/test code
