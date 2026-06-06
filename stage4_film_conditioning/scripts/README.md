# Stage 4 Scripts

## English

Stage 4 scripts are added incrementally during implementation.

Added in `4-I1`:
- `_stage4_script_utils.py`: adds Stage 4 `src` and Stage 2 `src` to
  `sys.path`.
- `check_stage4_scaffold.py`: validates config, local BTC/F&G paths, Stage 2
  dependency, and primary experiment names.

Added in `4-I2`:
- `audit_stage4_context_sources.py`: audits BTC/F&G coverage and the as-of
  no-future-leakage merge policy.
- `build_stage4_context_features.py`: builds raw/normalized structured context
  features and the train-only context scaler.

Added in `4-I3`:
- `check_stage4_context_encoder.py`: validates the shared numeric context MLP
  on dummy tensors and, when available, real normalized context rows.

Added in `4-I4`:
- `check_stage4_model_shapes.py`: validates the Stage 4 concat model tensor
  path, parameter count, and real normalized context forward pass.

Updated in `4-I5`:
- `check_stage4_model_shapes.py`: now also validates `--model gating`,
  final-block gate shape, identity initialization, and real-context forward
  pass.

Added in `4-I6`:
- `check_stage4_film_layers.py`: validates gamma-only and full FiLM parameter
  generators, block-wise gamma/beta shapes, identity initialization, and
  feature-wise affine modulation on I60 block feature maps.

Updated in `4-I7`:
- `check_stage4_model_shapes.py`: now also validates `--model film_gamma` and
  `--model film_full`, block-wise FiLM parameter shapes, identity
  initialization, parameter counts, and real-context forward passes.

Updated in `4-V7`:
- `check_stage4_model_shapes.py`: now also validates
  `--model film_full_bounded_last_block`, final-block-only bounded gamma/beta
  shapes, identity initialization, and parameter count.

Updated in `4-N12-A`:
- `check_stage4_model_shapes.py`: now validates
  `--model film_full_uncertainty_gated_last_block`, including
  `modulation_gate` and frozen Stage 2 `stage2_prob_up` tensor shapes.

Added in `4-I8`:
- `run_stage4_context_model.py`: runs one Stage 4 context-conditioned training
  job. It reuses the fixed Stage 2 BTC data/image/split/normalization path,
  builds context features, attaches `batch["context"]`, and trains one of
  `concat`, `gating`, `film_gamma`, `film_full`, or
  `film_full_bounded_last_block`. N12-A also supports
  `film_full_uncertainty_gated_last_block`.
  - Smoke example:
    `python scripts/run_stage4_context_model.py --config configs/env_local.yaml --context-method concat --max-epochs 1 --max-train-rows 16 --max-validation-rows 8 --max-test-rows 8`

Added in `4-I9`:
- `evaluate_stage4_predictions.py`: reloads a Stage 4 checkpoint, calls
  `model(image, context)`, and writes prediction CSV plus classification
  metrics.
- `evaluate_stage4_trading.py`: reads Stage 4 prediction CSV and writes BTC
  long/flat and long/short trading metrics.

Added in `4-I10`:
- `generate_stage4_gradcam_context.py`: reloads a Stage 4 checkpoint and
  prediction CSV, generates predicted-class Grad-CAM with `model(image,
  context)`, and writes `samples.csv`, `modulation_summary.csv`, and
  `modulation_values.json`.
- For `concat`, the export contains normalized context values and context
  embedding summaries.
- For `gating`, it additionally exports raw gate and final gate values.
- For `film_gamma`/`film_full`, it exports block-wise gamma and beta values.
- For `film_full_bounded_last_block`, it exports final-block bounded
  gamma/beta plus raw gamma/beta values.
- For `film_full_uncertainty_gated_last_block`, it also exports the
  uncertainty `modulation_gate` and frozen Stage 2 `stage2_prob_up`.

Added in `4-I11`:
- `check_stage4_outputs.py`: checks the complete Stage 4 output bundle for one
  experiment/seed. It verifies checkpoint, training metadata, predictions,
  classification metrics, trading metrics, Grad-CAM, samples, modulation
  summary/value exports, context artifacts, and run manifest.

Added in `4-V8`:
- `analyze_stage4_seed_collapse.py`: reads existing P7/P8 prediction CSVs,
  computes default-threshold metrics, calibrates a validation threshold, applies
  it to test predictions, and writes seed-collapse/pairwise comparison tables.

`4-V9` does not add a new script. It reuses `build_stage4_context_features.py`,
`run_stage4_context_model.py`, `evaluate_stage4_predictions.py`, and
`evaluate_stage4_trading.py` with a scale-specific `experiment_suffix`.

Added in `4-N1` to `4-N4`:
- `audit_stage4_news_source.py`: audits the selected BTC headline dataset,
  date coverage, source distribution, duplicates, and Stage 4 sample overlap.
- `audit_stage4_news_alignment.py`: locks strict `t-1` publication-time policy
  and audits 7/20/60-day headline-window coverage without future leakage.
- `build_stage4_news_headline_windows.py`: builds sample-level 7/20/60-day
  headline-window documents, counts, source counts, and hashes.
- `build_stage4_news_tfidf_svd.py`: fits TF-IDF/SVD on train headline-window
  documents only, transforms all splits, and writes fixed-size
  `news_svd_7d/20d/60d` vectors plus count features.
- `build_stage4_news_context_features.py`: converts the 4-N4 TF-IDF/SVD table
  into model-ready `context_features.csv`, `context_scaler.json`, and
  report tables. The first news context vector has `102` normalized features:
  `96` SVD features plus `6` log-count features.

Added in `4-N6`:
- `run_stage4_context_model.py`, `evaluate_stage4_predictions.py`,
  `evaluate_stage4_trading.py`, `generate_stage4_gradcam_context.py`, and
  `check_stage4_outputs.py` can consume a prebuilt news context artifact when
  `context.source` is set to `prebuilt_news`.
- The prebuilt context loader reads `context_scaler.json` for feature order and
  aligns `context_features.csv` to BTC samples through `sample_index`.

Added in `4-N13-1`:
- `build_stage4_fsi_context_features.py`: reads the official OFR Financial
  Stress Index CSV, aligns it to BTC sample image end dates with a conservative
  as-of lag, and writes model-ready `context_features.csv`,
  `context_scaler.json`, `context_feature_audit.json`, and
  `context_feature_summary.csv`.
- OFR FSI is recorded as an official financial-stress / risk-off proxy, not as
  a direct RORO index. The BTC direction is not hard-coded; the FiLM/context
  model learns the relationship.

Planned next scripts:
- `summarize_stage4_results.py`

Checklist item 4-8 fixes the expected Kaggle execution order and backup
contract for these scripts.

Checklist item 4-I0 fixes that these scripts must add both Stage 4 `src` and
Stage 2 `src` to `sys.path`.

## 한국어

Stage 4 script는 구현 단계에서 순차적으로 추가합니다.

`4-I1`에서 추가한 script:
- `_stage4_script_utils.py`: Stage 4 `src`와 Stage 2 `src`를 `sys.path`에
  추가합니다.
- `check_stage4_scaffold.py`: config, local BTC/F&G path, Stage 2 dependency,
  primary experiment name을 검증합니다.

`4-I2`에서 추가한 script:
- `audit_stage4_context_sources.py`: BTC/F&G coverage와 as-of no-future-leakage
  merge policy를 감사합니다.
- `build_stage4_context_features.py`: raw/normalized structured context feature와
  train-only context scaler를 생성합니다.

`4-I3`에서 추가한 script:
- `check_stage4_context_encoder.py`: shared numeric context MLP를 dummy tensor와,
  가능하면 실제 normalized context row로 검증합니다.

`4-I4`에서 추가한 script:
- `check_stage4_model_shapes.py`: Stage 4 concat model의 tensor path,
  parameter count, 실제 normalized context forward pass를 검증합니다.

`4-I5`에서 수정한 script:
- `check_stage4_model_shapes.py`: `--model gating`, final-block gate shape,
  identity initialization, 실제 context forward pass도 검증합니다.

`4-I6`에서 추가한 script:
- `check_stage4_film_layers.py`: gamma-only/full FiLM parameter generator,
  block별 gamma/beta shape, identity initialization, I60 block feature map의
  feature-wise affine modulation을 검증합니다.

`4-I7`에서 수정한 script:
- `check_stage4_model_shapes.py`: `--model film_gamma`, `--model film_full`,
  block별 FiLM parameter shape, identity initialization, parameter count,
  실제 context forward pass도 검증합니다.

`4-V7`에서 수정한 script:
- `check_stage4_model_shapes.py`: `--model film_full_bounded_last_block`,
  final-block-only bounded gamma/beta shape, identity initialization,
  parameter count를 검증합니다.

`4-N12-A`에서 수정한 script:
- `check_stage4_model_shapes.py`: `--model film_full_uncertainty_gated_last_block`,
  `modulation_gate`, frozen Stage 2 `stage2_prob_up` shape까지 검증합니다.

`4-I8`에서 추가한 script:
- `run_stage4_context_model.py`: Stage 4 context-conditioned training job 하나를
  실행합니다. 고정된 Stage 2 BTC data/image/split/normalization 경로를 재사용하고,
  context feature를 생성한 뒤 `batch["context"]`를 붙여 `concat`, `gating`,
  `film_gamma`, `film_full`, `film_full_bounded_last_block`,
  `film_full_uncertainty_gated_last_block` 중 하나를 학습합니다.
  - Smoke 예시:
    `python scripts/run_stage4_context_model.py --config configs/env_local.yaml --context-method concat --max-epochs 1 --max-train-rows 16 --max-validation-rows 8 --max-test-rows 8`

`4-I9`에서 추가한 script:
- `evaluate_stage4_predictions.py`: Stage 4 checkpoint를 다시 로드하고
  `model(image, context)`를 호출해 prediction CSV와 classification metrics를
  저장합니다.
- `evaluate_stage4_trading.py`: Stage 4 prediction CSV를 읽고 BTC long/flat,
  long/short trading metrics를 저장합니다.

`4-I10`에서 추가한 script:
- `generate_stage4_gradcam_context.py`: Stage 4 checkpoint와 prediction CSV를
  다시 로드하고 `model(image, context)` 기준 predicted-class Grad-CAM을 만든 뒤
  `samples.csv`, `modulation_summary.csv`, `modulation_values.json`를 저장합니다.
- `concat`은 normalized context 값과 context embedding summary를 저장합니다.
- `gating`은 raw gate와 최종 gate 값까지 저장합니다.
- `film_gamma`/`film_full`은 block별 gamma/beta 값을 저장합니다.
- `film_full_bounded_last_block`은 final-block bounded gamma/beta와 raw
  gamma/beta 값을 저장합니다.
- `film_full_uncertainty_gated_last_block`은 final-block bounded gamma/beta에
  더해 `modulation_gate`와 `stage2_prob_up`을 저장합니다.
- `film_full_confidence_gated_last_block`도 final-block bounded gamma/beta에
  더해 `modulation_gate`와 `stage2_prob_up`을 저장합니다.

`4-I11`에서 추가한 script:
- `check_stage4_outputs.py`: 한 experiment/seed의 Stage 4 output bundle 전체를
  확인합니다. Checkpoint, training metadata, prediction, classification metric,
  trading metric, Grad-CAM, samples, modulation summary/value export, context
  artifact, run manifest를 검사합니다.

`4-V8`에서 추가한 script:
- `analyze_stage4_seed_collapse.py`: 기존 P7/P8 prediction CSV를 읽고,
  default-threshold metric, validation threshold calibration, test 적용 결과,
  seed-collapse/pairwise comparison table을 저장합니다.

`4-V9`는 새 script를 추가하지 않습니다. Scale-specific `experiment_suffix`를
사용해 `build_stage4_context_features.py`, `run_stage4_context_model.py`,
`evaluate_stage4_predictions.py`, `evaluate_stage4_trading.py`를 재사용합니다.

`4-N1`부터 `4-N4`에서 추가한 script:
- `audit_stage4_news_source.py`: 선택한 BTC headline dataset, date coverage,
  source distribution, duplicate, Stage 4 sample overlap을 감사합니다.
- `audit_stage4_news_alignment.py`: strict `t-1` publication-time policy를
  고정하고 future leakage 없이 7/20/60-day headline-window coverage를 감사합니다.
- `build_stage4_news_headline_windows.py`: sample-level 7/20/60-day
  headline-window document, count, source count, hash를 만듭니다.
- `build_stage4_news_tfidf_svd.py`: train headline-window document에만
  TF-IDF/SVD를 fit하고, 모든 split을 transform해서 고정 길이
  `news_svd_7d/20d/60d` vector와 count feature를 저장합니다.
- `build_stage4_news_context_features.py`: 4-N4 TF-IDF/SVD table을
  model-ready `context_features.csv`, `context_scaler.json`, report table로
  변환합니다. 첫 news context vector는 `96`개 SVD feature와 `6`개 log-count
  feature로 된 `102`개 normalized feature입니다.

`4-N6`에서 보강한 실행 흐름:
- `context.source=prebuilt_news`일 때 `run_stage4_context_model.py`,
  `evaluate_stage4_predictions.py`, `evaluate_stage4_trading.py`,
  `generate_stage4_gradcam_context.py`, `check_stage4_outputs.py`가 N5에서 만든
  prebuilt news context artifact를 읽을 수 있습니다.
- prebuilt context loader는 `context_scaler.json`에서 feature order를 읽고,
  `sample_index`로 `context_features.csv`와 BTC sample을 정렬합니다.

다음 예정 script:
- `summarize_stage4_results.py`

4-8에서 이 script들의 Kaggle 실행 순서와 backup 계약을 고정했습니다.

4-I0에서 이 script들이 Stage 4 `src`와 Stage 2 `src`를 모두 `sys.path`에
추가해야 한다고 고정했습니다.
